"""学習前のモデル構造をend-to-endで検査する。

既存questionの出力が他questionの追加・位置変更で変わらないこと、複数branchを同時に
評価したときの実行時間の増え方、選択肢の追加と並べ替えが出力確率へ与える差を測定する
（並べ替えは学習前後のreadoutで比較する）。
"""

from __future__ import annotations

import itertools
import random
import time

import torch

from baselines.independent_scorer import IndependentScorer
from baselines.naive_reencode import naive_reencode_forward
from data.ag_news import eval_split, to_packed_example, train_subset
from experiments.train import _build_examples, train_readout
from model.forward import JevModel
from model.packer import Branch, PackedRequest

STATE = "Oil prices surged 5% today after OPEC announced production cuts."
CHOICE_OPTIONS = ["World", "Sports", "Business", "Sci-Tech"]


def _timeit(fn, repeats: int = 3) -> float:
    """`fn()`を`repeats`回呼んで、最小実行時間（秒）を返す（ノイズに強い代表値としてminを採用）。"""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def run_isolation_check(model: JevModel, atol: float = 1e-3) -> bool:
    """questionを追加・挿入しても、既存questionの答えが変わらないこと（isolation）をend-to-endで確認する。

    `tree_mask.py`のsmoke checkではmask/position_idというbookkeepingレベルでこの性質を確認したが、
    ここでは実際にbackbone+readoutをforwardして、出てくる確率が不変かまで確認する。

    注意：ここでの一致は「厳密なビット一致」ではなく`atol`までの近似一致で判定する。
    層ごとの隠れ状態を調べたところ、embedding層（token id・position idを引いた直後）は
    常にビット単位で完全一致する一方、Attention層を1つ通るごとに1e-6オーダーの浮動小数点誤差が生まれ、
    24層を経て最終的に1e-4オーダーまで蓄積することがわかった。これはmask・position idの
    ロジックバグではない（embedding層が完全一致＝token化とpositionの割り当ては正しい）——
    eager attentionがmaskで可視性を絞っても、QK^Tの計算自体は系列全体`(T, T)`に対して
    密に行われるため、行列積の内部縮約順序がテンソル全体の形状に依存し、丸め誤差の経路が
    わずかに変わることに起因する（密attention実装特有の数値的非結合性であり、設計の欠陥ではない）。
    """
    branch_a = Branch(kind="choice", instructions="この記事のトピックを選べ。", options=CHOICE_OPTIONS)
    branch_b = Branch(kind="noul", instructions="この記事は経済に関するものか？")
    branch_c = Branch(kind="score", instructions="緊急度を評価せよ。", options=["低い", "中程度", "高い"])

    result_alone = model.forward(PackedRequest(state=STATE, branches=[branch_a]))[0]
    result_first = model.forward(PackedRequest(state=STATE, branches=[branch_a, branch_b, branch_c]))[0]
    result_middle = model.forward(PackedRequest(state=STATE, branches=[branch_b, branch_a, branch_c]))[1]
    result_last = model.forward(PackedRequest(state=STATE, branches=[branch_b, branch_c, branch_a]))[2]

    print(f"branch_aのみ:              {result_alone['probabilities']}")
    print(f"branch_aが先頭(他2つ付加): {result_first['probabilities']}")
    print(f"branch_aが中央:            {result_middle['probabilities']}")
    print(f"branch_aが末尾:            {result_last['probabilities']}")

    variants = [result_alone, result_first, result_middle, result_last]
    max_diff = max(
        abs(a - b)
        for r1, r2 in zip(variants, variants[1:])
        for a, b in zip(r1["probabilities"], r2["probabilities"])
    )
    all_close = max_diff < atol
    print(f"最大差分: {max_diff:.3e}（許容誤差 atol={atol:.0e}）")

    if all_close:
        print(f"isolation: 予測通り。他のquestionを何個・どこに追加・挿入しても、"
              f"branch_aの確率は誤差{atol:.0e}の範囲内で不変（浮動小数点の丸め誤差を除き一致）。")
    else:
        print("isolation: 予測と食い違い。他のquestionの追加・挿入でbranch_aの確率が無視できない大きさで"
              "変わっている——mask・position idの実装にバグの可能性が高い。")

    return all_close


def run_p1(model: JevModel, branch_counts: list[int]) -> list[tuple[int, float, float]]:
    """questionの数Nを変えて、tree mask版（1回のforward）とnaive re-encode版（N回のforward）の実行時間を測る。"""
    results = []
    for n in branch_counts:
        request = PackedRequest(
            state=STATE,
            branches=[
                Branch(kind="choice", instructions=f"question {i}: トピックを選べ。", options=CHOICE_OPTIONS)
                for i in range(n)
            ],
        )

        # warmup（初回呼び出しのオーバーヘッドを timeit の対象から外す）
        model.forward(request)
        naive_reencode_forward(model, request)

        tree_mask_time = _timeit(lambda: model.forward(request))
        naive_time = _timeit(lambda: naive_reencode_forward(model, request))

        results.append((n, tree_mask_time, naive_time))
        print(f"N={n:3d}  tree_mask={tree_mask_time*1000:8.1f}ms  naive_reencode={naive_time*1000:8.1f}ms  "
              f"naive/tree={naive_time/tree_mask_time:.2f}x")

    return results


def run_p2(model: JevModel, scorer: IndependentScorer) -> None:
    """無関係な選択肢を1つ追加したとき、pointer方式(JevModel)とindependent scorerでlog-oddsがどう動くか比較する。"""
    base_options = ["World", "Sports"]
    extra_option = "Irrelevant Extra Option"

    def jev_log_odds(options: list[str]) -> float:
        request = PackedRequest(
            state=STATE,
            branches=[Branch(kind="choice", instructions="この記事のトピックを選べ。", options=options)],
        )
        result = model.forward(request)[0]
        p = result["probabilities"]
        return float(torch.log(torch.tensor(p[0]) / torch.tensor(p[1])))

    def scorer_log_odds(options: list[str]) -> float:
        z = scorer.logits(STATE, options)
        return float(z[0] - z[1])

    jev_before = jev_log_odds(base_options)
    jev_after = jev_log_odds(base_options + [extra_option])
    jev_diff = jev_after - jev_before

    scorer_before = scorer_log_odds(base_options)
    scorer_after = scorer_log_odds(base_options + [extra_option])
    scorer_diff = scorer_after - scorer_before

    print(f"JevModel (pointer方式):     log-odds before={jev_before:+.6f}  after={jev_after:+.6f}  "
          f"diff={jev_diff:+.6f}")
    print(f"IndependentScorer (対照): log-odds before={scorer_before:+.6f}  after={scorer_after:+.6f}  "
          f"diff={scorer_diff:+.6f}")

    return jev_diff, scorer_diff


P3_INSTRUCTIONS = "この記事のトピックを、選択肢の中から1つ選べ。"


def _p3_cases(n_ag_states: int = 3) -> list[tuple[str, str, list[str]]]:
    """P3測定ケース（state × 選択肢セット）の一覧を作る。

    stateは合成テキスト1つ＋AG Newsのheld-out（test split）記事3つ。選択肢セットは
    短いラベルのみ4択（`CHOICE_OPTIONS`）と、criteria説明付き4択の2種類。計8ケース。
    測定に使うstateは訓練データ（train split）と重ならない。
    """
    ds = eval_split(64)
    label_names = ds.features["label"].names
    criteria_options = to_packed_example(ds[0]["text"], ds[0]["label"], label_names).request.branches[0].options

    states = [("synthetic", STATE)] + [(f"AGNews#{i}", ds[i]["text"]) for i in range(n_ag_states)]
    option_sets = [("short4", CHOICE_OPTIONS), ("criteria4", criteria_options)]
    return [(f"{sname}×{oname}", text, options) for sname, text in states for oname, options in option_sets]


def _measure_order_sensitivity(model: JevModel, cases: list[tuple[str, str, list[str]]], atol: float) -> dict:
    """各ケースで選択肢の全順列をforwardし、同一選択肢（内容で紐付け）の確率の変化幅を集計する。"""
    rows = []
    all_ranges: list[float] = []

    for name, state, options in cases:
        probs: dict[str, list[float]] = {opt: [] for opt in options}
        top1: list[int] = []
        confidence: list[float] = []

        for perm in itertools.permutations(options):
            branch = Branch(kind="choice", instructions=P3_INSTRUCTIONS, options=list(perm))
            result = model.forward(PackedRequest(state=state, branches=[branch]))[0]
            ps = result["probabilities"]
            for opt, p in zip(perm, ps):
                probs[opt].append(p)
            top1.append(ps.index(max(ps)))
            confidence.append(max(ps))

        ranges = [max(v) - min(v) for v in probs.values()]
        row = {
            "case": name,
            "mean_range": sum(ranges) / len(ranges),
            "max_range": max(ranges),
            "top1_stability": sum(a == top1[0] for a in top1) / len(top1),
            "mean_confidence": sum(confidence) / len(confidence),
        }
        rows.append(row)
        all_ranges.extend(ranges)
        print(f"  {name:24s} range mean={row['mean_range']:.4f} max={row['max_range']:.4f}  "
              f"top1安定率={row['top1_stability']:.2f}  平均確信度={row['mean_confidence']:.4f}")

    pooled = {
        "n_options": len(all_ranges),
        "mean_range": sum(all_ranges) / len(all_ranges),
        "median_range": sorted(all_ranges)[len(all_ranges) // 2],
        "max_range": max(all_ranges),
        "exceed_rate": sum(r > atol for r in all_ranges) / len(all_ranges),
        "mean_confidence": sum(r["mean_confidence"] for r in rows) / len(rows),
    }
    print(f"  --> 統計（選択肢{pooled['n_options']}個）: range mean={pooled['mean_range']:.4f} "
          f"median={pooled['median_range']:.4f} max={pooled['max_range']:.4f} "
          f"atol({atol:.0e})超え率={pooled['exceed_rate']:.2f}")
    return {"rows": rows, **pooled}


def run_p3_order_sensitivity(model: JevModel, atol: float = 1e-3, train_size: int = 200) -> dict:
    """選択肢の内容・個数を変えず並び順だけを変えたとき、出力確率が動くかを学習前後で比較する。

    学習前のreadoutはランダム初期化で確率が0/1に飽和しやすく、大きな差分だけを見ると
    飽和のアーティファクトと構造的な性質の区別がつかない。そこでAG Newsで軽く訓練した
    readoutでも同じ測定を行い、確信度（平均最大確率）と変化幅がどう変わるかで切り分ける。
    """
    cases = _p3_cases()
    print(f"ケース: {len(cases)}（state×選択肢セット）、各ケース全順列（4!=24通り）を測定")

    print("\n--- 学習前（ランダム初期化のreadout） ---")
    before = _measure_order_sensitivity(model, cases, atol)

    print(f"\n--- 軽い訓練（AG News train split {train_size}件×1 epoch、readoutのみ・backbone凍結） ---")
    random.seed(0)
    examples = _build_examples(train_subset(train_size))
    train_readout(model, examples, epochs=1, log_every=100)

    print("\n--- 学習後 ---")
    after = _measure_order_sensitivity(model, cases, atol)

    print(f"\n確信度（全ケース平均の最大確率）: 学習前={before['mean_confidence']:.4f} -> "
          f"学習後={after['mean_confidence']:.4f}")
    print(f"並び順による確率変化幅: 学習前 mean={before['mean_range']:.4f}/max={before['max_range']:.4f} -> "
          f"学習後 mean={after['mean_range']:.4f}/max={after['max_range']:.4f}（ノイズ床 atol={atol:.0e}）")

    ok = after["max_range"] > atol
    if ok and after["mean_range"] > atol:
        print("P3: 並び順で確率は動いた。学習後も平均・最大ともにatolを明確に上回る——未学習の飽和に"
              "依らない、branch内のcausal attention（後ろの選択肢だけ前の選択肢を参照できる非対称性）に"
              "由来する構造的な性質として扱える。")
    elif ok:
        print("P3: 並び順で確率は動いた（最大値はatol超え）。ただし平均はatol未満で、動くのは一部の"
              "選択肢・ケースに限られる。")
    else:
        print("P3: 学習後は並び順で確率がほぼ動かない——学習前の大きな差分は未学習readoutの飽和に"
              "よるアーティファクトだった可能性が高い。")
    return {"before": before, "after": after, "ok": ok}


def _main() -> None:
    # 再現性: readoutの初期化と訓練時のシャッフルを固定する
    torch.manual_seed(0)
    random.seed(0)
    model = JevModel()

    print("=== isolation: questionを追加・挿入しても既存questionは不変か ===")
    isolation_ok = run_isolation_check(model)

    print("\n=== P1: questionを増やしたときのforward時間 ===")
    p1_results = run_p1(model, branch_counts=[1, 2, 4, 8, 16])

    n0, tree0, naive0 = p1_results[0]
    n_last, tree_last, naive_last = p1_results[-1]
    tree_growth = tree_last / tree0
    naive_growth = naive_last / naive0
    print(f"\nN={n0}->N={n_last}: tree_mask time growth = {tree_growth:.2f}x, "
          f"naive_reencode time growth = {naive_growth:.2f}x")
    if tree_growth < naive_growth:
        print("P1: 予測通り、tree mask版の方が緩やかに増えている。")
    else:
        print("P1: 予測と食い違い——tree mask版の方が急に増えている。"
              "考察: eager attentionはmaskで可視性を絞っても、QK^Tの計算自体は"
              "パックした系列全体 T×T に対して密に行われる（マスクはsoftmax前に足すだけで、"
              "計算そのものを間引かない）。そのためtree mask版の1回のforwardはT=state+ΣQ_iに対して"
              "O(T^2)、naive re-encodeはbranchごとの短い系列に対するO((S+Q_i)^2)をN回に分けて行うだけなので、"
              "むしろnaiveの方が総計算量で有利になり得る。stateのKVを1回だけ計算し、"
              "各branchへ追記する専用実装なら異なる計算量特性を得られるが、今回のeager dense"
              "アテンション実装はその方式を採用していない。")

    print("\n=== P2: 無関係な選択肢を追加したときのlog-odds ===")
    scorer = IndependentScorer()
    jev_diff, scorer_diff = run_p2(model, scorer)

    print(f"\nJevModel diff = {jev_diff:+.6f} (0から離れているほど、listwise相互作用が効いている)")
    print(f"IndependentScorer diff = {scorer_diff:+.6f} (理論上ちょうど0のはず)")

    p2_ok = abs(scorer_diff) < 1e-9 and abs(jev_diff) > 1e-6
    if p2_ok:
        print("P2: 予測通り。IndependentScorerのlog-oddsは不変（0）、JevModel（pointer方式）は動いた。")
    else:
        print("P2: 予測と食い違い。実装（mask・readoutの配線）を見直す必要がある。")

    print("\n=== P3: 選択肢の並び順だけを変えたときの出力確率（学習前後比較） ===")
    p3_summary = run_p3_order_sensitivity(model)

    print(f"\n=== まとめ === isolation: {'OK' if isolation_ok else 'NG'} / "
          f"P1: {'OK' if tree_growth < naive_growth else 'NG'} / P2: {'OK' if p2_ok else 'NG'} / "
          f"P3: {'OK' if p3_summary['ok'] else 'NG'}")

if __name__ == "__main__":
    _main()
