"""Score branchの劣化原因を、仮説ごとに分離して測る診断スクリプト。

5データセット混合訓練（sst5をScore教師データとして追加）で、公式比較のScore一致は
5/8 -> 1/8に後退した。考えられる原因は少なくとも3つあり、互いに独立している:

  (H1) レベル数Kのミスマッチ。sst5は常にK=5で訓練したが、公式のseverity/frustrationは
       K=3、report_qualityはK=4。
  (H2) 選択肢テキストの形式のミスマッチ。sst5の選択肢は`very negative`のような短い
       レベル名だが、公式は`Cosmetic; no impact to functionality`のような説明節。
       pointer readoutはstateの隠れ状態と選択肢の隠れ状態を突き合わせるので、
       選択肢側の表現形式が変わると効かなくなりうる。
  (H3) ドメインのミスマッチ。映画レビューの感情極性 vs バグの深刻度/報告品質。

さらに、argmax一致率だけでは「順序の信号が壊れている」のか「順序は保たれていて
絶対位置（校正）だけずれている」のかを区別できない。公式Scoreドキュメントの
severity表は同じ質問を深刻度の異なる5つのstateに当てるので、正解の順序が既知。
そこでargmax一致に加えてKendall tau（順位相関）も測る。

使い方:
    python3 -m experiments.score_diagnose checkpoints/pointer_diverse5_e3.pt
"""

from __future__ import annotations

import math
import random
import sys

import torch
from scipy.stats import kendalltau

from data.sst5 import INSTRUCTION_VARIANTS, LEVEL_ORDER, load_sst5, to_packed_example
from legacy.experiments.official_compare import SEVERITY_LEVELS
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest

# 公式Scoreドキュメントのseverity表: (state, 公式score)。公式scoreは0基準の位置。
SEVERITY_STATES: list[tuple[str, float]] = [
    ("The export button is misaligned by a few pixels on the settings page.", 0.0),
    (
        "The PDF export button does nothing when clicked. I can still export to CSV and "
        "convert it myself, but that takes ages.",
        1.0,
    ),
    (
        "Export to PDF fails with a spinner that never finishes. Some of our team say CSV "
        "export still works for them, others say it fails too.",
        1.12,
    ),
    (
        "The export button crashes the settings page in Safari. It works in Chrome, but a "
        "few of our customers only use Safari.",
        1.3,
    ),
    ("Nobody on our team can log in since this morning. We get a 500 error on every attempt.", 2.0),
]

SEVERITY_QUESTION = "How severe is the reported issue?"

# 2x2の設計行列: 選択肢テキスト（短いレベル名 / 説明節）× レベル数（K=3 / K=5）。
# H1（K）とH2（テキスト形式）を分離するため、片方だけを変えた4条件を作る。
SHORT_K3 = ["minor", "moderate", "critical"]
SHORT_K5 = ["very minor", "minor", "moderate", "severe", "very severe"]
LONG_K3 = list(SEVERITY_LEVELS)  # 公式そのまま
LONG_K5 = [
    "Cosmetic; no impact to functionality",
    "Minor annoyance; the feature still works",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue for some users; no workaround exists",
    "Blocking issue for everyone; the product is unusable",
]

VARIANTS: list[tuple[str, list[str]]] = [
    ("短いレベル名 K=3", SHORT_K3),
    ("短いレベル名 K=5", SHORT_K5),
    ("説明節 K=3 (公式そのまま)", LONG_K3),
    ("説明節 K=5", LONG_K5),
]


def _positions(model: JevModel, options: list[str]) -> tuple[list[float], list[int]]:
    """severity表の5stateに同じScore質問を当て、0基準の位置とargmaxレベルを返す。"""
    branch = Branch(kind="score", instructions=SEVERITY_QUESTION, options=options)
    positions: list[float] = []
    argmaxes: list[int] = []
    for state, _ in SEVERITY_STATES:
        result = model.forward(PackedRequest(state=state, branches=[branch]))[0]
        positions.append(result["score"] - 1.0)  # score_expectationは1基準
        argmaxes.append(int(max(range(len(options)), key=lambda i: result["probabilities"][i])))
    return positions, argmaxes


def diagnose_option_format(model: JevModel) -> None:
    """H1（K）とH2（選択肢テキスト形式）を分離する2x2診断。"""
    print("\n=== H1/H2: 選択肢の形式とレベル数を変えたときのseverity表 ===")
    official = [score for _, score in SEVERITY_STATES]
    print(f"公式score（0基準・正解順序）: {official}")

    for name, options in VARIANTS:
        positions, argmaxes = _positions(model, options)
        tau, p_value = kendalltau(official, positions)
        # 公式はK=3のときL0,L1,L1,L1,L2。K=5条件では公式argmaxが定義できないので順位相関のみ。
        line = f"  {name:<26} tau={tau:+.3f} (p={p_value:.3f})  位置=[" + ", ".join(
            f"{p:.2f}" for p in positions
        ) + "]"
        print(line)
        print(f"  {'':<26} argmax={argmaxes}  (公式K=3のargmaxは[0,1,1,1,2])")


def diagnose_monotonicity(model: JevModel) -> None:
    """順序信号は残っているのか（校正のズレだけなのか）を切り分ける。"""
    print("\n=== 順序信号 vs 校正: 公式そのままの条件で詳しく見る ===")
    positions, argmaxes = _positions(model, LONG_K3)
    official = [score for _, score in SEVERITY_STATES]

    tau_all, _ = kendalltau(official, positions)
    # 1つ目（cosmetic, 公式0.0）を除いた実バグ4件だけの順序相関。
    tau_bugs, _ = kendalltau(official[1:], positions[1:])
    print(f"  全5state     : tau={tau_all:+.3f}")
    print(f"  実バグ4件のみ: tau={tau_bugs:+.3f}  （cosmeticの1件を除外）")
    print(f"  位置の範囲   : min={min(positions):.3f} max={max(positions):.3f} "
          f"幅={max(positions) - min(positions):.3f} （公式の幅=2.000, K=3の理論上の幅=2.0）")
    print(f"  argmax       : {argmaxes}")


def diagnose_in_domain(model: JevModel, n: int = 200) -> None:
    """H3（ドメイン）の対照: 訓練ドメインであるsst5のheld-outでの順序性能。"""
    print(f"\n=== H3対照: sst5 held-out {n}件での順序性能（訓練ドメイン） ===")
    dataset = load_sst5("validation")
    rng = random.Random(0)
    golds: list[int] = []
    preds: list[float] = []
    for row in dataset.select(range(min(n, len(dataset)))):
        example = to_packed_example(row["text"], row["label"], rng)
        result = model.forward(example.request)[0]
        golds.append(example.label)
        preds.append(result["score"] - 1.0)

    tau, p_value = kendalltau(golds, preds)
    mae = sum(abs(g - p) for g, p in zip(golds, preds)) / len(golds)
    print(f"  tau={tau:+.3f} (p={p_value:.2e})  MAE(level)={mae:.3f}")
    print(f"  予測位置の範囲: min={min(preds):.3f} max={max(preds):.3f} "
          f"（gold範囲=0..4）")


def diagnose_k_generalization(model: JevModel, n: int = 200) -> None:
    """H1（レベル数K）とH3（ドメイン）を分離する。

    テキストはsst5のまま（=訓練ドメイン）で、選択肢だけをgoldを含む連続部分区間に狭める。
    ドメインが同じままKだけ変わるので、ここで性能が崩れればK=3への一般化失敗（H1）、
    崩れなければ残る容疑はドメイン（H3）だけになる。
    """
    print(f"\n=== H1 vs H3: sst5のテキストのままKだけ変える（各{n}件） ===")
    dataset = load_sst5("validation")
    rows = list(dataset.select(range(min(n, len(dataset)))))

    for k in (2, 3, 4, 5):
        rng = random.Random(0)
        golds: list[int] = []
        preds: list[float] = []
        for row in rows:
            label = int(row["label"])
            # goldを含む長さkの連続窓を一様サンプルし、窓内での相対位置を新しいgoldにする。
            lowest = max(0, label - k + 1)
            highest = min(label, len(LEVEL_ORDER) - k)
            start = rng.randint(lowest, highest)
            options = LEVEL_ORDER[start:start + k]
            branch = Branch(kind="score", instructions=INSTRUCTION_VARIANTS[0], options=options)
            result = model.forward(PackedRequest(state=row["text"], branches=[branch]))[0]
            golds.append(label - start)
            preds.append(result["score"] - 1.0)

        tau, p_value = kendalltau(golds, preds)
        mae = sum(abs(g - p) for g, p in zip(golds, preds)) / len(golds)
        print(f"  K={k}: tau={tau:+.3f} (p={p_value:.2e})  MAE={mae:.3f}  "
              f"予測幅={max(preds) - min(preds):.3f}（使える幅={k - 1}）  "
              f"範囲={min(preds):.2f}..{max(preds):.2f}")


def diagnose_pair_resolution(model: JevModel, n: int = 200) -> None:
    """`diagnose_k_generalization`の交絡を外す: Kを2に固定し、レベル距離dだけを振る。

    連続窓でKを狭めると選択肢間のレベル距離も同時に縮むので、「K=2でtau≈0」は
    K一般化の失敗とレベル解像度不足のどちらでも説明できてしまう（MAEは約1.0レベル）。
    ここではK=2に固定したまま、正解から距離dだけ離れた選択肢1つと二択にする。

    - dが大きいほど正解率が上がるなら、Kではなく解像度が制約（K一般化は問題ではない）。
    - dによらず正解率が0.5付近なら、二択という形式そのもの（=K）が壊れている。
    """
    print(f"\n=== 交絡の除去: K=2固定・レベル距離dを振る（各最大{n}件） ===")
    dataset = load_sst5("validation")
    rows = list(dataset.select(range(min(n * 3, len(dataset)))))

    for d in (1, 2, 3, 4):
        rng = random.Random(0)
        correct = 0
        total = 0
        for row in rows:
            label = int(row["label"])
            candidates = [x for x in (label - d, label + d) if 0 <= x < len(LEVEL_ORDER)]
            if not candidates:
                continue  # この正解レベルからは距離dの選択肢が作れない
            other = rng.choice(candidates)
            lo, hi = min(label, other), max(label, other)
            options = [LEVEL_ORDER[lo], LEVEL_ORDER[hi]]
            branch = Branch(kind="score", instructions=INSTRUCTION_VARIANTS[0], options=options)
            result = model.forward(PackedRequest(state=row["text"], branches=[branch]))[0]
            predicted = int(result["probabilities"][1] > result["probabilities"][0])
            correct += int(predicted == (0 if label == lo else 1))
            total += 1
            if total >= n:
                break

        print(f"  d={d}: accuracy={correct / total:.3f}  (n={total}, チャンス=0.500)")


def diagnose_extremes(model: JevModel, n_per_class: int = 60) -> None:
    """レイアウト（K）と中身（意味の判別）を、同じテキスト・同じ意味距離で切り分ける。

    最も離れた2レベル（very negative / very positive）の例だけを使い、
      (a) K=5の全選択肢を出して予測scoreを取り、very positive側のscoreが
          very negative側より高くなる割合（順位AUC）
      (b) 同じテキストに対し、選択肢を`very negative`と`very positive`の二択にしたときの正解率
    を比べる。意味距離は両方とも最大で同じなので、差が出るならレイアウト（K）の問題、
    両方とも0.5付近なら、そもそも極性の判別ができていない（readoutの中身の問題）。
    """
    print(f"\n=== レイアウト vs 中身: 最遠ペア（very negative / very positive）で比較 ===")
    dataset = load_sst5("validation")
    negatives: list[str] = []
    positives: list[str] = []
    for row in dataset:
        if int(row["label"]) == 0 and len(negatives) < n_per_class:
            negatives.append(row["text"])
        elif int(row["label"]) == 4 and len(positives) < n_per_class:
            positives.append(row["text"])
        if len(negatives) >= n_per_class and len(positives) >= n_per_class:
            break

    full_branch = Branch(kind="score", instructions=INSTRUCTION_VARIANTS[0], options=LEVEL_ORDER)
    pair_options = [LEVEL_ORDER[0], LEVEL_ORDER[-1]]
    pair_branch = Branch(kind="score", instructions=INSTRUCTION_VARIANTS[0], options=pair_options)

    def _full_score(text: str) -> float:
        return model.forward(PackedRequest(state=text, branches=[full_branch]))[0]["score"]

    def _pair_pick(text: str) -> int:
        probs = model.forward(PackedRequest(state=text, branches=[pair_branch]))[0]["probabilities"]
        return int(probs[1] > probs[0])

    neg_scores = [_full_score(t) for t in negatives]
    pos_scores = [_full_score(t) for t in positives]
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos_scores
        for n in neg_scores
    )
    auc = wins / (len(pos_scores) * len(neg_scores))

    pair_correct = sum(1 - _pair_pick(t) for t in negatives) + sum(_pair_pick(t) for t in positives)
    pair_accuracy = pair_correct / (len(negatives) + len(positives))

    print(f"  (a) K=5の予測scoreによる順位AUC : {auc:.3f}  (チャンス=0.500)")
    print(f"  (b) K=2（最遠ペア二択）の正解率  : {pair_accuracy:.3f}  (チャンス=0.500)")
    print(f"      n={len(negatives)}+{len(positives)}, "
          f"K=5のscore平均: very negative={sum(neg_scores) / len(neg_scores):.3f} / "
          f"very positive={sum(pos_scores) / len(pos_scores):.3f}")


def diagnose_option_bias(model: JevModel, n: int = 150) -> None:
    """「選択肢固有のバイアスが支配的で、テキスト依存の信号は微小」という診断を直接確かめる。

    - 最遠ペアの二択で、モデルがどちらの選択肢を選ぶかの偏り。テキストに依存して選んで
      いるなら正解ラベルの分布（約50/50）に近づくはず。片側に張り付くならバイアス支配。
    - sst5の多数派ベースライン（常に最頻レベルを答えたときの正解率）。Score readoutの
      正解率がこれを超えていなければ、そもそもタスクを学習できていない。
    """
    print(f"\n=== 選択肢バイアス vs テキスト依存 ===")
    dataset = load_sst5("validation")
    rows = list(dataset.select(range(min(n, len(dataset)))))

    counts = [0] * len(LEVEL_ORDER)
    for row in rows:
        counts[int(row["label"])] += 1
    majority = max(counts) / len(rows)
    print(f"  sst5 validation {len(rows)}件のレベル分布: {counts}")
    print(f"  多数派ベースライン（常に最頻レベル）: accuracy={majority:.3f}")

    pair_options = [LEVEL_ORDER[0], LEVEL_ORDER[-1]]
    branch = Branch(kind="score", instructions=INSTRUCTION_VARIANTS[0], options=pair_options)
    picks = [0, 0]
    extremes = [row for row in rows if int(row["label"]) in (0, 4)]
    for row in extremes:
        probs = model.forward(PackedRequest(state=row["text"], branches=[branch]))[0]["probabilities"]
        picks[int(probs[1] > probs[0])] += 1
    gold_high = sum(1 for row in extremes if int(row["label"]) == 4)
    print(f"  最遠ペア二択 n={len(extremes)}: モデルの選択 {pair_options[0]!r}={picks[0]} / "
          f"{pair_options[1]!r}={picks[1]}")
    print(f"                          正解の内訳 {pair_options[0]!r}={len(extremes) - gold_high} / "
          f"{pair_options[1]!r}={gold_high}")


# 内容を持たないstate。これらに対する出力は「テキストに依存しない選択肢固有の好み」そのもの。
NULL_STATES = ("N/A", "...")


def _log_probs(model: JevModel, state: str, branch: Branch) -> list[float]:
    probs = model.forward(PackedRequest(state=state, branches=[branch]))[0]["probabilities"]
    return [math.log(max(p, 1e-12)) for p in probs]


def _null_log_probs(model: JevModel, branch: Branch) -> list[float]:
    """内容のないstateでの平均log確率 = この選択肢集合に対する素のバイアス。"""
    per_state = [_log_probs(model, state, branch) for state in NULL_STATES]
    return [sum(col) / len(col) for col in zip(*per_state)]


def _calibrated(log_probs: list[float], bias: list[float]) -> list[float]:
    """バイアスを引いて再正規化した確率。

    校正後のlogitは`z(state)_i - z(null)_i`。softmaxは定数シフトに不変なので、
    logitの代わりにlog確率の差を使っても厳密に同じ分布が得られる（forward内部に触れずに済む）。
    """
    adjusted = [lp - b for lp, b in zip(log_probs, bias)]
    highest = max(adjusted)
    weights = [math.exp(a - highest) for a in adjusted]
    total = sum(weights)
    return [w / total for w in weights]


def _expectation(probs: list[float]) -> float:
    """0基準の期待位置。"""
    return sum(i * p for i, p in enumerate(probs))


def diagnose_null_calibration(model: JevModel, n: int = 150) -> None:
    """S1: 選択肢固有バイアスを推論時に引き算して、測定した病理が実際に消えるかを見る。

    `diagnose_option_bias`で、最遠ペア二択の39件中38件が同じ選択肢に入る（ほぼ定数予測器）
    一方、`diagnose_extremes`ではK=5の予測scoreによる順位AUCが0.708あることを確認した。
    AUCは定数バイアスに不変、argmaxは不変でない。したがって「テキストに依存しない加法的
    オフセットを引けば、既存の重みのままargmaxがその信号に従う」はずである。

    内容のないstate（`NULL_STATES`）での出力をバイアス推定に使う。再訓練は不要。
    """
    print(f"\n=== S1: null stateによるバイアス除去（再訓練なし） ===")
    dataset = load_sst5("validation")
    rows = list(dataset.select(range(min(n, len(dataset)))))

    # (a) 最遠ペア二択
    pair_branch = Branch(
        kind="score", instructions=INSTRUCTION_VARIANTS[0], options=[LEVEL_ORDER[0], LEVEL_ORDER[-1]]
    )
    pair_bias = _null_log_probs(model, pair_branch)
    extremes = [row for row in rows if int(row["label"]) in (0, 4)]
    raw_correct = calibrated_correct = 0
    raw_picks = [0, 0]
    calibrated_picks = [0, 0]
    for row in extremes:
        gold = 0 if int(row["label"]) == 0 else 1
        log_probs = _log_probs(model, row["text"], pair_branch)
        raw_pick = int(log_probs[1] > log_probs[0])
        calibrated_probs = _calibrated(log_probs, pair_bias)
        calibrated_pick = int(calibrated_probs[1] > calibrated_probs[0])
        raw_correct += int(raw_pick == gold)
        calibrated_correct += int(calibrated_pick == gold)
        raw_picks[raw_pick] += 1
        calibrated_picks[calibrated_pick] += 1
    print(f"  (a) 最遠ペア二択 n={len(extremes)}: "
          f"accuracy {raw_correct / len(extremes):.3f} -> {calibrated_correct / len(extremes):.3f}")
    print(f"      選択の偏り: {raw_picks} -> {calibrated_picks} "
          f"(正解の内訳={[sum(1 for r in extremes if int(r['label']) == 0), sum(1 for r in extremes if int(r['label']) == 4)]})")

    # (b) sst5 K=5（全レベル）
    full_branch = Branch(kind="score", instructions=INSTRUCTION_VARIANTS[0], options=LEVEL_ORDER)
    full_bias = _null_log_probs(model, full_branch)
    golds: list[int] = []
    raw_positions: list[float] = []
    calibrated_positions: list[float] = []
    raw_hits = calibrated_hits = 0
    for row in rows:
        gold = int(row["label"])
        log_probs = _log_probs(model, row["text"], full_branch)
        calibrated_probs = _calibrated(log_probs, full_bias)
        golds.append(gold)
        raw_positions.append(_expectation([math.exp(lp) for lp in log_probs]))
        calibrated_positions.append(_expectation(calibrated_probs))
        raw_hits += int(max(range(len(log_probs)), key=lambda i: log_probs[i]) == gold)
        calibrated_hits += int(max(range(len(calibrated_probs)), key=lambda i: calibrated_probs[i]) == gold)
    raw_tau, raw_p = kendalltau(golds, raw_positions)
    calibrated_tau, calibrated_p = kendalltau(golds, calibrated_positions)
    counts = [golds.count(level) for level in range(len(LEVEL_ORDER))]
    print(f"  (b) sst5 K=5 n={len(rows)}: accuracy {raw_hits / len(rows):.3f} -> "
          f"{calibrated_hits / len(rows):.3f}（多数派ベースライン={max(counts) / len(rows):.3f}）")
    print(f"      tau {raw_tau:+.3f} (p={raw_p:.2e}) -> {calibrated_tau:+.3f} (p={calibrated_p:.2e})")
    print(f"      予測位置の幅 {max(raw_positions) - min(raw_positions):.3f} -> "
          f"{max(calibrated_positions) - min(calibrated_positions):.3f}（gold範囲=0..4）")

    # (c) 公式severity表（別ドメイン・K=3）
    severity_branch = Branch(kind="score", instructions=SEVERITY_QUESTION, options=LONG_K3)
    severity_bias = _null_log_probs(model, severity_branch)
    official = [score for _, score in SEVERITY_STATES]
    raw_severity: list[float] = []
    calibrated_severity: list[float] = []
    calibrated_argmax: list[int] = []
    for state, _ in SEVERITY_STATES:
        log_probs = _log_probs(model, state, severity_branch)
        calibrated_probs = _calibrated(log_probs, severity_bias)
        raw_severity.append(_expectation([math.exp(lp) for lp in log_probs]))
        calibrated_severity.append(_expectation(calibrated_probs))
        calibrated_argmax.append(max(range(len(calibrated_probs)), key=lambda i: calibrated_probs[i]))
    raw_stau, _ = kendalltau(official, raw_severity)
    calibrated_stau, _ = kendalltau(official, calibrated_severity)
    print(f"  (c) 公式severity表 K=3: tau {raw_stau:+.3f} -> {calibrated_stau:+.3f}")
    print(f"      位置 [" + ", ".join(f"{p:.2f}" for p in raw_severity) + "] -> ["
          + ", ".join(f"{p:.2f}" for p in calibrated_severity) + "]")
    print(f"      校正後argmax={calibrated_argmax}  (公式={[0, 1, 1, 1, 2]})")


DIAGNOSTICS = {
    "format": diagnose_option_format,
    "monotonicity": diagnose_monotonicity,
    "in_domain": diagnose_in_domain,
    "k": diagnose_k_generalization,
    "pairs": diagnose_pair_resolution,
    "bias": diagnose_option_bias,
    "calibration": diagnose_null_calibration,
    "extremes": diagnose_extremes,
}


def _main() -> None:
    """`python3 -m experiments.score_diagnose <checkpoint> [診断名 ...]`。診断名を省略すると全部走る。"""
    torch.manual_seed(0)
    model = JevModel()
    checkpoint_path = sys.argv[1]
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.pointer.load_state_dict(ckpt["state_dict"])
    print(f"readout: {checkpoint_path} (epochs={ckpt.get('epochs')})")

    selected = sys.argv[2:] or list(DIAGNOSTICS)
    for name in selected:
        if name not in DIAGNOSTICS:
            raise SystemExit(f"未知の診断名: {name}（利用可能: {', '.join(DIAGNOSTICS)}）")
        DIAGNOSTICS[name](model)

if __name__ == "__main__":
    _main()
