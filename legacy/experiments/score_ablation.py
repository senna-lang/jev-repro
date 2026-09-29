"""Score訓練の条件を1つだけ変えて比較する、sst5単独の安価なablationドライバ。

アームは`(k_range, score_mse_weight)`で定義する。他の条件（データ、seed、readoutの初期化、
epoch数）は完全に共通なので、差は変えた1つの条件に帰属できる。

これまでの結果:

  1. `score_mse_weight` 1.0 vs 0.0（K=5固定）: MSE補助損失は犯人ではなく、外すと全指標が
     悪化した（sst5 tau +0.187 -> +0.137、MAE 1.099 -> 1.322、ECE 0.173 -> 0.399、
     公式severityのtau +0.400 -> -0.600）。仮説「MSEが期待値だけ合わせる平均回帰を招き
     分布をなまらせる」は棄却。

  2. K固定 vs K可変（本ファイルの現在のアーム）: `experiments/score_diagnose.py`で、
     5レベル固定で訓練したreadoutは同じドメイン・同じテキストでもK=2/3/4では
     Kendall tauがほぼ0（p>0.14）まで落ち、訓練時のK=5でのみ信号が残ることを確認した。
     公式JevのScoreはK=3/K=4で使われるため、これは致命的。訓練時にKを振れば
     K不変な比較を学べるはず、というのがここで検証する仮説。

決定的な指標はheld-outでの**K別のKendall tau**。K可変で訓練したアームは、K=2/3/4でも
tauが有意に正になるはず。正解率やMAEはKによって基準が変わるので主指標にしない。

使い方:
    python3 -m experiments.score_ablation
"""

from __future__ import annotations

import random
import time

import torch
from scipy.stats import kendalltau

from data.sst5 import INSTRUCTION_VARIANTS, LEVEL_ORDER, load_sst5, to_packed_example
from legacy.experiments.score_diagnose import LONG_K3, SEVERITY_QUESTION, SEVERITY_STATES
from legacy.experiments.train import train_readout
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest
from legacy.model_readout.readout import PointerReadout

TRAIN_SIZE = 1000
EVAL_SIZE = 200
EPOCHS = 1
INIT_SEED = 0
EVAL_KS = (2, 3, 4, 5)

# (表示名, k_range, score_mse_weight)
ARMS: list[tuple[str, tuple[int, int], float]] = [
    ("K=5固定 (現状)", (5, 5), 1.0),
    ("K可変 [2,5]", (2, 5), 1.0),
]


def _build_train(n: int, k_range: tuple[int, int], seed: int) -> list:
    dataset = load_sst5("train")
    rng = random.Random(seed)
    rows = dataset.select(range(min(n, len(dataset))))
    return [to_packed_example(row["text"], row["label"], rng, k_range=k_range) for row in rows]


@torch.no_grad()
def _tau_at_k(model: JevModel, rows: list, k: int) -> dict:
    """held-outのテキストはそのままに、選択肢を長さkの連続窓に固定して順序性能を測る。

    評価側のinstructionは1種に固定する（訓練時は3種を振るが、評価で振ると
    文言のばらつきが条件間の差に混ざるため）。
    """
    rng = random.Random(INIT_SEED)
    golds: list[int] = []
    preds: list[float] = []
    for row in rows:
        label = int(row["label"])
        start = rng.randint(max(0, label - k + 1), min(label, len(LEVEL_ORDER) - k))
        options = LEVEL_ORDER[start:start + k]
        branch = Branch(kind="score", instructions=INSTRUCTION_VARIANTS[0], options=options)
        result = model.forward(PackedRequest(state=row["text"], branches=[branch]))[0]
        golds.append(label - start)
        preds.append(result["score"] - 1.0)

    tau, p_value = kendalltau(golds, preds)
    mae = sum(abs(g - p) for g, p in zip(golds, preds)) / len(golds)
    return {"tau": tau, "p": p_value, "mae": mae, "width": max(preds) - min(preds)}


@torch.no_grad()
def _severity(model: JevModel) -> dict:
    """公式severity表（K=3・別ドメイン・説明節の選択肢）での順序と広がり。"""
    branch = Branch(kind="score", instructions=SEVERITY_QUESTION, options=LONG_K3)
    official = [score for _, score in SEVERITY_STATES]
    positions = [
        model.forward(PackedRequest(state=state, branches=[branch]))[0]["score"] - 1.0
        for state, _ in SEVERITY_STATES
    ]
    tau, _ = kendalltau(official, positions)
    return {"tau": tau, "width": max(positions) - min(positions), "positions": positions}


def run_arm(model: JevModel, name: str, k_range: tuple[int, int], weight: float, rows: list) -> dict:
    print(f"\n=== {name} (k_range={k_range}, score_mse_weight={weight}) ===")
    torch.manual_seed(INIT_SEED)
    model.pointer = PointerReadout(model.backbone.config.hidden_size)

    train_examples = _build_train(TRAIN_SIZE, k_range, seed=INIT_SEED)
    start = time.perf_counter()
    train_readout(
        model,
        train_examples,
        epochs=EPOCHS,
        log_every=500,
        shuffle_options=False,  # Score branchはそもそもシャッフルされない
        seed=INIT_SEED,
        score_mse_weight=weight,
    )
    print(f"  train {time.perf_counter() - start:.0f}s")

    per_k = {k: _tau_at_k(model, rows, k) for k in EVAL_KS}
    for k, stats in per_k.items():
        print(f"  K={k}: tau={stats['tau']:+.3f} (p={stats['p']:.2e})  "
              f"MAE={stats['mae']:.3f}  予測幅={stats['width']:.3f}（使える幅={k - 1}）")

    severity = _severity(model)
    print(f"  公式severity(K=3, 別ドメイン): tau={severity['tau']:+.3f} 幅={severity['width']:.3f} "
          f"位置=[" + ", ".join(f"{p:.2f}" for p in severity["positions"]) + "]")
    return {"name": name, "per_k": per_k, "severity": severity}


def _main() -> None:
    print(f"sst5単独 train={TRAIN_SIZE} eval={EVAL_SIZE} epochs={EPOCHS}: "
          f"レベル数Kの振り方だけを変えたablation")
    torch.manual_seed(INIT_SEED)
    model = JevModel()
    rows = list(load_sst5("validation").select(range(EVAL_SIZE)))

    results = [run_arm(model, name, k_range, weight, rows) for name, k_range, weight in ARMS]

    print("\n=== まとめ: held-outのK別Kendall tau ===")
    header = f"{'arm':<16} | " + " | ".join(f"{'K=' + str(k):>14}" for k in EVAL_KS) + " | severity tau"
    print(header)
    for r in results:
        cells = " | ".join(
            f"{r['per_k'][k]['tau']:>+7.3f}(p{r['per_k'][k]['p']:.2f})" for k in EVAL_KS
        )
        print(f"{r['name']:<16} | {cells} | {r['severity']['tau']:>+12.3f}")
    print("\nK可変が効いているなら、K=2/3/4でもtauが有意に正になる（K=5固定アームはp>0.1で潰れる）。")


if __name__ == "__main__":
    _main()
