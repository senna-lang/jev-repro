"""弱いドメイン（emotion, sst5）でhead-only（readoutのみ）とgradual unfreezing（backbone末尾n層
+readout）を比較するablation。

`results/logs/2026-09-22_score_diagnosis.md`で確認した通り、凍結backboneはLM head zero-shotなら
severityを4/5・sst5最遠ペアを0.900で解ける。にもかかわらず訓練済みpointer readoutは同じ課題で
大きく劣る。これは2つの仮説を区別できていない:

  (A) データカバレッジ不足。訓練ドメインを増やせば直る。
  (B) readout自体が汎化しない設計。backbone表現の中で情動・順序の区別が線形には現れておらず、
      浅いreadout（線形write projection + 内積）では引き出せない。

このablationは(B)を検証する。同じデータ量・エポック数で:
  1. head-only（既存の訓練方式、backbone完全凍結）
  2. backbone末尾1層 + readoutを同時学習（`backbone_lr`は小さく）
  3. backbone末尾2層 + readoutを同時学習

tau/accuracy/ECEが unfreeze で有意に伸びれば「backbone表現に情動・順序の区別が線形には現れて
いない」ことの直接証拠になる。伸びなければ、病理はbackboneではなく訓練手続き（選択肢固有バイアス
がcross-entropyの最短経路として学習されてしまうこと）にあると確定でき、backboneを動かす必要は
ないと言い切れる。

各armは新しい`JevModel()`を使う（`train_readout`はbackbone重みを直接書き換えるため、使い回すと
前armの学習が後armに漏れる）。

使い方:
    python3 -m experiments.unfreeze_ablation
"""

from __future__ import annotations

import random
import time

import torch
from scipy.stats import kendalltau

from data.emotion import load_emotion, to_packed_example as emotion_to_packed_example
from data.sst5 import INSTRUCTION_VARIANTS, LEVEL_ORDER, load_sst5, to_packed_example as sst5_to_packed_example
from legacy.experiments.score_diagnose import LONG_K3, SEVERITY_QUESTION, SEVERITY_STATES
from legacy.experiments.train import evaluate, train_readout
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest

TRAIN_SIZE = 1000
EVAL_SIZE = 200
EPOCHS = 1
INIT_SEED = 0
BACKBONE_LR = 1e-5

# (表示名, unfreeze_last_n_layers)
ARMS: list[tuple[str, int]] = [
    ("head-only", 0),
    ("unfreeze末尾1層", 1),
    ("unfreeze末尾2層", 2),
]


def _build_emotion(n: int, split: str, seed: int) -> list:
    dataset = load_emotion(split).shuffle(seed=seed).select(range(n))
    label_names = dataset.features["label"].names
    rng = random.Random(seed)
    return [emotion_to_packed_example(row["text"], row["label"], label_names, rng) for row in dataset]


def _build_sst5(n: int, split: str, seed: int) -> list:
    dataset = load_sst5(split).shuffle(seed=seed).select(range(n))
    rng = random.Random(seed)
    return [sst5_to_packed_example(row["text"], row["label"], rng) for row in dataset]


@torch.no_grad()
def _sst5_tau_at_k(model: JevModel, rows, k: int) -> dict:
    """sst5 held-outを、選択肢を長さkの連続窓に固定して評価する（`score_ablation.py`と同じ手法）。"""
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
    return {"tau": tau, "p": p_value, "mae": mae}


@torch.no_grad()
def _severity(model: JevModel) -> dict:
    """公式severity表（K=3・サポートチケットドメイン・訓練データに一度も出ないジャンル）。"""
    branch = Branch(kind="score", instructions=SEVERITY_QUESTION, options=LONG_K3)
    official = [score for _, score in SEVERITY_STATES]
    positions = [
        model.forward(PackedRequest(state=state, branches=[branch]))[0]["score"] - 1.0
        for state, _ in SEVERITY_STATES
    ]
    tau, _ = kendalltau(official, positions)
    return {"tau": tau, "positions": positions}


def run_arm(name: str, unfreeze_n: int, emotion_train, emotion_eval, sst5_train_rows, sst5_eval_rows) -> dict:
    print(f"\n=== {name} (unfreeze_last_n_layers={unfreeze_n}) ===")
    torch.manual_seed(INIT_SEED)
    model = JevModel()

    sst5_train_examples = [sst5_to_packed_example(r["text"], r["label"], random.Random(INIT_SEED)) for r in sst5_train_rows]
    train_examples = list(emotion_train) + sst5_train_examples

    start = time.perf_counter()
    train_readout(
        model, train_examples, epochs=EPOCHS, shuffle_options=True, seed=INIT_SEED,
        unfreeze_last_n_layers=unfreeze_n, backbone_lr=BACKBONE_LR,
    )
    elapsed = time.perf_counter() - start

    emotion_result = evaluate(model, emotion_eval)
    print(f"  emotion held-out: accuracy={emotion_result['accuracy']:.4f}  ECE={emotion_result['ece']:.4f}")

    sst5_metrics = {k: _sst5_tau_at_k(model, sst5_eval_rows, k) for k in (2, 3, 5)}
    for k, m in sst5_metrics.items():
        print(f"  sst5 K={k}: tau={m['tau']:+.3f} (p={m['p']:.3f})  mae={m['mae']:.3f}")

    severity = _severity(model)
    print(f"  公式severity(サポートチケット, out-of-domain): tau={severity['tau']:+.3f}  "
          f"positions={[f'{p:.2f}' for p in severity['positions']]}")
    print(f"  学習時間: {elapsed:.1f}s")

    return {
        "name": name,
        "emotion": emotion_result,
        "sst5": sst5_metrics,
        "severity": severity,
    }


def _main() -> None:
    print(f"emotion+sst5混合 train={TRAIN_SIZE}(各ドメイン) eval={EVAL_SIZE} epochs={EPOCHS} "
          f"backbone_lr={BACKBONE_LR}: head-only vs gradual unfreezing")

    emotion_train_dataset = load_emotion("train").shuffle(seed=INIT_SEED).select(range(TRAIN_SIZE))
    emotion_label_names = emotion_train_dataset.features["label"].names
    emotion_rng = random.Random(INIT_SEED)
    emotion_train = [
        emotion_to_packed_example(row["text"], row["label"], emotion_label_names, emotion_rng)
        for row in emotion_train_dataset
    ]
    emotion_eval_dataset = load_emotion("test").shuffle(seed=INIT_SEED).select(range(EVAL_SIZE))
    emotion_eval_rng = random.Random(INIT_SEED + 1)
    emotion_eval = [
        emotion_to_packed_example(row["text"], row["label"], emotion_label_names, emotion_eval_rng)
        for row in emotion_eval_dataset
    ]

    sst5_train_rows = load_sst5("train").shuffle(seed=INIT_SEED).select(range(TRAIN_SIZE))
    sst5_eval_rows = load_sst5("validation").shuffle(seed=INIT_SEED).select(range(EVAL_SIZE))

    results = [
        run_arm(name, n, emotion_train, emotion_eval, sst5_train_rows, sst5_eval_rows)
        for name, n in ARMS
    ]

    print("\n=== まとめ ===")
    print(f"{'arm':<20} emotion_acc  sst5_K3_tau  severity_tau")
    for r in results:
        print(f"{r['name']:<20} {r['emotion']['accuracy']:.4f}       "
              f"{r['sst5'][3]['tau']:+.3f}       {r['severity']['tau']:+.3f}")
    print("\nunfreezeアームでtauが有意に伸びれば(B)backbone表現の線形分離不足が支持される。"
          "伸びなければ(A)訓練手続き（選択肢固有バイアス）が病理と確定できる。")


if __name__ == "__main__":
    _main()
