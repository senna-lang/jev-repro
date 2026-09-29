"""複数ドメイン・複数タスク型を混ぜ、選択肢順序をエポックごとにシャッフルしてPointerReadoutを訓練する。

単一ドメイン・固定4択・固定順のAG Newsだけで訓練すると、readoutが「4文字列パターンの
位置」に過学習する（実測: W_q,W_k更新分の実効ランク1.83、選択肢順を反転しただけでaccuracy
0.808→0.183、base/reversed一致率9%）。AG News+banking77の2データセット混合＋順序シャッフル
で一致率10%→71-80%まで改善したが（`results/2026-09-21_diverse_training_order_invariance.md`）、
公式Jev実例との再比較でtone（態度判定）・severity/frustration（順序尺度）は依然不一致が
残った——訓練データにこのタスク型が一切存在しなかったため。

この実験はさらに3データセットを追加する:
- `emotion`（態度・感情判定Choice, tone問題に対応するタスク型）
- `sst5`（5段階順序尺度Score, severity/frustration問題に対応するタスク型。
  `experiments/train.py`のordinal MSE補助損失と組み合わせる）
- `dbpedia14`（Choice選択肢語彙のさらなる多様化, K=4-8）

各データセットはinstructions文面も複数用意し、エポックごとにランダムに選ぶ
（特定の質問文への癒着を防ぐ）。readoutは`model/readout.py`のLayerNorm付き版
（calibration/飽和バグ修正後）を使う。
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from data.ag_news import eval_split as ag_eval_split
from data.ag_news import to_packed_example as ag_to_packed_example
from data.ag_news import train_subset as ag_train_subset
from data.banking77 import load_banking77
from data.banking77 import to_packed_example as banking_to_packed_example
from data.common import ChoiceExample, reshuffled
from data.dbpedia14 import load_dbpedia14
from data.dbpedia14 import to_packed_example as dbpedia_to_packed_example
from data.emotion import load_emotion
from data.emotion import to_packed_example as emotion_to_packed_example
from data.sst5 import load_sst5
from data.sst5 import to_packed_example as sst5_to_packed_example
from legacy.experiments.train import evaluate, train_readout
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest
from legacy.model_readout.readout import PointerReadout

AG_TRAIN_SIZE = 1500
AG_EVAL_SIZE = 400
BANKING_TRAIN_SIZE = 1000
BANKING_EVAL_SIZE = 400
EMOTION_TRAIN_SIZE = 1000
EMOTION_EVAL_SIZE = 400
SST5_TRAIN_SIZE = 1000
SST5_EVAL_SIZE = 400
DBPEDIA_TRAIN_SIZE = 1000
DBPEDIA_EVAL_SIZE = 400
SEED = 0


def _build_ag_examples(dataset) -> list[ChoiceExample]:
    label_names = dataset.features["label"].names
    return [ag_to_packed_example(ex["text"], ex["label"], label_names) for ex in dataset]


def _build_banking_examples(dataset, rng: random.Random) -> list[ChoiceExample]:
    label_names = dataset.features["label"].names
    return [banking_to_packed_example(ex["text"], ex["label"], label_names, rng) for ex in dataset]


def _build_emotion_examples(dataset, rng: random.Random) -> list[ChoiceExample]:
    label_names = dataset.features["label"].names
    return [emotion_to_packed_example(ex["text"], ex["label"], label_names, rng) for ex in dataset]


def _build_sst5_examples(dataset, rng: random.Random) -> list[ChoiceExample]:
    return [sst5_to_packed_example(ex["text"], ex["label"], rng) for ex in dataset]


def _build_dbpedia_examples(dataset, rng: random.Random) -> list[ChoiceExample]:
    label_names = dataset.features["label"].names
    return [
        dbpedia_to_packed_example(ex["title"], ex["content"], ex["label"], label_names, rng)
        for ex in dataset
    ]


def _reversed_order_accuracy(model: JevModel, choice_eval: list[ChoiceExample]) -> dict:
    """Choice held-outで、選択肢順を反転させても同じ正解に到達できるかを測る（Score branchは対象外）。"""
    correct_base = correct_rev = agree = 0
    for ex in choice_eval:
        branch = ex.request.branches[0]
        k = len(branch.options)
        rev_options = list(reversed(branch.options))
        rev_label = k - 1 - ex.label
        rev_request = PackedRequest(
            state=ex.request.state,
            branches=[Branch(kind="choice", instructions=branch.instructions, options=rev_options)],
        )

        base_probs = model.forward(ex.request)[0]["probabilities"]
        rev_probs = model.forward(rev_request)[0]["probabilities"]
        base_pred = max(range(k), key=lambda i: base_probs[i])
        rev_pred_pos = max(range(k), key=lambda i: rev_probs[i])
        rev_pred = k - 1 - rev_pred_pos

        correct_base += base_pred == ex.label
        correct_rev += rev_pred == ex.label
        agree += base_pred == rev_pred

    n = len(choice_eval)
    return {"acc_base": correct_base / n, "acc_reversed": correct_rev / n, "base_rev_agreement": agree / n, "n": n}


def _score_mean_abs_error(model: JevModel, score_eval: list[ChoiceExample]) -> float:
    """Score held-outで、予測score（probability-weighted, 1始まり）と正解レベル（1始まり）の平均絶対誤差。"""
    errors = []
    for ex in score_eval:
        result = model.forward(ex.request)[0]
        errors.append(abs(result["score"] - (ex.label + 1)))
    return sum(errors) / len(errors)


def _main(epochs: int = 1) -> None:
    print(f"AG News={AG_TRAIN_SIZE}, banking77={BANKING_TRAIN_SIZE}(K3-6), emotion={EMOTION_TRAIN_SIZE}(K3-6), "
          f"sst5={SST5_TRAIN_SIZE}(score), dbpedia14={DBPEDIA_TRAIN_SIZE}(K4-8), "
          f"epochs={epochs}, shuffle_options=True, seed={SEED}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    model = JevModel()
    model.pointer = PointerReadout(model.backbone.config.hidden_size)

    data_rng = random.Random(SEED)
    ag_train = _build_ag_examples(ag_train_subset(AG_TRAIN_SIZE))
    banking_train = _build_banking_examples(
        load_banking77("train").shuffle(seed=SEED).select(range(BANKING_TRAIN_SIZE)), data_rng
    )
    emotion_train = _build_emotion_examples(
        load_emotion("train").shuffle(seed=SEED).select(range(EMOTION_TRAIN_SIZE)), data_rng
    )
    sst5_train = _build_sst5_examples(
        load_sst5("train").shuffle(seed=SEED).select(range(SST5_TRAIN_SIZE)), data_rng
    )
    dbpedia_train = _build_dbpedia_examples(
        load_dbpedia14("train").shuffle(seed=SEED).select(range(DBPEDIA_TRAIN_SIZE)), data_rng
    )
    train_examples = ag_train + banking_train + emotion_train + sst5_train + dbpedia_train

    ag_eval = _build_ag_examples(ag_eval_split(AG_EVAL_SIZE))
    banking_eval = _build_banking_examples(
        load_banking77("test").shuffle(seed=SEED).select(range(BANKING_EVAL_SIZE)), random.Random(SEED + 1)
    )
    emotion_eval = _build_emotion_examples(
        load_emotion("test").shuffle(seed=SEED).select(range(EMOTION_EVAL_SIZE)), random.Random(SEED + 2)
    )
    sst5_eval = _build_sst5_examples(
        load_sst5("test").shuffle(seed=SEED).select(range(SST5_EVAL_SIZE)), random.Random(SEED + 3)
    )
    dbpedia_eval = _build_dbpedia_examples(
        load_dbpedia14("test").shuffle(seed=SEED).select(range(DBPEDIA_EVAL_SIZE)), random.Random(SEED + 4)
    )

    eval_sets = {
        "AG News": ag_eval,
        "banking77": banking_eval,
        "emotion": emotion_eval,
        "dbpedia14": dbpedia_eval,
    }

    def _report(label: str) -> dict:
        results = {}
        for name, examples in eval_sets.items():
            r = evaluate(model, examples)
            results[name] = r
            print(f"  {name:10s}: accuracy={r['accuracy']:.4f}, ECE={r['ece']:.4f}")
        sst5_r = evaluate(model, sst5_eval)
        sst5_mae = _score_mean_abs_error(model, sst5_eval)
        results["sst5"] = {**sst5_r, "mae": sst5_mae}
        print(f"  {'sst5':10s}: accuracy={sst5_r['accuracy']:.4f}, ECE={sst5_r['ece']:.4f}, MAE(level)={sst5_mae:.4f}")
        rev = _reversed_order_accuracy(model, ag_eval[:80])
        results["reversed_order"] = rev
        print(f"  順序反転(AG News 80件): {rev}")
        print(f"=== {label} ===")
        return results

    print("\n=== 訓練前 ===")
    before = _report("訓練前")

    print("\n=== 訓練 (5データセット混合, エポックごとに選択肢順シャッフル) ===")
    start = time.perf_counter()
    train_readout(model, train_examples, epochs=epochs, shuffle_options=True, seed=SEED)
    elapsed = time.perf_counter() - start
    print(f"train time: {elapsed:.1f}s")

    print("\n=== 訓練後 ===")
    after = _report("訓練後")

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"pointer_diverse5_e{epochs}.pt"
    torch.save(
        {
            "readout": "PointerReadout (LayerNorm + 5-dataset diverse training + ordinal score loss)",
            "hidden_size": model.backbone.config.hidden_size,
            "datasets": ["ag_news", "banking77", "emotion", "sst5", "dbpedia14"],
            "epochs": epochs,
            "shuffle_options": True,
            "state_dict": model.pointer.state_dict(),
        },
        ckpt_path,
    )
    print(f"\ncheckpoint: {ckpt_path}")

    print("\n=== まとめ ===")
    for name in ("AG News", "banking77", "emotion", "dbpedia14"):
        print(f"{name:10s} accuracy: {before[name]['accuracy']:.4f} -> {after[name]['accuracy']:.4f}  "
              f"ECE: {before[name]['ece']:.4f} -> {after[name]['ece']:.4f}")
    print(f"{'sst5':10s} accuracy: {before['sst5']['accuracy']:.4f} -> {after['sst5']['accuracy']:.4f}  "
          f"MAE(level): {before['sst5']['mae']:.4f} -> {after['sst5']['mae']:.4f}")
    print(f"順序反転 base/rev一致率: {before['reversed_order']['base_rev_agreement']:.4f} "
          f"-> {after['reversed_order']['base_rev_agreement']:.4f}")


if __name__ == "__main__":
    _main()
