"""Gate 1-a: `h_opts`のpooling方式（末尾トークン1つ vs 選択肢区間mean-pool）を比較する。

設計: `internal-docs/EXP_HEAD_VS_BACKBONE.md` §2 Gate 1-a。Gate 0でseverity_tableドメインの
順序信号が凍結backboneに既に存在することを確認した（`results/2026-09-23_gate0_zeroshot_severity.md`,
tau+0.400・定数予測器でない）ため、backbone介入ではなくreadout側の安価な改善から着手する。

候補選定は`EXP_HEAD_VS_BACKBONE.md` §1.5に従う: **訓練ドメイン（AG News/banking77/emotion/
sst5/dbpedia14）のeval splitをプールしたaccuracyのみ**で比較する。公式Jev例・OODデータ
（Gate 1-bで追加するseverityデータ）は一切見ない（oracle選択の禁止）。

両条件とも`experiments/train_diverse.py`と同一のデータ構築・訓練recipeを使う
（5データセット混合・3エポック・選択肢順シャッフル・同じseed）。異なるのは
`JevModel(option_pooling=...)`だけ。

使い方:
    python3 -m experiments.pooling_ablation
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from data.ag_news import eval_split as ag_eval_split
from data.ag_news import train_subset as ag_train_subset
from data.banking77 import load_banking77
from data.dbpedia14 import load_dbpedia14
from data.emotion import load_emotion
from data.sst5 import load_sst5
from legacy.experiments.train import evaluate, train_readout
from legacy.experiments.train_diverse import (
    AG_EVAL_SIZE,
    AG_TRAIN_SIZE,
    BANKING_EVAL_SIZE,
    BANKING_TRAIN_SIZE,
    DBPEDIA_EVAL_SIZE,
    DBPEDIA_TRAIN_SIZE,
    EMOTION_EVAL_SIZE,
    EMOTION_TRAIN_SIZE,
    SEED,
    SST5_EVAL_SIZE,
    SST5_TRAIN_SIZE,
    _build_ag_examples,
    _build_banking_examples,
    _build_dbpedia_examples,
    _build_emotion_examples,
    _build_sst5_examples,
    _reversed_order_accuracy,
    _score_mean_abs_error,
)
from legacy.model_readout.forward import JevModel
from legacy.model_readout.readout import PointerReadout

EPOCHS = 3


def _build_train_and_eval(seed: int = SEED):
    data_rng = random.Random(seed)
    ag_train = _build_ag_examples(ag_train_subset(AG_TRAIN_SIZE))
    banking_train = _build_banking_examples(
        load_banking77("train").shuffle(seed=seed).select(range(BANKING_TRAIN_SIZE)), data_rng
    )
    emotion_train = _build_emotion_examples(
        load_emotion("train").shuffle(seed=seed).select(range(EMOTION_TRAIN_SIZE)), data_rng
    )
    sst5_train = _build_sst5_examples(
        load_sst5("train").shuffle(seed=seed).select(range(SST5_TRAIN_SIZE)), data_rng
    )
    dbpedia_train = _build_dbpedia_examples(
        load_dbpedia14("train").shuffle(seed=seed).select(range(DBPEDIA_TRAIN_SIZE)), data_rng
    )
    train_examples = ag_train + banking_train + emotion_train + sst5_train + dbpedia_train

    ag_eval = _build_ag_examples(ag_eval_split(AG_EVAL_SIZE))
    banking_eval = _build_banking_examples(
        load_banking77("test").shuffle(seed=seed).select(range(BANKING_EVAL_SIZE)), random.Random(seed + 1)
    )
    emotion_eval = _build_emotion_examples(
        load_emotion("test").shuffle(seed=seed).select(range(EMOTION_EVAL_SIZE)), random.Random(seed + 2)
    )
    sst5_eval = _build_sst5_examples(
        load_sst5("test").shuffle(seed=seed).select(range(SST5_EVAL_SIZE)), random.Random(seed + 3)
    )
    dbpedia_eval = _build_dbpedia_examples(
        load_dbpedia14("test").shuffle(seed=seed).select(range(DBPEDIA_EVAL_SIZE)), random.Random(seed + 4)
    )
    eval_sets = {
        "AG News": ag_eval,
        "banking77": banking_eval,
        "emotion": emotion_eval,
        "dbpedia14": dbpedia_eval,
        "sst5": sst5_eval,
    }
    return train_examples, eval_sets


def _run_condition(pooling: str, train_examples, eval_sets) -> dict:
    print(f"\n{'=' * 60}\n条件: option_pooling={pooling!r}\n{'=' * 60}")
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = JevModel(option_pooling=pooling)
    model.pointer = PointerReadout(model.backbone.config.hidden_size)

    start = time.perf_counter()
    train_readout(model, train_examples, epochs=EPOCHS, shuffle_options=True, seed=SEED)
    elapsed = time.perf_counter() - start
    print(f"train time: {elapsed:.1f}s")

    accuracies: dict[str, float] = {}
    for name, examples in eval_sets.items():
        r = evaluate(model, examples)
        accuracies[name] = r["accuracy"]
        print(f"  {name:10s}: accuracy={r['accuracy']:.4f}, ECE={r['ece']:.4f}")

    sst5_mae = _score_mean_abs_error(model, eval_sets["sst5"])
    print(f"  {'sst5':10s}: MAE(level)={sst5_mae:.4f}")

    rev = _reversed_order_accuracy(model, eval_sets["AG News"][:80])
    print(f"  順序反転(AG News 80件): {rev}")

    pooled_accuracy = sum(accuracies.values()) / len(accuracies)
    print(f"  ★ 訓練ドメインvalプール accuracy（選定基準）: {pooled_accuracy:.4f}")

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"pointer_pooling_{pooling}_e{EPOCHS}.pt"
    torch.save(
        {
            "readout": f"PointerReadout (option_pooling={pooling}, Gate 1-a pooling ablation)",
            "hidden_size": model.backbone.config.hidden_size,
            "option_pooling": pooling,
            "datasets": ["ag_news", "banking77", "emotion", "sst5", "dbpedia14"],
            "epochs": EPOCHS,
            "shuffle_options": True,
            "state_dict": model.pointer.state_dict(),
        },
        ckpt_path,
    )
    print(f"checkpoint: {ckpt_path}")

    return {
        "pooling": pooling,
        "accuracies": accuracies,
        "pooled_accuracy": pooled_accuracy,
        "sst5_mae": sst5_mae,
        "reversed_order": rev,
        "train_time": elapsed,
        "checkpoint": str(ckpt_path),
    }


def _main() -> None:
    print(f"Gate 1-a: pooling ablation（last-token vs mean-pool）, epochs={EPOCHS}, seed={SEED}")
    print("選定基準: 訓練ドメイン（AG News/banking77/emotion/sst5/dbpedia14）val poolのaccuracyのみ。"
          "公式Jev例・OODデータはこの選定には使わない。")

    train_examples, eval_sets = _build_train_and_eval()

    results = {}
    for pooling in ("last", "mean"):
        results[pooling] = _run_condition(pooling, train_examples, eval_sets)

    print(f"\n{'=' * 60}\nGate 1-a 結果まとめ\n{'=' * 60}")
    for pooling, r in results.items():
        print(f"  {pooling:6s}: pooled_accuracy={r['pooled_accuracy']:.4f}  "
              f"sst5_MAE={r['sst5_mae']:.4f}  "
              f"順序反転一致率={r['reversed_order']['base_rev_agreement']:.4f}  "
              f"train_time={r['train_time']:.1f}s")

    winner = max(results.values(), key=lambda r: r["pooled_accuracy"])
    print(f"\n★ 選定結果: option_pooling={winner['pooling']!r}"
          f"（pooled_accuracy={winner['pooled_accuracy']:.4f}）")
    print("この条件をGate 1-bで使う。")


if __name__ == "__main__":
    _main()
