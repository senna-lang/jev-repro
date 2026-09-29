"""Gate 2-a: 英語合成サポートデータ（`data/synthetic_support.py` v1）をChoice/Score訓練に追加する。

GOAL.mdを英語前提に切り替えたあとの最初の学習。Gold V1の文面は生成にも学習にも使わない。
ticket_queueは入れない（効果を切り分ける）。構成はGate 1-cと同じ
（既存5データセット + bugzilla severity、mean-pool、3 epoch、選択肢順シャッフル、seed=0）。

使い方:
    python3 -m experiments.synthetic_gate2a --check
    python3 -m experiments.synthetic_gate2a
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from data.synthetic_support import TASKS, load_rows, split_rows, to_packed_example
from legacy.experiments.pooling_ablation import _build_train_and_eval
from legacy.experiments.severity_gate1b import OPTION_POOLING, _ood_severity_report
from legacy.experiments.severity_gate1c import BUG_HELD_OUT_PER_LEVEL, BUG_TRAIN_PER_LEVEL, _build_bug_examples
from legacy.experiments.train import evaluate, train_readout
from legacy.experiments.train_diverse import _reversed_order_accuracy, _score_mean_abs_error
from legacy.model_readout.forward import JevModel
from legacy.model_readout.readout import PointerReadout

EPOCHS = 3
SEED = 0
VERSION = 1


def _build_synth(seed: int) -> tuple[list, list, list[str], list[str]]:
    rows = load_rows(VERSION)
    train_rows, held_rows = split_rows(rows)
    rng_train = random.Random(seed)
    rng_held = random.Random(seed + 1)
    train = [to_packed_example(row, rng_train) for row in train_rows]
    held = [to_packed_example(row, rng_held) for row in held_rows]
    return train, held, [row["text"] for row in train_rows], [row["text"] for row in held_rows]


def _check(train, held, train_texts, held_texts) -> None:
    overlap = set(train_texts) & set(held_texts)
    assert not overlap, f"train/held-out text overlap: {len(overlap)}"
    for example in train + held:
        branch = example.request.branches[0]
        assert 0 <= example.label < len(branch.options)
        assert branch.kind in ("choice", "score")
    print(f"OK: synth train={len(train)} held-out={len(held)} 重複0件 ラベル範囲内")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    train_synth, held_synth, train_texts, held_texts = _build_synth(SEED)
    _check(train_synth, held_synth, train_texts, held_texts)
    if args.check:
        return

    print(f"Gate 2-a: synthetic v{VERSION}, option_pooling={OPTION_POOLING!r}, epochs={EPOCHS}")
    train_examples, eval_sets = _build_train_and_eval(seed=SEED)
    bug_train = _build_bug_examples(BUG_TRAIN_PER_LEVEL, "first", seed=SEED)
    bug_held_out = _build_bug_examples(BUG_HELD_OUT_PER_LEVEL, "second", seed=SEED)
    train_examples = train_examples + bug_train + train_synth
    print(f"train examples={len(train_examples)}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    model = JevModel(option_pooling=OPTION_POOLING)
    model.pointer = PointerReadout(model.backbone.config.hidden_size)

    start = time.perf_counter()
    train_readout(model, train_examples, epochs=EPOCHS, shuffle_options=True, seed=SEED)
    print(f"train time: {time.perf_counter() - start:.1f}s")

    print("\n=== in-domain（Gate 1-c: AG 0.8650 / banking 0.7750 / emotion 0.5475 / dbpedia 0.9050 / sst5 0.4550） ===")
    for name, examples in eval_sets.items():
        result = evaluate(model, examples)
        print(f"  {name:10s}: accuracy={result['accuracy']:.4f}, ECE={result['ece']:.4f}")
    print(f"  sst5 MAE={_score_mean_abs_error(model, eval_sets['sst5']):.4f}（Gate 1-c 0.6435）")
    rev = _reversed_order_accuracy(model, eval_sets["AG News"][:80])
    print(f"  順序反転: {rev}（Gate 1-c 0.9500）")

    ood = _ood_severity_report(model, bug_held_out)
    print(f"\n=== bugzilla（Gate 1-c tau=+0.423） ===\n"
          f"  n={ood['n']} accuracy={ood['accuracy']:.4f} tau={ood['tau']:+.3f} (p={ood['p_value']:.3f}) "
          f"定数予測器={'YES' if ood['is_constant'] else 'no'} spread={ood['spread']:.3f}")

    by_task: dict[str, list] = defaultdict(list)
    held_rows = split_rows(load_rows(VERSION))[1]
    rng = random.Random(SEED + 1)
    for row, example in zip(held_rows, held_synth):
        by_task[row["task"]].append(example)
    print("\n=== synthetic held-out ===")
    for task in TASKS:
        result = evaluate(model, by_task[task])
        print(f"  {task:20s}: n={result['n']} accuracy={result['accuracy']:.4f}")

    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / f"pointer_gate2a_{OPTION_POOLING}_e{EPOCHS}.pt"
    torch.save(
        {
            "readout": f"PointerReadout (option_pooling={OPTION_POOLING}, Gate 2-a synthetic v{VERSION})",
            "hidden_size": model.backbone.config.hidden_size,
            "option_pooling": OPTION_POOLING,
            "datasets": ["ag_news", "banking77", "emotion", "sst5", "dbpedia14", "bugzilla_severity", f"synthetic_support_v{VERSION}"],
            "epochs": EPOCHS,
            "shuffle_options": True,
            "state_dict": model.pointer.state_dict(),
        },
        ckpt_path,
    )
    print(f"\ncheckpoint: {ckpt_path}")


if __name__ == "__main__":
    _main()
