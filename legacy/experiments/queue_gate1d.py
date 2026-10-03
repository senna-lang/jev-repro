"""Gate 1-d: サポートチケットの部署振り分け（`data/ticket_queue.py`）をChoice訓練データに追加する。

Gold V1のChoice（部署振り分け等）は学習済みpointerで0.475に留まる。採用根拠
（`results/logs/2026-09-24_signal_diagnosis.md`）:
  - LM head zero-shotではqueueは0.320（chance 0.297）で有意でなかったが、
  - 凍結backbone隠れ状態のlinear probeではheld-out 0.480〜0.495（chance 0.200, p<1e-4）。
    本番pointerも学習readoutなので、採否はlinear probeを基準にする。

Gate 1-cと同じ構成（既存5データセット + bugzilla severity、mean-pool、3 epoch、選択肢順シャッフル、
seed=0）にticket_queue（300件/部署×5）を足す。Gold V1は一切使わない（評価は`evaluation.run`で別途）。

使い方:
    python3 -m experiments.queue_gate1d
    python3 -m experiments.queue_gate1d --check   # example構築の検査だけ（学習しない）
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from data.ticket_queue import QUEUES, load_ticket_queue_stratified
from data.ticket_queue import to_packed_example as queue_to_packed_example
from legacy.experiments.pooling_ablation import _build_train_and_eval
from legacy.experiments.severity_gate1b import OPTION_POOLING, _ood_severity_report
from legacy.experiments.severity_gate1c import BUG_HELD_OUT_PER_LEVEL, BUG_TRAIN_PER_LEVEL, _build_bug_examples
from legacy.experiments.train import evaluate, train_readout
from legacy.experiments.train_diverse import _reversed_order_accuracy, _score_mean_abs_error
from legacy.model_readout.forward import JevModel
from legacy.model_readout.readout import PointerReadout

EPOCHS = 3
SEED = 0
QUEUE_TRAIN_PER = 300  # 5 queues × 300 = 1500件
QUEUE_HELD_OUT_PER = 40  # 5 queues × 40 = 200件（offset=QUEUE_TRAIN_PERで重複なし）


def _build_queue_examples(n_per_queue: int, offset: int, seed: int) -> tuple[list, list[tuple[str, str]]]:
    rng = random.Random(seed + offset)
    rows = load_ticket_queue_stratified(n_per_queue, seed=seed, offset=offset)
    examples = [queue_to_packed_example(r["subject"], r["body"], r["queue"], rng) for r in rows]
    keys = [(r["subject"] or "", r["body"] or "") for r in rows]
    return examples, keys


def _check(queue_train, queue_train_keys, queue_held, queue_held_keys) -> None:
    assert len(queue_train) == QUEUE_TRAIN_PER * len(QUEUES), len(queue_train)
    assert len(queue_held) == QUEUE_HELD_OUT_PER * len(QUEUES), len(queue_held)
    overlap = set(queue_train_keys) & set(queue_held_keys)
    assert not overlap, f"train/held-out重複: {len(overlap)}件"
    for ex in queue_train + queue_held:
        branch = ex.request.branches[0]
        assert branch.kind == "choice" and 0 <= ex.label < len(branch.options), (ex.label, branch.options)
    chance = sum(1 / len(ex.request.branches[0].options) for ex in queue_held) / len(queue_held)
    print(f"OK: queue train={len(queue_train)} held-out={len(queue_held)} 重複0件 ラベル範囲内 held-out chance={chance:.3f}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="example構築の検査だけ行い学習しない")
    args = parser.parse_args()

    queue_train, queue_train_keys = _build_queue_examples(QUEUE_TRAIN_PER, 0, SEED)
    queue_held, queue_held_keys = _build_queue_examples(QUEUE_HELD_OUT_PER, QUEUE_TRAIN_PER, SEED)
    _check(queue_train, queue_train_keys, queue_held, queue_held_keys)
    if args.check:
        return

    print(f"Gate 1-d: ticket_queue追加, option_pooling={OPTION_POOLING!r}, epochs={EPOCHS}, "
          f"queue train={QUEUE_TRAIN_PER}/queue x {len(QUEUES)}, held_out={QUEUE_HELD_OUT_PER}/queue")
    train_examples, eval_sets = _build_train_and_eval(seed=SEED)
    bug_train = _build_bug_examples(BUG_TRAIN_PER_LEVEL, "first", seed=SEED)
    bug_held_out = _build_bug_examples(BUG_HELD_OUT_PER_LEVEL, "second", seed=SEED)
    train_examples = train_examples + bug_train + queue_train
    print(f"train examples={len(train_examples)}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    model = JevModel(option_pooling=OPTION_POOLING)
    model.pointer = PointerReadout(model.backbone.config.hidden_size)

    before = evaluate(model, queue_held)
    print(f"\n訓練前 queue held-out: accuracy={before['accuracy']:.4f}, ECE={before['ece']:.4f}")

    start = time.perf_counter()
    train_readout(model, train_examples, epochs=EPOCHS, shuffle_options=True, seed=SEED)
    print(f"train time: {time.perf_counter() - start:.1f}s")

    print("\n=== 訓練後: in-domain（Gate 1-c: AG 0.8650 / banking 0.7750 / emotion 0.5475 / dbpedia 0.9050 / sst5 0.4550） ===")
    for name, examples in eval_sets.items():
        r = evaluate(model, examples)
        print(f"  {name:10s}: accuracy={r['accuracy']:.4f}, ECE={r['ece']:.4f}")
    print(f"  {'sst5':10s}: MAE(level)={_score_mean_abs_error(model, eval_sets['sst5']):.4f}（Gate 1-c 0.6435）")
    rev = _reversed_order_accuracy(model, eval_sets["AG News"][:80])
    print(f"  順序反転(AG News 80件): {rev}（Gate 1-c 一致率0.9500）")

    ood = _ood_severity_report(model, bug_held_out)
    print(f"\n=== bugzilla severity held-out（Gate 1-c tau=+0.423） ===\n"
          f"  n={ood['n']} accuracy={ood['accuracy']:.4f} tau={ood['tau']:+.3f} (p={ood['p_value']:.3f}) "
          f"定数予測器={'YES' if ood['is_constant'] else 'no'} spread={ood['spread']:.3f}")

    after = evaluate(model, queue_held)
    print(f"\n=== ticket_queue held-out ===\n  accuracy={after['accuracy']:.4f}, ECE={after['ece']:.4f}（訓練前 {before['accuracy']:.4f}）")

    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / f"pointer_gate1d_{OPTION_POOLING}_e{EPOCHS}.pt"
    torch.save(
        {
            "readout": f"PointerReadout (option_pooling={OPTION_POOLING}, Gate 1-d ticket_queue data)",
            "hidden_size": model.backbone.config.hidden_size,
            "option_pooling": OPTION_POOLING,
            "datasets": ["ag_news", "banking77", "emotion", "sst5", "dbpedia14", "bugzilla_severity", "ticket_queue"],
            "epochs": EPOCHS,
            "shuffle_options": True,
            "state_dict": model.pointer.state_dict(),
        },
        ckpt_path,
    )
    print(f"\ncheckpoint: {ckpt_path}")


if __name__ == "__main__":
    _main()
