"""訓練データ量ごとのPointerReadoutの分類性能とECEを測定する。

各データ量で同じ初期readoutから訓練を開始し、固定した評価splitで正解率とECEを比較する。
バックボーンは固定し、各runは1 epochだけ訓練する。
"""

from __future__ import annotations

import time

import torch

from data.ag_news import TRAIN_SUBSET_SIZES, eval_split, train_subset
from model.forward import JevModel
from model.readout import PointerReadout
from experiments.train import _build_examples, evaluate, train_readout

EVAL_SIZE = 500
SEED = 0


def run_scaling_curve(sizes: list[int] = TRAIN_SUBSET_SIZES, eval_size: int = EVAL_SIZE) -> list[dict]:
    model = JevModel()
    hidden_size = model.backbone.config.hidden_size

    eval_examples = _build_examples(eval_split(eval_size))

    # サイズ=0（訓練なし、ランダム初期化のまま）の基準点。以降の全サイズも同じシードから始めるので、
    # この基準点は全サイズ共通で使い回せる。
    torch.manual_seed(SEED)
    model.pointer = PointerReadout(hidden_size)
    baseline = evaluate(model, eval_examples)
    print(f"size=0 (訓練前): accuracy={baseline['accuracy']:.4f}, ECE={baseline['ece']:.4f}")

    results = [{"size": 0, **baseline}]

    for size in sizes:
        print(f"\n=== size={size} ===")
        # 毎回同じシードでPointerReadoutを作り直す（データ量だけを変数にするため）
        torch.manual_seed(SEED)
        model.pointer = PointerReadout(hidden_size)

        train_examples = _build_examples(train_subset(size))

        start = time.perf_counter()
        train_readout(model, train_examples, epochs=1, log_every=max(1, size // 5))
        elapsed = time.perf_counter() - start

        result = evaluate(model, eval_examples)
        result["size"] = size
        result["train_seconds"] = elapsed
        results.append(result)
        print(f"size={size}: accuracy={result['accuracy']:.4f}, ECE={result['ece']:.4f}, "
              f"train_time={elapsed:.1f}s")

    return results


def _main() -> None:
    results = run_scaling_curve()

    print("\n=== スケーリングカーブ まとめ ===")
    print(f"{'size':>8} | {'accuracy':>8} | {'ECE':>8}")
    for r in results:
        print(f"{r['size']:>8} | {r['accuracy']:>8.4f} | {r['ece']:>8.4f}")

    accs = [r["accuracy"] for r in results]
    eces = [r["ece"] for r in results]
    if all(a2 >= a1 for a1, a2 in zip(accs, accs[1:])):
        print("\n正解率はデータ量とともに単調に改善している（予測通り）。")
    else:
        print("\n正解率はデータ量に対して単調には改善していない（サイズ間で上下している箇所がある）。")

    if all(e2 <= e1 for e1, e2 in zip(eces, eces[1:])):
        print("ECEもデータ量とともに単調に改善している。")
    else:
        print("ECEはデータ量に対して単調には改善していない——正解率ほど素直にスケールしない、"
              "というGuo et al. 2017の指摘と整合する可能性がある。")


if __name__ == "__main__":
    _main()
