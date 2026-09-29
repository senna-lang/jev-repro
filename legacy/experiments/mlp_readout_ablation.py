"""仮説1の検証: 非線形readout（`MLPPointerReadout`）でsupport_tickets OOD Kendall tauが改善するか。

Gate 1-b（`results/2026-09-23_gate1_pooling_and_severity_data.md`）で、線形pointer readout
（bilinear dot product、`PointerReadout`）はsupport_tickets held-outでtau+0.005のまま動かず、
停止条件を満たさなかった。2つの仮説のうち仮説1（信号はあるが線形readoutでは取り出せない）を
検証するため、`model/readout.py::MLPPointerReadout`（decision/option特徴を連結し2層MLPで
logitへ落とす非線形readout）に差し替えて、Gate 1-bと全く同じデータ・pooling・訓練recipeで
再訓練し、OOD tauを比較する。

Gate 1-bとの唯一の違いは`model.pointer`の型（`PointerReadout` -> `MLPPointerReadout`）。
選択肢pooling（Gate 1-a選定結果のmean-pool）・データ構成（既存5データセット+support_tickets）・
エポック数・seedはすべて同一に保つ。

使い方:
    python3 -m experiments.mlp_readout_ablation
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from legacy.experiments.pooling_ablation import _build_train_and_eval
from legacy.experiments.severity_gate1b import (
    OPTION_POOLING,
    TICKET_HELD_OUT_SIZE,
    TICKET_TRAIN_SIZE,
    _build_ticket_examples,
    _official_severity_qualitative,
    _ood_severity_report,
)
from legacy.experiments.train import evaluate, train_readout
from legacy.experiments.train_diverse import _reversed_order_accuracy, _score_mean_abs_error
from legacy.model_readout.forward import JevModel
from legacy.model_readout.readout import MLPPointerReadout

EPOCHS = 3
SEED = 0


def _main() -> None:
    print(f"仮説1検証: MLPPointerReadout（非線形）, option_pooling={OPTION_POOLING!r}, epochs={EPOCHS}, "
          f"support_tickets train={TICKET_TRAIN_SIZE} / held_out={TICKET_HELD_OUT_SIZE}")

    train_examples, eval_sets = _build_train_and_eval(seed=SEED)
    ticket_train = _build_ticket_examples("train", TICKET_TRAIN_SIZE, seed=SEED)
    ticket_held_out = _build_ticket_examples("held_out", TICKET_HELD_OUT_SIZE, seed=SEED)
    train_examples = train_examples + ticket_train

    torch.manual_seed(SEED)
    random.seed(SEED)
    model = JevModel(option_pooling=OPTION_POOLING)
    model.pointer = MLPPointerReadout(model.backbone.config.hidden_size)  # Gate 1-bとの唯一の違い

    print("\n=== 訓練前（OOD/IDの出発点） ===")
    before_ood = _ood_severity_report(model, ticket_held_out)
    print(f"  OOD(support_tickets held_out, n={before_ood['n']}): accuracy={before_ood['accuracy']:.4f}, "
          f"tau={before_ood['tau']:+.3f} (p={before_ood['p_value']:.3f}), "
          f"定数予測器={'YES' if before_ood['is_constant'] else 'no'}, spread={before_ood['spread']:.3f}, "
          f"ECE={before_ood['ece']:.4f}")

    print("\n=== 訓練（6データセット混合: 既存5 + support_tickets, エポックごとに選択肢順シャッフル） ===")
    start = time.perf_counter()
    train_readout(model, train_examples, epochs=EPOCHS, shuffle_options=True, seed=SEED)
    elapsed = time.perf_counter() - start
    print(f"train time: {elapsed:.1f}s")

    print("\n=== 訓練後: in-domain ===")
    id_results = {}
    for name, examples in eval_sets.items():
        r = evaluate(model, examples)
        id_results[name] = r
        print(f"  {name:10s}: accuracy={r['accuracy']:.4f}, ECE={r['ece']:.4f}")
    sst5_mae = _score_mean_abs_error(model, eval_sets["sst5"])
    print(f"  {'sst5':10s}: MAE(level)={sst5_mae:.4f}")
    rev = _reversed_order_accuracy(model, eval_sets["AG News"][:80])
    print(f"  順序反転(AG News 80件): {rev}")

    print("\n=== 訓練後: out-of-domain（support_tickets held-out, 停止条件の主判定） ===")
    after_ood = _ood_severity_report(model, ticket_held_out)
    print(f"  OOD(support_tickets held_out, n={after_ood['n']}): accuracy={after_ood['accuracy']:.4f}, "
          f"tau={after_ood['tau']:+.3f} (p={after_ood['p_value']:.3f}), "
          f"定数予測器={'YES' if after_ood['is_constant'] else 'no'}, spread={after_ood['spread']:.3f}, "
          f"ECE={after_ood['ece']:.4f}")

    _official_severity_qualitative(model)

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"mlp_pointer_gate1b_{OPTION_POOLING}_e{EPOCHS}.pt"
    torch.save(
        {
            "readout": f"MLPPointerReadout (option_pooling={OPTION_POOLING}, 仮説1検証: 非線形readout)",
            "hidden_size": model.backbone.config.hidden_size,
            "option_pooling": OPTION_POOLING,
            "datasets": ["ag_news", "banking77", "emotion", "sst5", "dbpedia14", "support_tickets"],
            "epochs": EPOCHS,
            "shuffle_options": True,
            "state_dict": model.pointer.state_dict(),
        },
        ckpt_path,
    )
    print(f"\ncheckpoint: {ckpt_path}")

    print(f"\n{'=' * 60}\n仮説1検証: 線形readout(Gate 1-b)との比較\n{'=' * 60}")
    print(f"  Gate 1-b（線形PointerReadout）: OOD tau=+0.005 (p=0.916), 定数予測器=no")
    print(f"  本実験（MLPPointerReadout）  : OOD tau={after_ood['tau']:+.3f} (p={after_ood['p_value']:.3f}), "
          f"定数予測器={'YES' if after_ood['is_constant'] else 'no'}")
    if after_ood["tau"] > 0.15 and after_ood["p_value"] < 0.1 and not after_ood["is_constant"]:
        print("  -> 非線形readoutでOOD tauが明確に改善。仮説1（線形性がボトルネック）を支持。")
    else:
        print("  -> 非線形readoutでもOOD tauは改善せず。仮説1は支持されない"
              "（readoutの表現力ではなく、データ/ラベル側の問題の可能性が高まる）。")
    print(f"  in-domain accuracy: " +
          "  ".join(f"{name}={r['accuracy']:.4f}" for name, r in id_results.items()))
    print(f"  順序反転一致率: {rev['base_rev_agreement']:.4f}")


if __name__ == "__main__":
    _main()
