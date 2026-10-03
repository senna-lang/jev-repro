"""Gate 1-c: bugzilla severity（実バグ報告の技術的深刻度、K=6）を訓練データに追加する。

Gate 1-b（`support_tickets`のpriority）はOOD Kendall tauが実質ゼロのままで停止条件を
満たさず、並行検証（仮説1: 非線形readout／仮説2: zero-shotラベル相関）の結果、
「`priority`ラベルがそもそもテキストと弱くしか相関していない」ことが主因と判明した
（`results/logs/2026-09-23_gate1_pooling_and_severity_data.md`）。

本Gateでは、`data/bugzilla_severity.py`（`AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset`、
実際のバグトリアージのSeverity Label: blocker/critical/major/normal/minor/trivial, K=6）を
severity代理データとして使う。訓練前に`experiments/bugzilla_severity_probe.py`でQ1を確認済み
（zero-shot tau 校正後+0.392, p=0.000, n=60, 定数予測器でない — Gate 0のseverity_table
（tau+0.400・n=5）と同等以上の強さ、かつ統計的に有意）。

severity分布が'normal'に81%偏っているため、訓練・held-out双方とも
`load_bugzilla_severity_stratified`でレベルごとに均等にサンプルする
（訓練=前半half="first"、held-out=後半half="second"、重複なし）。

Gate 1-a選定結果のmean-poolを使い、既存5データセット + bugzilla_severityの6データセット
混合で訓練する。評価はGate 1-bと同じ停止条件（§6）。

使い方:
    python3 -m experiments.severity_gate1c
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from data.bugzilla_severity import load_bugzilla_severity_stratified
from data.bugzilla_severity import to_packed_example as bug_to_packed_example
from legacy.experiments.pooling_ablation import _build_train_and_eval
from legacy.experiments.severity_gate1b import OPTION_POOLING, _official_severity_qualitative, _ood_severity_report
from legacy.experiments.train import evaluate, train_readout
from legacy.experiments.train_diverse import _reversed_order_accuracy, _score_mean_abs_error
from legacy.model_readout.forward import JevModel
from legacy.model_readout.readout import PointerReadout

EPOCHS = 3
SEED = 0
BUG_TRAIN_PER_LEVEL = 300  # 6 levels x 300 = 1800件
BUG_HELD_OUT_PER_LEVEL = 50  # 6 levels x 50 = 300件（support_ticketsと同規模で比較可能に）


def _build_bug_examples(n_per_level: int, half: str, seed: int) -> list:
    rng = random.Random(seed)
    rows = load_bugzilla_severity_stratified(n_per_level, seed=seed, half=half)
    return [bug_to_packed_example(r["Short Description"], r["Severity Label"], rng) for r in rows]


def _main() -> None:
    print(f"Gate 1-c: bugzilla severity訓練データ追加, option_pooling={OPTION_POOLING!r}, epochs={EPOCHS}, "
          f"train={BUG_TRAIN_PER_LEVEL}/level x 6, held_out={BUG_HELD_OUT_PER_LEVEL}/level x 6")

    train_examples, eval_sets = _build_train_and_eval(seed=SEED)
    bug_train = _build_bug_examples(BUG_TRAIN_PER_LEVEL, "first", seed=SEED)
    bug_held_out = _build_bug_examples(BUG_HELD_OUT_PER_LEVEL, "second", seed=SEED)
    train_examples = train_examples + bug_train

    torch.manual_seed(SEED)
    random.seed(SEED)
    model = JevModel(option_pooling=OPTION_POOLING)
    model.pointer = PointerReadout(model.backbone.config.hidden_size)

    print("\n=== 訓練前（OOD/IDの出発点） ===")
    before_ood = _ood_severity_report(model, bug_held_out)
    print(f"  OOD(bugzilla severity held_out, n={before_ood['n']}): accuracy={before_ood['accuracy']:.4f}, "
          f"tau={before_ood['tau']:+.3f} (p={before_ood['p_value']:.3f}), "
          f"定数予測器={'YES' if before_ood['is_constant'] else 'no'}, spread={before_ood['spread']:.3f}, "
          f"ECE={before_ood['ece']:.4f}")

    print("\n=== 訓練（6データセット混合: 既存5 + bugzilla severity, エポックごとに選択肢順シャッフル） ===")
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

    print("\n=== 訓練後: out-of-domain（bugzilla severity held-out, 停止条件の主判定） ===")
    after_ood = _ood_severity_report(model, bug_held_out)
    print(f"  OOD(bugzilla severity held_out, n={after_ood['n']}): accuracy={after_ood['accuracy']:.4f}, "
          f"tau={after_ood['tau']:+.3f} (p={after_ood['p_value']:.3f}), "
          f"定数予測器={'YES' if after_ood['is_constant'] else 'no'}, spread={after_ood['spread']:.3f}, "
          f"ECE={after_ood['ece']:.4f}")

    _official_severity_qualitative(model)

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"pointer_gate1c_{OPTION_POOLING}_e{EPOCHS}.pt"
    torch.save(
        {
            "readout": f"PointerReadout (option_pooling={OPTION_POOLING}, Gate 1-c bugzilla severity data)",
            "hidden_size": model.backbone.config.hidden_size,
            "option_pooling": OPTION_POOLING,
            "datasets": ["ag_news", "banking77", "emotion", "sst5", "dbpedia14", "bugzilla_severity"],
            "epochs": EPOCHS,
            "shuffle_options": True,
            "state_dict": model.pointer.state_dict(),
        },
        ckpt_path,
    )
    print(f"\ncheckpoint: {ckpt_path}")

    print(f"\n{'=' * 60}\nGate 1-c 停止条件（§6）\n{'=' * 60}")
    ood_ok = (
        after_ood["tau"] > 0
        and after_ood["p_value"] < 0.1
        and not after_ood["is_constant"]
        and after_ood["spread"] > 0.05
    )
    print(f"  1. OOD tau有意に正・定数予測器でない・予測幅退化なし: "
          f"tau={after_ood['tau']:+.3f} (p={after_ood['p_value']:.3f}), "
          f"定数予測器={'YES' if after_ood['is_constant'] else 'no'}, "
          f"spread={after_ood['spread']:.3f}  -> {'満たす' if ood_ok else '満たさない'}")
    print(f"  2. in-domain accuracy: " +
          "  ".join(f"{name}={r['accuracy']:.4f}" for name, r in id_results.items()))
    print(f"  3. 順序反転一致率: {rev['base_rev_agreement']:.4f}")
    print(f"\n  比較: Gate 1-b（support_tickets priority）OOD tau=+0.005 (p=0.916)")
    print(f"       Gate 1-c（bugzilla severity）      OOD tau={after_ood['tau']:+.3f} (p={after_ood['p_value']:.3f})")


if __name__ == "__main__":
    _main()
