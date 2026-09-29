"""Gate 1-b: severity寄り訓練データ（サポートチケットpriority, K=3）を追加して再訓練する。

設計: `internal-docs/EXP_HEAD_VS_BACKBONE.md` §2 Gate 1-b。Gate 0でseverity_tableドメインの
順序信号は凍結backboneに既に存在することを確認済み（tau+0.400・定数予測器でない）。
Gate 1-aで選ばれたpooling条件（`OPTION_POOLING`, Gate 1-a実行後に確定させる）の上に、
`data/support_tickets.py`（`Tobi-Bueck/customer-support-tickets`英語サブセット, priority
low/medium/high, K=3）を訓練データへ追加する。既存5データセット（AG News/banking77/emotion/
sst5/dbpedia14）はそのまま維持する。

停止条件（§6, 人手ラベルのみで判定）:
  1. OOD（support_tickets held-out）: Kendall tauが有意に正、定数予測器でない、予測幅が退化していない
  2. in-domain（sst5 held-out・各eval split）: accuracy/tau/ECEが退行していない
  3. 順序反転base/rev一致率が退行していない

公式Jev例（severity_table）の一致度は停止条件に含めない。構造チェック・定性比較として
併走させるのみ（§3.1）。

使い方:
    python3 -m experiments.severity_gate1b
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
from scipy.stats import kendalltau

from data.support_tickets import load_support_tickets
from data.support_tickets import to_packed_example as ticket_to_packed_example
from legacy.experiments.official_compare import SEVERITY_LEVELS
from legacy.experiments.pooling_ablation import _build_train_and_eval
from legacy.experiments.train import evaluate, expected_calibration_error, train_readout
from legacy.experiments.train_diverse import _reversed_order_accuracy, _score_mean_abs_error
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest
from legacy.model_readout.readout import PointerReadout

# Gate 1-aの結果: pooled_accuracy 0.6700 (mean) > 0.6560 (last) -> mean-poolを採用。
OPTION_POOLING = "mean"

EPOCHS = 3
SEED = 0
TICKET_TRAIN_SIZE = 1000
TICKET_HELD_OUT_SIZE = 300

SEVERITY_QUESTION_FOR_TABLE = "How severe is the reported issue?"
SEVERITY_STATES = [
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


def _build_ticket_examples(split: str, n: int, seed: int) -> list:
    rng = random.Random(seed)
    rows = load_support_tickets(split, n, seed=seed)
    return [ticket_to_packed_example(r["subject"], r["body"], r["priority"], rng) for r in rows]


def _ood_severity_report(model: JevModel, ticket_held_out: list) -> dict:
    """support_tickets held-outでOOD停止条件（§6-1）を測る。"""
    positions: list[float] = []
    golds: list[int] = []
    argmaxes: list[int] = []
    confidences: list[float] = []
    corrects: list[bool] = []
    for ex in ticket_held_out:
        result = model.forward(ex.request)[0]
        ps = result["probabilities"]
        argmax = max(range(len(ps)), key=lambda i: ps[i])
        positions.append(result["score"] - 1)
        golds.append(ex.label)
        argmaxes.append(argmax)
        confidences.append(ps[argmax])
        corrects.append(argmax == ex.label)

    tau, p_value = kendalltau(golds, positions)
    is_constant = len(set(argmaxes)) == 1
    spread = max(positions) - min(positions)
    accuracy = sum(corrects) / len(corrects)
    ece = expected_calibration_error(confidences, corrects)
    return {
        "n": len(ticket_held_out),
        "accuracy": accuracy,
        "ece": ece,
        "tau": tau,
        "p_value": p_value,
        "is_constant": is_constant,
        "spread": spread,
    }


def _official_severity_qualitative(model: JevModel) -> None:
    """公式severity_tableでの定性比較（§3.1: 停止条件には使わない、構造チェックのみ）。"""
    print("\n  --- 参考（停止条件には使わない）: 公式severity_table ---")
    branch = Branch(kind="score", instructions=SEVERITY_QUESTION_FOR_TABLE, options=list(SEVERITY_LEVELS))
    official = [score for _, score in SEVERITY_STATES]
    positions = []
    argmaxes = []
    for state, official_score in SEVERITY_STATES:
        result = model.forward(PackedRequest(state=state, branches=[branch]))[0]
        ps = result["probabilities"]
        argmax = max(range(len(ps)), key=lambda i: ps[i])
        positions.append(result["score"] - 1)
        argmaxes.append(argmax)
        print(f"    {state[:52]:<52} 公式={official_score:.2f}  ours={result['score'] - 1:.3f}  "
              f"argmax={argmax}")
    tau, p_value = kendalltau(official, positions)
    is_constant = len(set(argmaxes)) == 1
    print(f"    Kendall tau（公式・定性）: {tau:+.3f} (p={p_value:.3f})  定数予測器: "
          f"{'YES' if is_constant else 'no'}")


def _main() -> None:
    print(f"Gate 1-b: severity訓練データ追加, option_pooling={OPTION_POOLING!r}, epochs={EPOCHS}, "
          f"support_tickets train={TICKET_TRAIN_SIZE} / held_out={TICKET_HELD_OUT_SIZE}")

    train_examples, eval_sets = _build_train_and_eval(seed=SEED)
    ticket_train = _build_ticket_examples("train", TICKET_TRAIN_SIZE, seed=SEED)
    ticket_held_out = _build_ticket_examples("held_out", TICKET_HELD_OUT_SIZE, seed=SEED)
    train_examples = train_examples + ticket_train

    torch.manual_seed(SEED)
    random.seed(SEED)
    model = JevModel(option_pooling=OPTION_POOLING)
    model.pointer = PointerReadout(model.backbone.config.hidden_size)

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
    ckpt_path = ckpt_dir / f"pointer_gate1b_{OPTION_POOLING}_e{EPOCHS}.pt"
    torch.save(
        {
            "readout": f"PointerReadout (option_pooling={OPTION_POOLING}, Gate 1-b severity data)",
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

    print(f"\n{'=' * 60}\nGate 1-b 停止条件（§6）\n{'=' * 60}")
    ood_ok = (
        after_ood["tau"] > 0
        and not after_ood["is_constant"]
        and after_ood["spread"] > 0.05
    )
    print(f"  1. OOD tau有意に正・定数予測器でない・予測幅退化なし: "
          f"tau={after_ood['tau']:+.3f}, 定数予測器={'YES' if after_ood['is_constant'] else 'no'}, "
          f"spread={after_ood['spread']:.3f}  -> {'満たす' if ood_ok else '満たさない'}")
    print(f"  2. in-domain accuracy: " +
          "  ".join(f"{name}={r['accuracy']:.4f}" for name, r in id_results.items()))
    print(f"  3. 順序反転一致率: {rev['base_rev_agreement']:.4f}")
    print("\n（既存pointer_diverse5_e3.ptとの比較は結果ファイルに人手で記録する）")


if __name__ == "__main__":
    _main()
