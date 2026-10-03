"""Q1検証: Bugzilla severityラベル（技術的深刻度）に対する凍結backboneのzero-shot信号を測る。

`support_tickets`のpriorityで同じ手順を省略して無駄な訓練を回した反省
（`results/logs/2026-09-23_gate1_pooling_and_severity_data.md` Gate 1-b）を踏まえ、
訓練データとして採用する前にQ1（テキストとラベルの相関が凍結backboneに既に見えるか）を
zero-shot probeで確認する。`experiments/backbone_probe.py`と同じ手法。

severityが'normal'に81%偏っているため、`load_bugzilla_severity_stratified`で
レベルごとに均等なサンプルを取り、Kendall tauを測る。

使い方:
    python3 -m experiments.bugzilla_severity_probe
"""

from __future__ import annotations

import math

import torch
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.bugzilla_severity import SEVERITY_ORDER, load_bugzilla_severity_stratified
from legacy.experiments.backbone_probe import _calibrated, _option_avg_log_probs
from legacy.model_readout.backbone import MODEL_NAME

QUESTION = "How severe is the bug described in this report?"
NULL_STATE = "N/A"
N_PER_LEVEL = 10


def _prose_cue_prompt(state: str) -> str:
    joined = ", ".join(SEVERITY_ORDER[:-1]) + ", " + SEVERITY_ORDER[-1] + "."
    return f"{state}\n\n{QUESTION} Answer with one word — {joined}\nAnswer:"


def _main() -> None:
    print(f"model: {MODEL_NAME}（凍結・LM head込み、readoutは通さない）")
    causal_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    causal_lm.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\n=== Q1検証: bugzilla severityのzero-shot信号（n_per_level={N_PER_LEVEL}, "
          f"K=6, 均等サンプル） ===")
    rows = load_bugzilla_severity_stratified(N_PER_LEVEL, seed=0)

    null_prompt = _prose_cue_prompt(NULL_STATE)
    bias = _option_avg_log_probs(causal_lm, tokenizer, null_prompt, SEVERITY_ORDER)

    level_to_index = {name: i for i, name in enumerate(SEVERITY_ORDER)}
    golds: list[int] = []
    raw_positions: list[float] = []
    calibrated_positions: list[float] = []
    raw_argmax: list[int] = []
    calibrated_argmax: list[int] = []

    for row in rows:
        state = row["Short Description"]
        prompt = _prose_cue_prompt(state)
        log_probs = _option_avg_log_probs(causal_lm, tokenizer, prompt, SEVERITY_ORDER)
        exp_probs = [math.exp(p) for p in log_probs]
        total = sum(exp_probs)
        raw_probs = [p / total for p in exp_probs]
        calibrated_probs = _calibrated(log_probs, bias)

        gold = level_to_index[row["Severity Label"]]
        golds.append(gold)
        raw_positions.append(sum(i * p for i, p in enumerate(raw_probs)))
        calibrated_positions.append(sum(i * p for i, p in enumerate(calibrated_probs)))
        raw_argmax.append(max(range(len(raw_probs)), key=lambda i: raw_probs[i]))
        calibrated_argmax.append(max(range(len(calibrated_probs)), key=lambda i: calibrated_probs[i]))

    raw_tau, raw_p = kendalltau(golds, raw_positions)
    calibrated_tau, calibrated_p = kendalltau(golds, calibrated_positions)
    raw_constant = len(set(raw_argmax)) == 1
    calibrated_constant = len(set(calibrated_argmax)) == 1
    raw_spread = max(raw_positions) - min(raw_positions)
    calibrated_spread = max(calibrated_positions) - min(calibrated_positions)
    raw_acc = sum(int(a == g) for a, g in zip(raw_argmax, golds)) / len(golds)
    calibrated_acc = sum(int(a == g) for a, g in zip(calibrated_argmax, golds)) / len(golds)

    print(f"  n={len(golds)} ({N_PER_LEVEL}/level x {len(SEVERITY_ORDER)} levels)")
    print(f"  tau: 校正前 {raw_tau:+.3f} (p={raw_p:.3f}) / 校正後 {calibrated_tau:+.3f} (p={calibrated_p:.3f})")
    print(f"  accuracy: 校正前 {raw_acc:.3f} / 校正後 {calibrated_acc:.3f}  (チャンス={1/len(SEVERITY_ORDER):.3f})")
    print(f"  定数予測器: 校正前 {'YES' if raw_constant else 'no'} / 校正後 {'YES' if calibrated_constant else 'no'}")
    print(f"  予測幅(max-min): 校正前 {raw_spread:.3f} / 校正後 {calibrated_spread:.3f}")

    print("\n=== 判定 ===")
    if calibrated_tau > 0.15 and calibrated_p < 0.1 and not calibrated_constant:
        print("  zero-shot tauが正方向かつ有意 -> Q1真。信号はbackboneにある。"
              "訓練データとして採用してよい（Gate 1-c候補）。")
    else:
        print("  zero-shot tauが弱い/非有意/定数予測器 -> このデータソースもseverityの"
              "代理として不適切な可能性。別のデータソースを探すか、Gate 2/3を検討。")


if __name__ == "__main__":
    _main()
