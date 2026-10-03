"""仮説2の検証: `support_tickets`のpriorityラベルに対する凍結backboneのzero-shot信号を測る。

Gate 1-b（`results/logs/2026-09-23_gate1_pooling_and_severity_data.md`）で、support_tickets priority
データを訓練に追加してもOOD Kendall tauが実質ゼロ（+0.005, p=0.916）のまま動かなかった。
2つの仮説が残る:

  (1) 信号は凍結backboneにあるが線形readoutでは取り出せない
      → `experiments/mlp_readout_ablation.py`で検証（非線形readout）
  (2) `priority`ラベルの軸がテキスト内容とずれている（業務優先度はSLA・顧客ランク等
      テキスト外要因に依存しうる）ため、そもそもテキストから予測しにくい
      → 本スクリプトで検証（readoutを一切通さないzero-shot probe）

`experiments/backbone_probe.py`と同じ手法（`model/packer.py::pack()`と同一の地の文
プロンプト、teacher-forcing平均log確率によるoption scoring、null state校正）を
support_tickets held-outサンプルに適用する。zero-shot tauが severity_table（+0.400）と
同程度に正であれば仮説(2)を支持（labelの軸自体に問題は無く、readout/訓練側の問題）。
zero-shot tauも0に近ければ仮説(2)を支持（priorityというラベル自体がテキストから
予測困難）。

使い方:
    python3 -m experiments.support_tickets_probe
"""

from __future__ import annotations

import math

import torch
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.support_tickets import LEVEL_ORDER, load_support_tickets
from legacy.experiments.backbone_probe import _calibrated, _option_avg_log_probs
from legacy.model_readout.backbone import MODEL_NAME

QUESTION = "How severe is the issue described in this support ticket?"
NULL_STATE = "N/A"
N_SAMPLES = 60
MAX_BODY_CHARS = 600


def _state_text(subject: str, body: str) -> str:
    subject = subject or ""
    body = body or ""
    return f"{subject}\n\n{body}"[: MAX_BODY_CHARS + len(subject) + 2]


def _prose_cue_prompt(state: str) -> str:
    joined = ", ".join(LEVEL_ORDER[:-1]) + ", " + LEVEL_ORDER[-1] + "."
    return f"{state}\n\n{QUESTION} Answer with one word — {joined}\nAnswer:"


def _main() -> None:
    print(f"model: {MODEL_NAME}（凍結・LM head込み、readoutは通さない）")
    causal_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    causal_lm.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\n=== 仮説2検証: support_tickets priorityのzero-shot信号（n={N_SAMPLES}, "
          f"held_out split） ===")
    rows = load_support_tickets("held_out", N_SAMPLES, seed=0)

    null_prompt = _prose_cue_prompt(NULL_STATE)
    bias = _option_avg_log_probs(causal_lm, tokenizer, null_prompt, LEVEL_ORDER)

    golds: list[int] = []
    raw_positions: list[float] = []
    calibrated_positions: list[float] = []
    raw_argmax: list[int] = []
    calibrated_argmax: list[int] = []
    priority_to_label = {"low": 0, "medium": 1, "high": 2}

    for row in rows:
        state = _state_text(row["subject"], row["body"])
        prompt = _prose_cue_prompt(state)
        log_probs = _option_avg_log_probs(causal_lm, tokenizer, prompt, LEVEL_ORDER)
        exp_probs = [math.exp(p) for p in log_probs]
        total = sum(exp_probs)
        raw_probs = [p / total for p in exp_probs]
        calibrated_probs = _calibrated(log_probs, bias)

        gold = priority_to_label[row["priority"]]
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

    print(f"  n={len(golds)}, ラベル分布: "
          f"low={golds.count(0)} medium={golds.count(1)} high={golds.count(2)}")
    print(f"  tau: 校正前 {raw_tau:+.3f} (p={raw_p:.3f}) / 校正後 {calibrated_tau:+.3f} (p={calibrated_p:.3f})")
    print(f"  accuracy: 校正前 {raw_acc:.3f} / 校正後 {calibrated_acc:.3f}  (チャンス=0.333)")
    print(f"  定数予測器: 校正前 {'YES' if raw_constant else 'no'} / 校正後 {'YES' if calibrated_constant else 'no'}")
    print(f"  予測幅(max-min): 校正前 {raw_spread:.3f} / 校正後 {calibrated_spread:.3f}")

    print("\n=== 判定 ===")
    if calibrated_tau > 0.15 and not calibrated_constant:
        print("  zero-shot tauが正方向 -> 信号はbackboneにある。Gate 1-bのtau≈0はreadout/訓練側の"
              "問題（仮説1寄り）を支持。")
    else:
        print("  zero-shot tauも0に近い/定数予測器 -> priorityラベル自体がテキストから予測しにくい"
              "可能性が高い（仮説2寄り）。severity_tableとpriorityのラベル軸のずれが疑われる。")


if __name__ == "__main__":
    _main()
