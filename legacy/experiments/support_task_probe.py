"""Gold V1と同構造の英語データに対し、凍結backboneのzero-shot信号を測る（学習前のQ1）。

候補:
  Choice: `data/ticket_queue.py`（サポートチケットの部署振り分け、説明付きK=3〜4）
  Score:  `data/anger_intensity.py`（怒りの強度 low/medium/high、Gold V1 frustrationと同じ向き）
report_qualityに相当する公開データはHF検索で見つからず、本プローブの対象外。

手法は`experiments/backbone_probe.py`と同じ（地の文プロンプト、選択肢teacher-forcing平均
log確率、null state校正）。Gold V1は読まない。重み・optimizerは作らない。

判定（Gate 1-cと同じ形）:
  Choice: top1がランダム基準（平均1/K）を片側二項検定で上回る（p<0.05）
  Score: Kendall τが正かつp<0.05、かつ定数予測器でない
"""

from __future__ import annotations

import math
import random
from collections import Counter

import torch
from scipy.stats import binomtest, kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.anger_intensity import LEVEL_ORDER, load_anger_intensity_stratified, to_packed_example as anger_to_packed
from data.ticket_queue import load_ticket_queue_stratified, to_packed_example as queue_to_packed
from legacy.experiments.backbone_probe import NULL_STATE, _calibrated, _option_avg_log_probs, _prose_cue_prompt
from legacy.model_readout.backbone import MODEL_NAME
from model.packer import Branch

QUEUE_N_PER = 20  # 5 queues × 20 = 100
ANGER_N_PER = 30  # 3 levels × 30 = 90
SEED = 0


def _softmax_log(log_probs: list[float]) -> list[float]:
    highest = max(log_probs)
    weights = [math.exp(lp - highest) for lp in log_probs]
    total = sum(weights)
    return [w / total for w in weights]


def _score_branch(causal_lm, tokenizer, state: str, branch: Branch, bias_cache: dict) -> tuple[list[float], list[float]]:
    options = list(branch.options)
    template = (branch.instructions, tuple(options))
    if template not in bias_cache:
        null_prompt = _prose_cue_prompt(NULL_STATE, branch.instructions, options)
        bias_cache[template] = _option_avg_log_probs(causal_lm, tokenizer, null_prompt, options)
    log_probs = _option_avg_log_probs(
        causal_lm, tokenizer, _prose_cue_prompt(state, branch.instructions, options), options
    )
    return _softmax_log(log_probs), _calibrated(log_probs, bias_cache[template])


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def probe_queue(causal_lm, tokenizer) -> None:
    rng = random.Random(SEED)
    ds = load_ticket_queue_stratified(QUEUE_N_PER, seed=SEED)
    bias_cache: dict = {}
    rows = []
    for row in ds:
        example = queue_to_packed(row["subject"], row["body"], row["queue"], rng)
        branch = example.request.branches[0]
        raw, cal = _score_branch(causal_lm, tokenizer, example.request.state, branch, bias_cache)
        rows.append((example.label, len(branch.options), raw, cal))

    chance = sum(1 / k for _, k, _, _ in rows) / len(rows)
    for name, idx in (("raw", 2), ("cal", 3)):
        golds = [r[0] for r in rows]
        preds = [_argmax(r[idx]) for r in rows]
        correct = sum(g == p for g, p in zip(golds, preds))
        n = len(rows)
        p_value = binomtest(correct, n, p=chance, alternative="greater").pvalue
        positions = Counter(preds)
        mode = positions.most_common(1)[0][1]
        print(f"  queue Choice {name}: n={n} top1={correct / n:.3f} chance={chance:.3f} "
              f"p={p_value:.4f} argmax={dict(sorted(positions.items()))} mode={mode}/{n}")
        print(f"    採用: top1>chance かつ p<0.05 -> {correct / n > chance and p_value < 0.05}")


def probe_anger(causal_lm, tokenizer) -> None:
    rng = random.Random(SEED)
    ds = load_anger_intensity_stratified(ANGER_N_PER, seed=SEED)
    bias_cache: dict = {}
    golds: list[int] = []
    raw_scores: list[float] = []
    cal_scores: list[float] = []
    raw_argmax: list[int] = []
    cal_argmax: list[int] = []
    for row in ds:
        example = anger_to_packed(row["tweet"], row["sentiment_intensity"], rng)
        branch = example.request.branches[0]
        raw, cal = _score_branch(causal_lm, tokenizer, example.request.state, branch, bias_cache)
        golds.append(example.label)
        raw_scores.append(sum(i * p for i, p in enumerate(raw)))
        cal_scores.append(sum(i * p for i, p in enumerate(cal)))
        raw_argmax.append(_argmax(raw))
        cal_argmax.append(_argmax(cal))

    n = len(golds)
    for name, scores, argmaxes in (("raw", raw_scores, raw_argmax), ("cal", cal_scores, cal_argmax)):
        tau, p_value = kendalltau(golds, scores)
        spread = max(scores) - min(scores)
        unique = len(set(round(s, 4) for s in scores))
        constant = unique == 1
        top1 = sum(g == a for g, a in zip(golds, argmaxes)) / n
        print(f"  anger Score {name}: n={n} top1={top1:.3f} tau={tau:+.3f} p={p_value:.4f} "
              f"spread={spread:.3f} unique={unique} constant={constant} argmax={dict(sorted(Counter(argmaxes).items()))}")
        print(f"    採用: tau>0 かつ p<0.05 かつ not constant -> {tau > 0 and p_value < 0.05 and not constant}")


def _main() -> None:
    print(f"model: {MODEL_NAME}（凍結・LM head、readoutなし）")
    print("report_quality相当の公開データは見つからず、本プローブの対象外。")
    causal_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    causal_lm.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\n=== Choice: ticket queue（n_per_queue={QUEUE_N_PER} × 5） ===")
    probe_queue(causal_lm, tokenizer)
    print(f"\n=== Score: anger intensity（n_per_level={ANGER_N_PER} × 3） ===")
    probe_anger(causal_lm, tokenizer)


if __name__ == "__main__":
    _main()
