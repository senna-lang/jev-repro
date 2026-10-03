"""Gold V1のChoice/ScoreをLM head zero-shot（readoutを通さない）で測り、学習済みpointerと比べる。

`results/logs/2026-09-24_language_mismatch_hypothesis_test.md`とnull state校正（`evaluation/run.py
--null-calibration`）の結果、Gold V1でScoreがランダム以下なのは加法バイアスでも言語でもないと
分かった。残る切り分け:
    (a) 凍結backboneは信号を持つが、readoutが取り出せていない
    (b) backboneがそもそもGold V1の設問に信号を持っていない
Gate 0と同じく、`experiments/backbone_probe.py`の地の文プロンプト・選択肢teacher-forcing平均
log確率・null state校正をそのまま再利用し、指標は`evaluation.run.score_quality`で出す
（評価パイプラインと同一定義）。Gold V1は評価のみに使い、重み・optimizerは一切作らない。

使い方:
    python3 -m experiments.gold_v1_zeroshot_probe
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.run import _normalise_result, score_quality
from evaluation.schema import BenchmarkCase, load_cases
from legacy.experiments.backbone_probe import NULL_STATE, _calibrated, _option_avg_log_probs, _prose_cue_prompt
from legacy.model_readout.backbone import MODEL_NAME

DATASETS = {
    "JA": Path("evaluation/private/gold_v1_adjudicated.jsonl"),
    "EN": Path("evaluation/private/gold_v1_adjudicated_en_mt.jsonl"),
}
POINTER_PREDICTIONS = {
    ("JA", "raw"): Path("/tmp/ja_nocal_predictions.jsonl"),
    ("JA", "cal"): Path("/tmp/ja_cal_predictions.jsonl"),
    ("EN", "raw"): Path("/tmp/en_mt_predictions.jsonl"),
    ("EN", "cal"): Path("/tmp/en_cal_predictions.jsonl"),
}

Outputs = dict[tuple[str, str], dict]


def _choice_score_cases(cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
    """noul branchを落としたcase（noulはこのプローブの範囲外）。"""
    kept = []
    for case in cases:
        branches = tuple(b for b in case.branches if b.kind != "noul")
        if branches:
            kept.append(BenchmarkCase(case.id, case.state, case.tags, case.label_status, branches))
    return kept


def _softmax_log(log_probs: list[float]) -> list[float]:
    highest = max(log_probs)
    weights = [math.exp(lp - highest) for lp in log_probs]
    total = sum(weights)
    return [w / total for w in weights]


def zero_shot_outputs(causal_lm, tokenizer, cases: list[BenchmarkCase]) -> tuple[Outputs, Outputs]:
    raw: Outputs = {}
    cal: Outputs = {}
    bias_cache: dict[tuple[str, tuple[str, ...]], list[float]] = {}
    for case in cases:
        for branch in case.branches:
            options = list(branch.options)
            template = (branch.instructions, tuple(options))
            if template not in bias_cache:
                null_prompt = _prose_cue_prompt(NULL_STATE, branch.instructions, options)
                bias_cache[template] = _option_avg_log_probs(causal_lm, tokenizer, null_prompt, options)
            prompt = _prose_cue_prompt(case.state, branch.instructions, options)
            log_probs = _option_avg_log_probs(causal_lm, tokenizer, prompt, options)
            key = (case.id, branch.id)
            context = f"{case.id}/{branch.id}"
            raw[key] = _normalise_result(branch, {"kind": branch.kind, "probabilities": _softmax_log(log_probs)}, context)
            cal[key] = _normalise_result(
                branch, {"kind": branch.kind, "probabilities": _calibrated(log_probs, bias_cache[template])}, context
            )
    return raw, cal


def load_pointer_outputs(path: Path, cases: list[BenchmarkCase]) -> Outputs:
    branches = {(c.id, b.id): b for c in cases for b in c.branches}
    outputs: Outputs = {}
    for line in path.open():
        row = json.loads(line)
        key = (row["case_id"], row["branch_id"])
        if key in branches:
            outputs[key] = _normalise_result(branches[key], row, f"{key[0]}/{key[1]}")
    assert set(outputs) == set(branches), f"{path}: branch集合が一致しない"
    return outputs


def _argmax(values) -> int:
    return max(range(len(values)), key=values.__getitem__)


def diagnostics(cases: list[BenchmarkCase], outputs: Outputs) -> dict:
    quality = score_quality(cases, outputs)
    positions: Counter = Counter()
    per_question: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    predicted_scores: list[float] = []
    for case in cases:
        for branch in case.branches:
            probabilities = outputs[(case.id, branch.id)]["probabilities"]
            if branch.kind == "choice":
                positions[f"K{len(branch.options)}:{_argmax(probabilities)}"] += 1
            else:
                predicted = sum((i + 1) * p for i, p in enumerate(probabilities))
                target = sum((i + 1) * p for i, p in enumerate(branch.target_probabilities))
                predicted_scores.append(predicted)
                per_question[branch.id][0].append(target)
                per_question[branch.id][1].append(predicted)
    taus = {}
    for question, (targets, predicted) in per_question.items():
        tau, p_value = kendalltau(targets, predicted)
        taus[question] = {"n": len(targets), "tau": tau, "p": p_value}
    return {
        "choice_top1": quality["choice"]["top1_accuracy"],
        "choice_brier": quality["choice"]["brier"],
        "score_top1": quality["score"]["top1_accuracy"],
        "score_mae": quality["score"]["expected_score_mae"],
        "score_pred_sd": statistics.pstdev(predicted_scores),
        "score_tau": taus,
        "choice_argmax_positions": dict(sorted(positions.items())),
    }


def _format_row(label: str, d: dict) -> str:
    taus = "  ".join(f"{q}={v['tau']:+.3f}(p={v['p']:.3f})" for q, v in d["score_tau"].items())
    return (f"{label:22s} choice_top1={d['choice_top1']:.3f} brier={d['choice_brier']:.3f} | "
            f"score_top1={d['score_top1']:.3f} mae={d['score_mae']:.3f} sd={d['score_pred_sd']:.2f} | {taus}\n"
            f"{'':22s} choice argmax位置: {d['choice_argmax_positions']}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace causal LM id（既定は現行backbone）")
    args = parser.parse_args()
    model_name = args.model
    tag = model_name.rsplit("/", 1)[-1].replace(".", "")
    print(f"model: {model_name}（凍結・LM head、readoutなし）")
    causal_lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    causal_lm.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    loaded = {lang: _choice_score_cases(load_cases(path)) for lang, path in DATASETS.items()}
    ids = {lang: [(c.id, tuple(b.id for b in c.branches)) for c in cases] for lang, cases in loaded.items()}
    assert ids["JA"] == ids["EN"], "JA/ENのcase・branch IDが一致しない"

    for lang, cases in loaded.items():
        raw, cal = zero_shot_outputs(causal_lm, tokenizer, cases)
        for variant, outputs in (("raw", raw), ("cal", cal)):
            with open(f"/tmp/zs_{tag}_{lang.lower()}_{variant}.jsonl", "w") as handle:
                for (case_id, branch_id), result in outputs.items():
                    handle.write(json.dumps({"case_id": case_id, "branch_id": branch_id, **result}, ensure_ascii=False) + "\n")
        print(f"\n=== {lang} {model_name}（choice n={sum(b.kind == 'choice' for c in cases for b in c.branches)}, "
              f"score n={sum(b.kind == 'score' for c in cases for b in c.branches)}） ===")
        print(_format_row("zero-shot raw", diagnostics(cases, raw)))
        print(_format_row("zero-shot cal", diagnostics(cases, cal)))
        for variant in ("raw", "cal"):
            pointer = load_pointer_outputs(POINTER_PREDICTIONS[(lang, variant)], cases)
            print(_format_row(f"trained pointer {variant}", diagnostics(cases, pointer)))


if __name__ == "__main__":
    _main()
