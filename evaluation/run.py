"""Evaluate a Jev-style model against versioned targets, references, and invariants.

Usage:
    python3 -m evaluation.run --validate
    python3 -m evaluation.run --checkpoint checkpoints/pointer_diverse_...pt
    python3 -m evaluation.run --checkpoint ... --robustness --measure-latency
    python3 -m evaluation.run --checkpoint ... --reference evaluation/private/jev.jsonl

A reference JSONL row is ``{"case_id", "branch_id", "kind", ...output fields}``,
which is also the format produced by ``--write-predictions``. References need not be
Jev; they can be another local checkpoint. The runner never treats agreement with a
reference as ground truth: quality is always measured against the target distribution.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch

from evaluation.schema import BenchmarkBranch, BenchmarkCase, load_cases
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest

DEFAULT_DATASET = Path(__file__).with_name("data") / "seed_v1.jsonl"
EPSILON = 1e-12


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _validate_distribution(probabilities: Any, expected_len: int, context: str) -> list[float]:
    if not isinstance(probabilities, list) or len(probabilities) != expected_len:
        raise ValueError(f"{context}: expected {expected_len} probabilities")
    result = [float(value) for value in probabilities]
    if any(not 0.0 <= value <= 1.0 for value in result) or abs(sum(result) - 1.0) > 1e-5:
        raise ValueError(f"{context}: probabilities must be within [0, 1] and sum to 1")
    return result


def _normalise_result(branch: BenchmarkBranch, result: dict[str, Any], context: str) -> dict[str, Any]:
    if result.get("kind") != branch.kind:
        raise ValueError(f"{context}: expected kind={branch.kind}, got {result.get('kind')!r}")
    if branch.kind == "noul":
        value = result.get("p_true")
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{context}: noul p_true must be within [0, 1]")
        return {"kind": "noul", "p_true": float(value)}

    probabilities = _validate_distribution(result.get("probabilities"), len(branch.options), context)
    normalised = {"kind": branch.kind, "probabilities": probabilities}
    if branch.kind == "score":
        normalised["score"] = sum((index + 1) * value for index, value in enumerate(probabilities))
    return normalised


# Content-free states; their outputs are the text-independent per-option preference
# (same definition as `experiments/score_diagnose.py::NULL_STATES`, validated there on sst5).
NULL_STATES = ("N/A", "...")


def _null_log_probs(model: JevModel, branch: Branch) -> list[float]:
    """Mean log-probability over `NULL_STATES` for this branch alone = its option bias."""
    per_state = []
    for state in NULL_STATES:
        probabilities = model.forward(PackedRequest(state=state, branches=[branch]))[0]["probabilities"]
        per_state.append([math.log(max(p, EPSILON)) for p in probabilities])
    return [sum(column) / len(column) for column in zip(*per_state)]


def _null_calibrated(probabilities: list[float], bias: list[float]) -> list[float]:
    """softmax(log p - bias); equals softmax(z(state) - z(null)) because softmax is shift-invariant."""
    adjusted = [math.log(max(p, EPSILON)) - b for p, b in zip(probabilities, bias)]
    highest = max(adjusted)
    weights = [math.exp(a - highest) for a in adjusted]
    total = sum(weights)
    return [w / total for w in weights]


def run_model(
    model: JevModel, cases: list[BenchmarkCase], null_calibration: bool = False
) -> dict[tuple[str, str], dict[str, Any]]:
    """Run every benchmark request once; retain identifiers rather than positional output.

    `null_calibration=True` subtracts each choice/score branch's content-free option bias
    (ROADMAP #3-4 pathology). The tree mask isolates branches, so the bias is computed on a
    single-branch null request and cached per (kind, instructions, options) template.
    """
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    bias_cache: dict[tuple[str, str, tuple[str, ...]], list[float]] = {}
    for case in cases:
        results = model.forward(case.request())
        if len(results) != len(case.branches):
            raise ValueError(f"{case.id}: model returned {len(results)} results for {len(case.branches)} branches")
        for branch, result in zip(case.branches, results):
            if null_calibration and branch.kind != "noul":
                template = (branch.kind, branch.instructions, tuple(branch.options))
                if template not in bias_cache:
                    bias_cache[template] = _null_log_probs(model, branch.request_branch())
                result = {**result, "probabilities": _null_calibrated(result["probabilities"], bias_cache[template])}
            key = (case.id, branch.id)
            outputs[key] = _normalise_result(branch, result, f"{case.id}/{branch.id}")
    return outputs


def target_output(branch: BenchmarkBranch) -> dict[str, Any]:
    """A perfect model output used by metric self-checks and never by model evaluation."""
    if branch.kind == "noul":
        assert branch.target_p_true is not None
        return {"kind": "noul", "p_true": branch.target_p_true}
    assert branch.target_probabilities is not None
    probabilities = list(branch.target_probabilities)
    result: dict[str, Any] = {"kind": branch.kind, "probabilities": probabilities}
    if branch.kind == "score":
        result["score"] = sum((index + 1) * value for index, value in enumerate(probabilities))
    return result


def _distribution_metrics(predicted: list[float], target: tuple[float, ...]) -> dict[str, float]:
    return {
        "brier": sum((actual - desired) ** 2 for actual, desired in zip(predicted, target)),
        "nll": -sum(desired * math.log(max(actual, EPSILON)) for actual, desired in zip(predicted, target)),
    }


def score_quality(cases: list[BenchmarkCase], outputs: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Score outputs against targets; soft targets support multi-annotator vote distributions."""
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for case in cases:
        for branch in case.branches:
            result = outputs[(case.id, branch.id)]
            if branch.kind == "noul":
                assert branch.target_p_true is not None
                predicted = result["p_true"]
                target = branch.target_p_true
                values["noul"]["brier"].append((predicted - target) ** 2)
                values["noul"]["nll"].append(
                    -(target * math.log(max(predicted, EPSILON)) + (1 - target) * math.log(max(1 - predicted, EPSILON)))
                )
                values["noul"]["threshold_accuracy"].append(float((predicted >= 0.5) == (target >= 0.5)))
                continue

            assert branch.target_probabilities is not None
            probabilities = result["probabilities"]
            distribution = _distribution_metrics(probabilities, branch.target_probabilities)
            values[branch.kind]["brier"].append(distribution["brier"])
            values[branch.kind]["nll"].append(distribution["nll"])
            values[branch.kind]["top1_accuracy"].append(
                float(_argmax(probabilities) == _argmax(list(branch.target_probabilities)))
            )

            if branch.kind == "score":
                predicted_score = sum((index + 1) * value for index, value in enumerate(probabilities))
                target_score = sum((index + 1) * value for index, value in enumerate(branch.target_probabilities))
                values["score"]["expected_score_mae"].append(abs(predicted_score - target_score))
                predicted_cdf = 0.0
                target_cdf = 0.0
                emd = 0.0
                for predicted_probability, target_probability in zip(probabilities[:-1], branch.target_probabilities[:-1]):
                    predicted_cdf += predicted_probability
                    target_cdf += target_probability
                    emd += abs(predicted_cdf - target_cdf)
                values["score"]["normalised_emd"].append(emd / (len(probabilities) - 1))

    return {
        kind: {"n": len(next(iter(metric_values.values()))), **{name: _mean(samples) for name, samples in metric_values.items()}}
        for kind, metric_values in sorted(values.items())
    }


def _js_divergence(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]
    return 0.5 * sum(a * math.log(max(a, EPSILON) / max(m, EPSILON)) for a, m in zip(left, midpoint)) + 0.5 * sum(
        b * math.log(max(b, EPSILON) / max(m, EPSILON)) for b, m in zip(right, midpoint)
    )


def load_reference(path: str | Path, cases: list[BenchmarkCase]) -> dict[tuple[str, str], dict[str, Any]]:
    """Read serialized predictions and require exactly one compatible row per benchmark branch."""
    branch_index = {(case.id, branch.id): branch for case in cases for branch in case.branches}
    reference: dict[tuple[str, str], dict[str, Any]] = {}
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        key = (raw.get("case_id"), raw.get("branch_id"))
        if key not in branch_index:
            raise ValueError(f"{path}:{line_no}: unknown case_id/branch_id {key!r}")
        if key in reference:
            raise ValueError(f"{path}:{line_no}: duplicate case_id/branch_id {key!r}")
        reference[key] = _normalise_result(branch_index[key], raw, f"{path}:{line_no}")
    if reference.keys() != branch_index.keys():
        missing = sorted(branch_index.keys() - reference.keys())
        extra = sorted(reference.keys() - branch_index.keys())
        raise ValueError(f"{path}: reference must cover the benchmark exactly; missing={missing}, extra={extra}")
    return reference


def score_reference_agreement(
    cases: list[BenchmarkCase], outputs: dict[tuple[str, str], dict[str, Any]], reference: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        for branch in case.branches:
            local = outputs[(case.id, branch.id)]
            teacher = reference[(case.id, branch.id)]
            if branch.kind == "noul":
                values["noul"]["mean_absolute_difference"].append(abs(local["p_true"] - teacher["p_true"]))
                continue
            local_probabilities = local["probabilities"]
            teacher_probabilities = teacher["probabilities"]
            values[branch.kind]["top1_agreement"].append(
                float(_argmax(local_probabilities) == _argmax(teacher_probabilities))
            )
            values[branch.kind]["js_divergence"].append(_js_divergence(local_probabilities, teacher_probabilities))
            if branch.kind == "score":
                local_score = sum((index + 1) * value for index, value in enumerate(local_probabilities))
                teacher_score = sum((index + 1) * value for index, value in enumerate(teacher_probabilities))
                values["score"]["expected_score_mae_vs_reference"].append(abs(local_score - teacher_score))
    return {
        kind: {"n": len(next(iter(metric_values.values()))), **{name: _mean(samples) for name, samples in metric_values.items()}}
        for kind, metric_values in sorted(values.items())
    }


def _reverse_choice_options(case: BenchmarkCase) -> PackedRequest:
    branches = [
        Branch(
            kind=branch.kind,
            instructions=branch.instructions,
            options=list(reversed(branch.options)) if branch.kind == "choice" else list(branch.options),
        )
        for branch in case.branches
    ]
    return PackedRequest(state=case.state, branches=branches)


_UNRELATED_BRANCH = Branch(kind="noul", instructions="Does the message mention a solar eclipse?", options=[])


def _reordered_request(case: BenchmarkCase, rng: random.Random) -> tuple[PackedRequest, list[BenchmarkBranch]]:
    """Shuffle branch order (state and each branch's own content are untouched)."""
    order = list(case.branches)
    rng.shuffle(order)
    return PackedRequest(state=case.state, branches=[branch.request_branch() for branch in order]), order


def _request_with_unrelated_branch(case: BenchmarkCase) -> PackedRequest:
    """Append one fixed, semantically unrelated branch after all of the case's real branches."""
    branches = [branch.request_branch() for branch in case.branches] + [_UNRELATED_BRANCH]
    return PackedRequest(state=case.state, branches=branches)


def _request_without_last_branch(case: BenchmarkCase) -> tuple[PackedRequest, tuple[BenchmarkBranch, ...]]:
    """Drop the case's last branch; the remaining branches keep their relative order."""
    kept = case.branches[:-1]
    return PackedRequest(state=case.state, branches=[branch.request_branch() for branch in kept]), kept


def _compare_results(kind: str, baseline: dict[str, Any], other: dict[str, Any]) -> tuple[float | None, float]:
    """`(top1_match, drift)`: top1_match is `None` for noul (no top-1 concept); drift is mean L1 or |Δp_true|."""
    if kind == "noul":
        return None, abs(baseline["p_true"] - other["p_true"])
    base_probs = baseline["probabilities"]
    other_probs = other["probabilities"]
    top1 = float(_argmax(base_probs) == _argmax(other_probs))
    drift = sum(abs(a - b) for a, b in zip(base_probs, other_probs)) / len(base_probs)
    return top1, drift


def score_robustness(
    model: JevModel, cases: list[BenchmarkCase], baseline: dict[tuple[str, str], dict[str, Any]], seed: int = 0
) -> dict[str, Any]:
    """Measure semantics-preserving option, packing, and branch-set transformations.

    Score options are intentionally never permuted: their physical position encodes
    ordinal meaning. Every check compares against `baseline` (the model's own output
    on the original, unmodified request) rather than the human gold target — this
    isolates the model's self-consistency from its accuracy.
    """
    rng = random.Random(seed)
    reversed_top1: list[float] = []
    reversed_l1: list[float] = []
    packing_top1: list[float] = []
    packing_l1: list[float] = []
    packing_abs: list[float] = []
    reorder_top1: list[float] = []
    reorder_drift: list[float] = []
    add_unrelated_top1: list[float] = []
    add_unrelated_drift: list[float] = []
    remove_sibling_top1: list[float] = []
    remove_sibling_drift: list[float] = []

    for case in cases:
        reversed_results = model.forward(_reverse_choice_options(case))
        for branch, reversed_result in zip(case.branches, reversed_results):
            if branch.kind != "choice":
                continue
            original = baseline[(case.id, branch.id)]["probabilities"]
            transformed = list(reversed(_normalise_result(branch, reversed_result, f"{case.id}/{branch.id} reverse")["probabilities"]))
            reversed_top1.append(float(_argmax(original) == _argmax(transformed)))
            reversed_l1.append(sum(abs(a - b) for a, b in zip(original, transformed)) / len(original))

        for branch in case.branches:
            single_result = model.forward(PackedRequest(state=case.state, branches=[branch.request_branch()]))[0]
            baseline_result = baseline[(case.id, branch.id)]
            normalised = _normalise_result(branch, single_result, f"{case.id}/{branch.id} single")
            if branch.kind == "noul":
                packing_abs.append(abs(baseline_result["p_true"] - normalised["p_true"]))
            else:
                base_probs = baseline_result["probabilities"]
                single_probs = normalised["probabilities"]
                packing_top1.append(float(_argmax(base_probs) == _argmax(single_probs)))
                packing_l1.append(sum(abs(a - b) for a, b in zip(base_probs, single_probs)) / len(base_probs))

        if len(case.branches) >= 2:
            reordered_request, order = _reordered_request(case, rng)
            reordered_results = model.forward(reordered_request)
            for branch, result in zip(order, reordered_results):
                normalised = _normalise_result(branch, result, f"{case.id}/{branch.id} reorder")
                top1, drift = _compare_results(branch.kind, baseline[(case.id, branch.id)], normalised)
                if top1 is not None:
                    reorder_top1.append(top1)
                reorder_drift.append(drift)

        add_unrelated_results = model.forward(_request_with_unrelated_branch(case))
        for branch, result in zip(case.branches, add_unrelated_results[: len(case.branches)]):
            normalised = _normalise_result(branch, result, f"{case.id}/{branch.id} add-unrelated")
            top1, drift = _compare_results(branch.kind, baseline[(case.id, branch.id)], normalised)
            if top1 is not None:
                add_unrelated_top1.append(top1)
            add_unrelated_drift.append(drift)

        if len(case.branches) >= 2:
            reduced_request, kept = _request_without_last_branch(case)
            reduced_results = model.forward(reduced_request)
            for branch, result in zip(kept, reduced_results):
                normalised = _normalise_result(branch, result, f"{case.id}/{branch.id} remove-sibling")
                top1, drift = _compare_results(branch.kind, baseline[(case.id, branch.id)], normalised)
                if top1 is not None:
                    remove_sibling_top1.append(top1)
                remove_sibling_drift.append(drift)

    return {
        "choice_option_reversal": {"n": len(reversed_top1), "top1_consistency": _mean(reversed_top1), "mean_l1_drift": _mean(reversed_l1)},
        "single_branch_packing": {
            "distribution_n": len(packing_top1),
            "top1_consistency": _mean(packing_top1),
            "mean_l1_drift": _mean(packing_l1),
            "noul_n": len(packing_abs),
            "noul_mean_absolute_drift": _mean(packing_abs),
        },
        "branch_reorder": {
            "distribution_n": len(reorder_top1),
            "top1_consistency": _mean(reorder_top1),
            "mean_drift": _mean(reorder_drift),
        },
        "branch_add_unrelated": {
            "distribution_n": len(add_unrelated_top1),
            "top1_consistency": _mean(add_unrelated_top1),
            "mean_drift": _mean(add_unrelated_drift),
        },
        "branch_remove_sibling": {
            "distribution_n": len(remove_sibling_top1),
            "top1_consistency": _mean(remove_sibling_top1),
            "mean_drift": _mean(remove_sibling_drift),
        },
    }


def measure_latency(model: JevModel, cases: list[BenchmarkCase], warmups: int, repeats: int) -> dict[str, Any]:
    """Measure warm end-to-end model.forward latency; packer and result formatting are included."""
    if repeats < 1:
        raise ValueError("latency repeats must be at least 1")
    request = cases[0].request()
    for _ in range(warmups):
        model.forward(request)

    samples_ms: list[float] = []
    for case in cases:
        request = case.request()
        for _ in range(repeats):
            start = time.perf_counter()
            model.forward(request)
            samples_ms.append((time.perf_counter() - start) * 1_000)
    ordered = sorted(samples_ms)
    return {
        "measurement": "warm end-to-end model.forward (packing, backbone, readout, result formatting)",
        "warmups": warmups,
        "repeats_per_request": repeats,
        "samples": len(samples_ms),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99_ms": ordered[math.ceil(0.99 * len(ordered)) - 1],
    }


_LATENCY_STATE = (
    "A customer writes in about a billing discrepancy on their latest invoice, mentions they "
    "already tried the self-service portal without success, and asks for a prompt resolution."
)
_LATENCY_BRANCH = Branch(
    kind="choice",
    instructions="Which team should handle this?",
    options=["billing", "technical", "sales"],
)


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99_ms": ordered[math.ceil(0.99 * len(ordered)) - 1],
    }


def measure_latency_scaling(
    model: JevModel, warmups: int, repeats: int, branch_counts: tuple[int, ...] = (1, 2, 4, 8, 16)
) -> dict[str, Any]:
    """Controlled packed-vs-naive-re-encode latency at fixed branch counts.

    Every branch count uses the same synthetic state/branch content (only the branch
    count varies), so the comparison isolates the effect of packing from confounds
    like state length or branch content. ``packed``: one `model.forward` call holding
    all N branches. ``naive_re_encode``: N separate single-branch `model.forward`
    calls summed per repeat — the baseline this project's design is meant to beat.
    """
    warmup_request = PackedRequest(state=_LATENCY_STATE, branches=[_LATENCY_BRANCH])
    for _ in range(warmups):
        model.forward(warmup_request)

    results: dict[str, Any] = {}
    for n in branch_counts:
        packed_request = PackedRequest(state=_LATENCY_STATE, branches=[_LATENCY_BRANCH] * n)
        packed_samples: list[float] = []
        naive_samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.forward(packed_request)
            packed_samples.append((time.perf_counter() - start) * 1_000)

            start = time.perf_counter()
            for _ in range(n):
                model.forward(PackedRequest(state=_LATENCY_STATE, branches=[_LATENCY_BRANCH]))
            naive_samples.append((time.perf_counter() - start) * 1_000)

        packed_p50 = _percentiles(packed_samples)["p50_ms"]
        naive_p50 = _percentiles(naive_samples)["p50_ms"]
        results[str(n)] = {
            "branches": n,
            "packed": {"samples": len(packed_samples), **_percentiles(packed_samples)},
            "naive_re_encode": {"samples": len(naive_samples), **_percentiles(naive_samples)},
            "naive_over_packed_ratio_p50": naive_p50 / packed_p50 if packed_p50 > 0 else float("nan"),
        }
    return {
        "measurement": "controlled synthetic state/branch content, varying only branch count",
        "warmups": warmups,
        "repeats": repeats,
        "branch_counts": results,
    }


def _serialise_outputs(cases: list[BenchmarkCase], outputs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for branch in case.branches:
            rows.append({"case_id": case.id, "branch_id": branch.id, **outputs[(case.id, branch.id)]})
    return rows


def _self_test(cases: list[BenchmarkCase]) -> None:
    perfect = {(case.id, branch.id): target_output(branch) for case in cases for branch in case.branches}
    quality = score_quality(cases, perfect)
    for kind, metrics in quality.items():
        assert metrics["brier"] == 0.0, f"{kind}: perfect target output must have zero Brier score"
        assert math.isfinite(metrics["nll"]) and metrics["nll"] >= 0.0, (
            f"{kind}: target entropy (NLL of a perfect soft-label prediction) must be finite"
        )
    agreement = score_reference_agreement(cases, perfect, perfect)
    for kind, metrics in agreement.items():
        if kind == "noul":
            assert metrics["mean_absolute_difference"] == 0.0
        else:
            assert metrics["top1_agreement"] == 1.0
            assert metrics["js_divergence"] == 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, help="PointerReadout checkpoint to evaluate")
    parser.add_argument("--noul-checkpoint", type=Path, help="NoulReadout checkpoint (separate from --checkpoint; NoulReadout is trained independently of PointerReadout)")
    parser.add_argument("--reference", type=Path, help="Serialized comparison model output JSONL")
    parser.add_argument("--write-predictions", type=Path, help="Write this model's output in reference JSONL format")
    parser.add_argument("--validate", action="store_true", help="Validate schema and metric invariants without loading a model")
    parser.add_argument("--robustness", action="store_true", help="Run option-order and branch-packing metamorphic checks")
    parser.add_argument("--measure-latency", action="store_true", help="Measure warm end-to-end latency")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--null-calibration", action="store_true", help="Subtract content-free (null state) option bias from choice/score outputs")
    args = parser.parse_args()
    if args.null_calibration and (args.robustness or args.measure_latency):
        parser.error("--null-calibration applies to quality outputs only; robustness/latency would measure the uncalibrated model")

    cases = load_cases(args.dataset)
    statuses = Counter(case.label_status for case in cases)
    branch_count = sum(len(case.branches) for case in cases)
    print(json.dumps({"dataset": str(args.dataset), "cases": len(cases), "branches": branch_count, "label_status": statuses}, ensure_ascii=False))
    _self_test(cases)
    if args.validate:
        print("OK: schema validation and perfect-prediction metric invariants passed")
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --validate is used")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = JevModel(option_pooling=checkpoint.get("option_pooling", "last"))
    model.pointer.load_state_dict(checkpoint["state_dict"])
    if args.noul_checkpoint is not None:
        noul_checkpoint = torch.load(args.noul_checkpoint, map_location="cpu", weights_only=False)
        model.noul.load_state_dict(noul_checkpoint["noul_state_dict"])
    else:
        print("WARNING: no --noul-checkpoint given; NoulReadout stays randomly initialized (noul quality metrics are meaningless).")
    outputs = run_model(model, cases, null_calibration=args.null_calibration)

    summary: dict[str, Any] = {
        "dataset": str(args.dataset),
        "null_calibration": args.null_calibration,
        "label_status": dict(statuses),
        "quality": score_quality(cases, outputs),
    }
    if args.reference:
        summary["reference_agreement"] = score_reference_agreement(cases, outputs, load_reference(args.reference, cases))
    if args.robustness:
        summary["robustness"] = score_robustness(model, cases, outputs)
    if args.measure_latency:
        summary["latency"] = measure_latency(model, cases, args.warmups, args.latency_repeats)
    if args.write_predictions:
        args.write_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.write_predictions.open("w") as output:
            for row in _serialise_outputs(cases, outputs):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if set(statuses) != {"adjudicated"}:
        print("WARNING: this dataset includes non-adjudicated labels; use it for regression/dev only, not a Jev-equivalence claim.")


if __name__ == "__main__":
    main()
