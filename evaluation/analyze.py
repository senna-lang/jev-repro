"""Analyze JSONL predictions against a versioned Jev benchmark without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.run import _argmax, _distribution_metrics, _js_divergence, load_reference
from evaluation.schema import load_cases


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _bootstrap(values: list[float], repeats: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(repeats))
    return {"mean": _mean(values), "ci95_low": samples[int(.025 * (repeats - 1))], "ci95_high": samples[int(.975 * (repeats - 1))]}


def _audit_index(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    return {(row["case_id"], row["branch_id"]): row for row in json.loads(path.read_text())["branches"]}


def analyze(dataset: Path, predictions: Path, audit: Path | None, bootstrap_repeats: int, seed: int) -> dict[str, Any]:
    cases = load_cases(dataset)
    output = load_reference(predictions, cases)
    audit_rows = _audit_index(audit)
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows: list[dict[str, Any]] = []
    for case in cases:
        for branch in case.branches:
            key = (case.id, branch.id)
            pred = output[key]
            audit_row = audit_rows.get(key, {})
            slices = ["all", f"kind:{branch.kind}", *(f"tag:{tag}" for tag in case.tags)]
            if audit_row:
                slices += ["consensus:" + ("3/3" if len(set(audit_row["votes"])) == 1 else "2/1"), "flagged" if any(audit_row["flags"]) else "unflagged"]
                if audit_row.get("replacement_revision"):
                    slices.append("revision")
            row: dict[str, Any] = {"case_id": case.id, "branch_id": branch.id, "kind": branch.kind, "prediction": pred, "slices": slices}
            if branch.kind == "noul":
                target = branch.target_p_true
                p = pred["p_true"]
                metrics = {"brier": (p-target)**2, "nll": -(target*math.log(max(p,1e-12))+(1-target)*math.log(max(1-p,1e-12))), "threshold_accuracy": float((p >= .5) == (target >= .5))}
                row["gold_target"] = target
            else:
                target = list(branch.target_probabilities)
                metrics = _distribution_metrics(pred["probabilities"], branch.target_probabilities)
                metrics["top1_accuracy"] = float(_argmax(pred["probabilities"]) == _argmax(target))
                row["gold_target"] = target
                if branch.kind == "score":
                    expected = lambda ps: sum((i+1)*x for i,x in enumerate(ps))
                    metrics["expected_score_mae"] = abs(expected(pred["probabilities"])-expected(target))
            row["metrics"] = metrics
            rows.append(row)
            for slice_name in slices:
                for name, value in metrics.items(): groups[slice_name][name].append(value)
    return {"dataset": str(dataset), "predictions": str(predictions), "branches": rows, "slices": {s: {m: _bootstrap(v, bootstrap_repeats, seed) for m,v in metrics.items()} | {"n": len(next(iter(metrics.values())))} for s,metrics in sorted(groups.items())}}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = analyze(args.dataset, args.predictions, args.audit, args.bootstrap_repeats, args.seed)
    report["manifest"] = {"dataset_sha256": _sha256(args.dataset), "predictions_sha256": _sha256(args.predictions), "audit_sha256": _sha256(args.audit) if args.audit else None, "python": sys.version, "platform": platform.platform(), "bootstrap_repeats": args.bootstrap_repeats, "seed": args.seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "branches": len(report["branches"]), "slices": len(report["slices"])}, ensure_ascii=False))

if __name__ == "__main__":
    main()
