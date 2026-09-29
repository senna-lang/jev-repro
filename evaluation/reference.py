"""Validate, compare, and aggregate multiple typed-decision reference JSONL files."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.run import _js_divergence, load_reference
from evaluation.schema import load_cases


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(dataset: Path, inputs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(inputs) < 2:
        raise ValueError("at least two independent reference prediction files are required")
    cases = load_cases(dataset)
    references = [load_reference(path, cases) for path in inputs]
    rows: list[dict[str, Any]] = []
    disagreement: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for branch in case.branches:
            key = (case.id, branch.id)
            values = [reference[key] for reference in references]
            if branch.kind == "noul":
                probabilities = [value["p_true"] for value in values]
                mean = _mean(probabilities)
                pairwise = [abs(a - b) for i, a in enumerate(probabilities) for b in probabilities[i + 1:]]
                rows.append({"case_id": case.id, "branch_id": branch.id, "kind": "noul", "p_true": mean})
                disagreement["noul_mean_absolute_difference"].extend(pairwise)
            else:
                distributions = [value["probabilities"] for value in values]
                mean = [_mean([distribution[i] for distribution in distributions]) for i in range(len(branch.options))]
                pairwise = [_js_divergence(a, b) for i, a in enumerate(distributions) for b in distributions[i + 1:]]
                rows.append({"case_id": case.id, "branch_id": branch.id, "kind": branch.kind, "probabilities": mean})
                disagreement[f"{branch.kind}_js_divergence"].extend(pairwise)
    return rows, {"sources": [str(path) for path in inputs], "pairwise_disagreement": {name: _mean(values) for name, values in sorted(disagreement.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True, help="Repeat for each independently generated reference JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = aggregate(args.dataset, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "branches": len(rows), **report}, ensure_ascii=False))

if __name__ == "__main__":
    main()
