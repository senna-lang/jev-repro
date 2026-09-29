"""Run the full local GOAL.md evaluation pipeline for one checkpoint end-to-end.

Usage:
    python3 -m evaluation.evaluate_checkpoint \\
      --checkpoint checkpoints/<checkpoint>.pt \\
      --output-dir evaluation/private/runs/<checkpoint-name>

    python3 -m evaluation.evaluate_checkpoint \\
      --checkpoint checkpoints/<checkpoint>.pt \\
      --output-dir evaluation/private/runs/<checkpoint-name> \\
      --reference evaluation/private/<reference>.jsonl

Steps, in order (`GOAL.md` §2-4):
  1. Gold V1 prediction JSONL generation (one forward pass per case).
  2. Gold slice / branch diagnostics with bootstrap CIs (`evaluation.analyze`).
  3. Gold V1 full robustness: single-branch-packing, branch-reorder,
     branch-add-unrelated, branch-remove-sibling, choice-option-reversal.
  4. Controlled packed-vs-naive-re-encode latency (branch counts 1/2/4/8/16,
     fixed synthetic state/branch content so only branch count varies).
  5. If `--reference` is given: reference-model agreement (never treated as
     ground truth; quality in step 1-2 is always scored against the Gold target).

Every artifact is written under `--output-dir`; `summary.json` aggregates all
steps plus a manifest (checksums, checkpoint metadata, timestamps) for
reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from evaluation.analyze import analyze
from evaluation.run import (
    _self_test,
    load_reference,
    measure_latency_scaling,
    run_model,
    score_quality,
    score_reference_agreement,
    score_robustness,
)
from evaluation.schema import load_cases
from legacy.model_readout.forward import JevModel

# 主評価は英語版（GOAL.md §2）。case/branch IDは原本と同一なのでauditはそのまま使える。
DEFAULT_DATASET = Path(__file__).with_name("private") / "gold_v1_adjudicated_en_mt.jsonl"
DEFAULT_AUDIT = Path(__file__).with_name("private") / "gold_v1_adjudication_audit.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="PointerReadout checkpoint to evaluate")
    parser.add_argument("--noul-checkpoint", type=Path, help="NoulReadout checkpoint (trained separately; omit to leave NoulReadout randomly initialized)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for all run artifacts")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Gold-format JSONL (default: Gold V1 English)")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help="Adjudication audit JSON for slice metrics")
    parser.add_argument("--reference", type=Path, help="Serialized comparison model output JSONL")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--latency-repeats", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.dataset)
    _self_test(cases)
    print(json.dumps(
        {"step": "load", "dataset": str(args.dataset), "cases": len(cases), "branches": sum(len(c.branches) for c in cases)},
        ensure_ascii=False,
    ))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    option_pooling = checkpoint.get("option_pooling", "last")
    model = JevModel(option_pooling=option_pooling)
    model.pointer.load_state_dict(checkpoint["state_dict"])
    print(json.dumps({"step": "checkpoint", "path": str(args.checkpoint), "option_pooling": option_pooling}, ensure_ascii=False))

    if args.noul_checkpoint is not None:
        noul_checkpoint = torch.load(args.noul_checkpoint, map_location="cpu", weights_only=False)
        model.noul.load_state_dict(noul_checkpoint["noul_state_dict"])
        print(json.dumps({"step": "noul_checkpoint", "path": str(args.noul_checkpoint)}, ensure_ascii=False))
    else:
        print(json.dumps({"step": "noul_checkpoint", "warning": "no --noul-checkpoint given; NoulReadout stays randomly initialized (noul quality metrics are meaningless)"}, ensure_ascii=False))

    # 1. Gold V1 predictions.
    outputs = run_model(model, cases)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w") as handle:
        for case in cases:
            for branch in case.branches:
                row = {"case_id": case.id, "branch_id": branch.id, **outputs[(case.id, branch.id)]}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"step": "predictions", "path": str(predictions_path), "rows": sum(len(c.branches) for c in cases)}, ensure_ascii=False))

    # 2. Slice / branch diagnostics with bootstrap CIs.
    audit_path = args.audit if args.audit.exists() else None
    if args.audit and not audit_path:
        print(json.dumps({"step": "analysis", "warning": f"audit file not found, skipping slices: {args.audit}"}, ensure_ascii=False))
    analysis_report = analyze(args.dataset, predictions_path, audit_path, args.bootstrap_repeats, args.seed)
    _write_json(output_dir / "analysis.json", analysis_report)
    print(json.dumps({"step": "analysis", "path": str(output_dir / "analysis.json"), "slices": sorted(analysis_report["slices"])}, ensure_ascii=False))

    # 3. Full robustness suite.
    robustness_report = score_robustness(model, cases, outputs, seed=args.seed)
    _write_json(output_dir / "robustness.json", robustness_report)
    print(json.dumps({"step": "robustness", "path": str(output_dir / "robustness.json"), "checks": sorted(robustness_report)}, ensure_ascii=False))

    # 4. Controlled packed-vs-naive-re-encode latency.
    latency_report = measure_latency_scaling(model, args.warmups, args.latency_repeats)
    _write_json(output_dir / "latency.json", latency_report)
    print(json.dumps({"step": "latency", "path": str(output_dir / "latency.json"), "branch_counts": sorted(latency_report["branch_counts"], key=int)}, ensure_ascii=False))

    # 5. Optional reference-model agreement.
    reference_report = None
    if args.reference:
        reference_report = score_reference_agreement(cases, outputs, load_reference(args.reference, cases))
        _write_json(output_dir / "reference_agreement.json", reference_report)
        print(json.dumps({"step": "reference_agreement", "path": str(output_dir / "reference_agreement.json")}, ensure_ascii=False))

    quality_report = score_quality(cases, outputs)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_option_pooling": option_pooling,
        "checkpoint_datasets": checkpoint.get("datasets"),
        "noul_checkpoint": str(args.noul_checkpoint) if args.noul_checkpoint else None,
        "noul_checkpoint_sha256": _sha256(args.noul_checkpoint) if args.noul_checkpoint else None,
        "noul_checkpoint_held_out_accuracy": (
            torch.load(args.noul_checkpoint, map_location="cpu", weights_only=False).get("held_out_accuracy")
            if args.noul_checkpoint
            else None
        ),
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "audit": str(audit_path) if audit_path else None,
        "reference": str(args.reference) if args.reference else None,
        "python": sys.version,
        "platform": platform.platform(),
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
        "warmups": args.warmups,
        "latency_repeats": args.latency_repeats,
        "started_at": started,
        "elapsed_seconds": time.time() - started,
    }
    summary = {
        "manifest": manifest,
        "quality": quality_report,
        "robustness": robustness_report,
        "latency": latency_report,
        "reference_agreement": reference_report,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"step": "done", "output_dir": str(output_dir), "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False))
    if any(status != "adjudicated" for case in cases for status in [case.label_status]):
        print("WARNING: this dataset includes non-adjudicated labels; use it for regression/dev only, not a Jev-equivalence claim.")


if __name__ == "__main__":
    main()
