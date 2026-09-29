"""Gold-wide metamorphic checks for packed Jev requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluation.run import _argmax, _normalise_result, run_model
from evaluation.schema import BenchmarkBranch, BenchmarkCase, load_cases
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest


def _l1(a: list[float], b: list[float]) -> float:
    return sum(abs(x-y) for x,y in zip(a,b)) / len(a)


def _output_drift(branch: BenchmarkBranch, before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, float]:
    if branch.kind == "noul":
        return True, abs(before["p_true"] - after["p_true"])
    return _argmax(before["probabilities"]) == _argmax(after["probabilities"]), _l1(before["probabilities"], after["probabilities"])


def evaluate(model: JevModel, cases: list[BenchmarkCase]) -> dict[str, Any]:
    baseline = run_model(model, cases)
    rows: list[dict[str, Any]] = []
    for case in cases:
        reversed_request = PackedRequest(case.state, [Branch(b.kind, b.instructions, list(reversed(b.options)) if b.kind == "choice" else list(b.options)) for b in case.branches])
        reverse = model.forward(reversed_request)
        reorder_request = PackedRequest(case.state, [b.request_branch() for b in reversed(case.branches)])
        reorder = list(reversed(model.forward(reorder_request)))
        added_request = PackedRequest(
            case.state,
            [
                *[branch.request_branch() for branch in case.branches],
                Branch("noul", "Is this an unrelated synthetic control question?"),
            ],
        )
        added = model.forward(added_request)
        for index, branch in enumerate(case.branches):
            key = (case.id, branch.id)
            original = baseline[key]
            single = _normalise_result(branch, model.forward(PackedRequest(case.state, [branch.request_branch()]))[0], f"{case.id}/{branch.id}/single")
            ordered = _normalise_result(branch, reorder[index], f"{case.id}/{branch.id}/reorder")
            appended = _normalise_result(branch, added[index], f"{case.id}/{branch.id}/add")
            variants = {"single": single, "reorder": ordered, "add_unrelated": appended}
            if len(case.branches) > 1:
                retained = [candidate.request_branch() for candidate_index, candidate in enumerate(case.branches) if candidate_index != index]
                removed = model.forward(PackedRequest(case.state, retained))
                for candidate_index, candidate in enumerate(case.branches):
                    if candidate_index == index:
                        continue
                    result_index = candidate_index if candidate_index < index else candidate_index - 1
                    base = baseline[(case.id, candidate.id)]
                    transformed = _normalise_result(candidate, removed[result_index], f"{case.id}/{candidate.id}/remove")
                    top1, drift = _output_drift(candidate, base, transformed)
                    rows.append({"case_id":case.id,"branch_id":candidate.id,"kind":candidate.kind,"transform":"remove_sibling","top1_consistent":top1,"mean_l1_or_abs_drift":drift})
            if branch.kind == "choice":
                raw = _normalise_result(branch, reverse[index], f"{case.id}/{branch.id}/reverse")
                variants["option_reverse"] = {**raw, "probabilities": list(reversed(raw["probabilities"]))}
            for name, transformed in variants.items():
                top1, drift = _output_drift(branch, original, transformed)
                rows.append({"case_id":case.id,"branch_id":branch.id,"kind":branch.kind,"transform":name,"top1_consistent":top1,"mean_l1_or_abs_drift":drift})
    summary: dict[str, dict[str, float]] = {}
    for transform in sorted({row["transform"] for row in rows}):
        group=[row for row in rows if row["transform"]==transform]
        summary[transform]={"n":len(group),"top1_consistency":sum(row["top1_consistent"] for row in group)/len(group),"mean_drift":sum(row["mean_l1_or_abs_drift"] for row in group)/len(group),"max_drift":max(row["mean_l1_or_abs_drift"] for row in group)}
    return {"branches":rows,"summary":summary}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=JevModel(); model.pointer.load_state_dict(checkpoint["state_dict"])
    report=evaluate(model,load_cases(args.dataset))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report["summary"],ensure_ascii=False))

if __name__ == "__main__": main()
