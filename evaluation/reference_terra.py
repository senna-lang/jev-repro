"""Generate typed reference predictions with an OpenAI-compatible Terra model.

The generated JSONL is reference agreement evidence, never a Gold label source.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.schema import BenchmarkCase, load_cases

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def prompt_for(case: BenchmarkCase) -> str:
    branches = []
    for branch in case.branches:
        branches.append({"branch_id": branch.id, "kind": branch.kind, "instructions": branch.instructions, "options": list(branch.options)})
    return """You are a calibrated structured-decision reference model. Use only the supplied state, instructions, and options. Return JSON only: `{"answers":[{"branch_id":"exact input ID","probabilities":[...]}]}` for choice/score, or `{"answers":[{"branch_id":"exact input ID","p_true":0.0}]}` for noul. Return exactly one object per branch in `answers`.\n\nFor choice and score, return a `probabilities` array aligned to the supplied option order. Values must be numbers in [0,1] and sum to exactly 1. Score option order is ordinal and MUST NOT be reordered. For noul, return `p_true`, a number in [0,1]. Do not return explanations, labels, confidence fields, `kind`, or extra branches.\n\nINPUT:\n""" + json.dumps({"case_id": case.id, "state": case.state, "branches": branches}, ensure_ascii=False)


def _post(endpoint: str, api_key: str, model: str, prompt: str) -> dict[str, Any]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}).encode()
    request = urllib.request.Request(endpoint, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = json.loads(response.read())
    return json.loads(raw["choices"][0]["message"]["content"])


def normalise(case: BenchmarkCase, response: dict[str, Any]) -> list[dict[str, Any]]:
    answers = response.get("answers")
    if not isinstance(answers, list):
        raise ValueError(f"{case.id}: response has no answers array")
    expected = {branch.id: branch for branch in case.branches}
    by_id = {answer.get("branch_id"): answer for answer in answers if isinstance(answer, dict)}
    if set(by_id) != set(expected):
        raise ValueError(f"{case.id}: expected branch IDs {sorted(expected)}, got {sorted(by_id)}")
    rows = []
    for branch_id, branch in expected.items():
        answer = by_id[branch_id]
        if branch.kind == "noul":
            p = answer.get("p_true")
            if not isinstance(p, (int, float)) or not 0 <= p <= 1:
                raise ValueError(f"{case.id}/{branch_id}: invalid p_true")
            rows.append({"case_id": case.id, "branch_id": branch_id, "kind": "noul", "p_true": float(p)})
        else:
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, list) or len(probabilities) != len(branch.options) or any(not isinstance(x, (int, float)) or x < 0 or x > 1 for x in probabilities) or abs(sum(probabilities) - 1) > 1e-6:
                raise ValueError(f"{case.id}/{branch_id}: invalid probability distribution")
            rows.append({"case_id": case.id, "branch_id": branch_id, "kind": branch.kind, "probabilities": [float(x) for x in probabilities]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.dry_run:
        print(prompt_for(cases[0])); return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY is required unless --dry-run is used")
    rows = []
    for case in cases:
        try:
            rows.extend(normalise(case, _post(args.endpoint, api_key, args.model, prompt_for(case))))
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as error:
            raise SystemExit(f"reference generation failed at {case.id}: {error}") from error
        if args.delay_seconds: time.sleep(args.delay_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"role":"reference-only; never Gold evidence","provider":"openai-compatible","model":args.model,"endpoint":args.endpoint,"dataset":str(args.dataset),"cases":len(cases),"branches":len(rows),"delay_seconds":args.delay_seconds}, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"branches":len(rows),"model":args.model},ensure_ascii=False))

if __name__ == "__main__": main()
