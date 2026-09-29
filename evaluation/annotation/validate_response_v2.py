"""Validate an exported response from ``pilot_v2.html``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED: dict[tuple[str, str], tuple[str, int]] = {
    ("v2-control-01", "human_escalation"): ("noul", 2),
    ("v2-control-02", "is_urgent"): ("noul", 2),
    ("v2-control-03", "report_quality"): ("score", 4),
    ("v2-control-04", "report_quality"): ("score", 4),
    ("v2-boundary-01", "tone"): ("choice", 3),
    ("v2-boundary-01", "frustration"): ("score", 3),
    ("v2-boundary-02", "severity"): ("score", 3),
    ("v2-boundary-03", "department"): ("choice", 3),
}


def validate_response(raw: Any) -> str:
    if not isinstance(raw, dict):
        raise ValueError("response must be a JSON object")
    if raw.get("schema_version") != 1 or raw.get("instrument") != "gold-annotation-pilot":
        raise ValueError("unexpected response schema or instrument")
    if raw.get("form_version") != "pilot_v2":
        raise ValueError("expected form_version='pilot_v2'")
    annotator_id = raw.get("annotator_id")
    if not isinstance(annotator_id, str) or not annotator_id.strip():
        raise ValueError("annotator_id must be a non-empty string")
    if not isinstance(raw.get("submitted_at"), str) or not raw["submitted_at"]:
        raise ValueError("submitted_at must be present")

    answers = raw.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answers must be a list")
    seen: set[tuple[str, str]] = set()
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("every answer must be an object")
        key = (answer.get("case_id"), answer.get("branch_id"))
        if key not in EXPECTED or key in seen:
            raise ValueError(f"unexpected or duplicate answer {key!r}")
        seen.add(key)
        kind, option_count = EXPECTED[key]
        selection = answer.get("selection")
        if answer.get("kind") != kind:
            raise ValueError(f"{key!r}: wrong branch kind")
        if kind == "noul":
            if type(selection) is not bool:
                raise ValueError(f"{key!r}: noul selection must be boolean")
        elif type(selection) is not int or not 0 <= selection < option_count:
            raise ValueError(f"{key!r}: selection outside option range")
        if type(answer.get("flagged")) is not bool or not isinstance(answer.get("note"), str):
            raise ValueError(f"{key!r}: invalid flag or note")
    if seen != EXPECTED.keys():
        raise ValueError(f"response must contain all {len(EXPECTED)} answers; missing={sorted(EXPECTED.keys() - seen)}")
    return annotator_id.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    try:
        evaluator = validate_response(json.loads(args.response.read_text()))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"INVALID: {exc}") from exc
    print(f"OK: {args.response} — annotator={evaluator}, answers={len(EXPECTED)}, form=pilot_v2")


if __name__ == "__main__":
    main()
