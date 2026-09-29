"""Validate one raw response exported by ``pilot_v1.html``.

This accepts individual, unlabeled human responses only. It deliberately does not
aggregate votes or assign a gold target; aggregation happens after all independent
responses arrive and disagreement has been adjudicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FORM_VERSION = "pilot_v1"
INSTRUMENT = "gold-annotation-pilot"

# (case id, branch id) -> (kind, number of selectable answers). These contain no intended labels.
EXPECTED: dict[tuple[str, str], tuple[str, int]] = {
    ("pilot-01", "department"): ("choice", 3),
    ("pilot-01", "is_urgent"): ("noul", 2),
    ("pilot-02", "department"): ("choice", 3),
    ("pilot-02", "payment_problem"): ("noul", 2),
    ("pilot-03", "shipping_issue"): ("choice", 4),
    ("pilot-04", "return_reason"): ("choice", 4),
    ("pilot-04", "requested_resolution"): ("choice", 4),
    ("pilot-05", "severity"): ("score", 3),
    ("pilot-05", "human_escalation"): ("noul", 2),
    ("pilot-06", "severity"): ("score", 3),
    ("pilot-07", "severity"): ("score", 3),
    ("pilot-07", "report_quality"): ("score", 4),
    ("pilot-08", "tone"): ("choice", 3),
    ("pilot-08", "frustration"): ("score", 3),
    ("pilot-09", "tone"): ("choice", 3),
    ("pilot-09", "is_urgent"): ("noul", 2),
    ("pilot-10", "human_escalation"): ("noul", 2),
    ("pilot-10", "repeat_contact"): ("noul", 2),
    ("pilot-11", "shipping_issue"): ("choice", 4),
    ("pilot-11", "severity"): ("score", 3),
    ("pilot-12", "return_topic"): ("choice", 3),
    ("pilot-13", "is_urgent"): ("noul", 2),
    ("pilot-13", "severity"): ("score", 3),
    ("pilot-14", "department"): ("choice", 3),
    ("pilot-14", "report_quality"): ("score", 4),
    ("pilot-15", "department"): ("choice", 3),
    ("pilot-15", "tone"): ("choice", 3),
}


def validate_response(raw: Any) -> str:
    """Validate a response object and return its evaluator identifier."""
    if not isinstance(raw, dict):
        raise ValueError("response must be a JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("expected schema_version=1")
    if raw.get("instrument") != INSTRUMENT:
        raise ValueError(f"expected instrument={INSTRUMENT!r}")
    if raw.get("form_version") != FORM_VERSION:
        raise ValueError(f"expected form_version={FORM_VERSION!r}")

    annotator_id = raw.get("annotator_id")
    if not isinstance(annotator_id, str) or not annotator_id.strip():
        raise ValueError("annotator_id must be a non-empty string")
    if not isinstance(raw.get("submitted_at"), str) or not raw["submitted_at"]:
        raise ValueError("submitted_at must be a non-empty ISO-8601 string")

    answers = raw.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answers must be a list")
    seen: set[tuple[str, str]] = set()
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("every answer must be an object")
        key = (answer.get("case_id"), answer.get("branch_id"))
        if key not in EXPECTED:
            raise ValueError(f"unexpected case/branch {key!r}")
        if key in seen:
            raise ValueError(f"duplicate answer for {key!r}")
        seen.add(key)

        kind, option_count = EXPECTED[key]
        if answer.get("kind") != kind:
            raise ValueError(f"{key!r}: expected kind={kind!r}")
        selection = answer.get("selection")
        if kind == "noul":
            if type(selection) is not bool:
                raise ValueError(f"{key!r}: noul selection must be true or false")
        elif type(selection) is not int or not 0 <= selection < option_count:
            raise ValueError(f"{key!r}: selection must be an option index from 0 to {option_count - 1}")
        if type(answer.get("flagged")) is not bool:
            raise ValueError(f"{key!r}: flagged must be boolean")
        if not isinstance(answer.get("note"), str):
            raise ValueError(f"{key!r}: note must be a string")

    if seen != EXPECTED.keys():
        missing = sorted(EXPECTED.keys() - seen)
        extra = sorted(seen - EXPECTED.keys())
        raise ValueError(f"response must contain all {len(EXPECTED)} answers; missing={missing}, extra={extra}")
    return annotator_id.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path, help="JSON downloaded from pilot_v1.html")
    args = parser.parse_args()
    try:
        raw = json.loads(args.response.read_text())
        annotator_id = validate_response(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"INVALID: {exc}") from exc
    print(f"OK: {args.response} — annotator={annotator_id}, answers={len(EXPECTED)}, form={FORM_VERSION}")


if __name__ == "__main__":
    main()
