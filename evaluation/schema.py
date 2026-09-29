"""Versioned, model-agnostic schema for Jev-style quality benchmarks.

The public seed set is deliberately labelled ``provisional``: its labels are authored by
one maintainer and make a useful regression/dev set, not evidence of Jev equivalence.
A final benchmark uses the same JSONL schema but lives outside version control with
multi-annotator target distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from model.packer import Branch, PackedRequest

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkBranch:
    """A typed question and its target distribution.

    ``target_probabilities`` applies to choice/score branches and uses exactly the
    branch option order. ``target_p_true`` applies to noul branches. Target values
    may represent a human vote distribution rather than a one-hot label.
    """

    id: str
    kind: str
    instructions: str
    options: tuple[str, ...]
    target_probabilities: tuple[float, ...] | None
    target_p_true: float | None

    def request_branch(self) -> Branch:
        return Branch(kind=self.kind, instructions=self.instructions, options=list(self.options))


@dataclass(frozen=True)
class BenchmarkCase:
    """One state and all questions that must be evaluated in a single forward pass."""

    id: str
    state: str
    tags: tuple[str, ...]
    label_status: str
    branches: tuple[BenchmarkBranch, ...]

    def request(self) -> PackedRequest:
        return PackedRequest(state=self.state, branches=[branch.request_branch() for branch in self.branches])


def _require_str(value: Any, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{case_id}: {field} must be a non-empty string")
    return value


def _probabilities(value: Any, expected_len: int, case_id: str, branch_id: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected_len:
        raise ValueError(
            f"{case_id}/{branch_id}: target_probabilities must contain one value per option "
            f"({expected_len})"
        )
    probabilities = tuple(float(item) for item in value)
    if any(not 0.0 <= item <= 1.0 for item in probabilities):
        raise ValueError(f"{case_id}/{branch_id}: target_probabilities must be within [0, 1]")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError(f"{case_id}/{branch_id}: target_probabilities must sum to 1")
    return probabilities


def _parse_branch(raw: Any, case_id: str) -> BenchmarkBranch:
    if not isinstance(raw, dict):
        raise ValueError(f"{case_id}: each branch must be an object")

    branch_id = _require_str(raw.get("id"), "branch.id", case_id)
    kind = raw.get("kind")
    if kind not in {"choice", "score", "noul"}:
        raise ValueError(f"{case_id}/{branch_id}: unsupported kind {kind!r}")

    instructions = _require_str(raw.get("instructions"), "branch.instructions", case_id)
    raw_options = raw.get("options", [])
    if not isinstance(raw_options, list) or not all(isinstance(option, str) and option for option in raw_options):
        raise ValueError(f"{case_id}/{branch_id}: options must be a list of non-empty strings")
    options = tuple(raw_options)

    if kind == "noul":
        if options:
            raise ValueError(f"{case_id}/{branch_id}: noul must not contain options")
        p_true = raw.get("target_p_true")
        if not isinstance(p_true, (int, float)) or not 0.0 <= float(p_true) <= 1.0:
            raise ValueError(f"{case_id}/{branch_id}: noul target_p_true must be within [0, 1]")
        if "target_probabilities" in raw:
            raise ValueError(f"{case_id}/{branch_id}: noul must use target_p_true, not target_probabilities")
        return BenchmarkBranch(branch_id, kind, instructions, options, None, float(p_true))

    if len(options) < 2:
        raise ValueError(f"{case_id}/{branch_id}: {kind} needs at least two options")
    if "target_p_true" in raw:
        raise ValueError(f"{case_id}/{branch_id}: {kind} must use target_probabilities, not target_p_true")
    return BenchmarkBranch(
        branch_id,
        kind,
        instructions,
        options,
        _probabilities(raw.get("target_probabilities"), len(options), case_id, branch_id),
        None,
    )


def parse_case(raw: Any) -> BenchmarkCase:
    """Parse and validate one schema-v1 JSON benchmark record."""
    if not isinstance(raw, dict):
        raise ValueError("benchmark row must be an object")

    case_id = _require_str(raw.get("id"), "id", "<unknown>")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{case_id}: expected schema_version={SCHEMA_VERSION}")
    state = _require_str(raw.get("state"), "state", case_id)
    label_status = _require_str(raw.get("label_status"), "label_status", case_id)

    raw_tags = raw.get("tags", [])
    if not isinstance(raw_tags, list) or not all(isinstance(tag, str) and tag for tag in raw_tags):
        raise ValueError(f"{case_id}: tags must be a list of non-empty strings")

    raw_branches = raw.get("branches")
    if not isinstance(raw_branches, list) or not raw_branches:
        raise ValueError(f"{case_id}: branches must be a non-empty list")
    branches = tuple(_parse_branch(branch, case_id) for branch in raw_branches)
    branch_ids = [branch.id for branch in branches]
    if len(set(branch_ids)) != len(branch_ids):
        raise ValueError(f"{case_id}: branch ids must be unique within a request")

    return BenchmarkCase(case_id, state, tuple(raw_tags), label_status, branches)


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    """Load JSONL records, validate them, and reject duplicate case identifiers."""
    source = Path(path)
    cases: list[BenchmarkCase] = []
    for line_no, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_no}: invalid JSON: {exc.msg}") from exc
        try:
            cases.append(parse_case(raw))
        except ValueError as exc:
            raise ValueError(f"{source}:{line_no}: {exc}") from exc

    if not cases:
        raise ValueError(f"{source}: benchmark contains no cases")
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source}: case ids must be globally unique")
    return cases
