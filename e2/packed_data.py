from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from model.packer import Branch, PackedRequest, PackedSequence, pack


def load_requests(branches_path: Path, states_path: Path) -> Iterator[tuple[list[dict], PackedRequest]]:
    states = {row["state_id"]: row["text"] for row in map(json.loads, states_path.read_text().splitlines())}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in map(json.loads, branches_path.read_text().splitlines()):
        groups[row["state_id"]].append(row)
    for state_id, rows in groups.items():
        if len(rows) != 4:
            raise ValueError(f"{state_id}: expected 4 branches, got {len(rows)}")
        branches = []
        for row in rows:
            if row["kind"] == "noul":
                branches.append(Branch(kind="noul", instructions=row["question"]))
            else:
                branches.append(Branch(kind=row["kind"], instructions=row["question"], options=row["options"]))
        yield rows, PackedRequest(state=states[state_id], branches=branches)


def pack_requests(branches_path: Path, states_path: Path, tokenizer) -> Iterator[tuple[list[dict], PackedSequence]]:
    for rows, request in load_requests(branches_path, states_path):
        yield rows, pack(request, tokenizer)
