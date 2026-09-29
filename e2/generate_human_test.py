from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260927
RATERS = ("R1", "R2", "R3")
SAMPLE_SIZE = 400


def sample(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["domain"])].append(row)
    picked = []
    keys = sorted(groups)
    for key in keys:
        rng = random.Random(f"{SEED}:{key}")
        rng.shuffle(groups[key])
    while len(picked) < SAMPLE_SIZE:
        for key in keys:
            if groups[key] and len(picked) < SAMPLE_SIZE:
                picked.append(groups[key].pop())
    return picked


def make_sheet(rows, rater):
    shuffled = list(rows)
    random.Random(f"{SEED}:{rater}:questions").shuffle(shuffled)
    lines = ["# Human evaluation sheet", "", "Read each record and choose the best-supported answer. Use `X` only when the record does not allow a judgment. Do not consult other raters.", ""]
    key = []
    for number, row in enumerate(shuffled, 1):
        options = list(enumerate(row["options"]))
        if row["kind"] != "score":
            random.Random(f"{SEED}:{rater}:{row['case_id']}").shuffle(options)
        labels = [chr(65 + index) for index in range(len(options))]
        lines += [f"## {number}", "", "### Record", row["state"], "", "### Question", row["question"], ""]
        lines += [f"{label}. {option}" for label, (_, option) in zip(labels, options)]
        lines += ["X. Cannot determine", ""]
        key.append({"number": number, "case_id": row["case_id"], "display_to_canonical": [index for index, _ in options]})
    lines += ["# Answers", "", *[f"{number}: " for number in range(1, len(shuffled) + 1)]]
    return "\n".join(lines) + "\n", key


def main():
    root = Path("data/e2")
    states = {row["state_id"]: row["text"] for row in map(json.loads, (root / "states.jsonl").read_text().splitlines())}
    rows = [row for row in map(json.loads, (root / "teacher_branches.jsonl").read_text().splitlines()) if row["split"] == "eval"]
    for row in rows: row["state"] = states[row["state_id"]]
    selected = sample(rows)
    out = root / "human_test"; out.mkdir(exist_ok=True)
    for rater in RATERS:
        sheet, key = make_sheet(selected, rater)
        (out / f"sheet_{rater}.md").write_text(sheet)
        (out / f"key_{rater}.json").write_text(json.dumps(key, indent=2) + "\n")

if __name__ == "__main__": main()
