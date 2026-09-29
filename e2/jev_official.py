"""公式例20問を今の jev-1.13.0 にAPIで投げ、掲載値と並べる。GPU不要。

使い方: set -a && . ./.env && set +a && python3 -m e2.jev_official
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from e2.jev_eval import MODEL, URL
from e2.official_compare import _cases, _official_distribution


def _question(case: dict) -> dict:
    branch = case["branch"]
    if branch.kind == "noul":
        return {"type": "noul", "instructions": branch.instructions}
    if branch.kind == "choice":
        return {"type": "choice", "instructions": branch.instructions,
                "criteria": {name: option for name, option in zip(case["names"], branch.options)}}
    return {"type": "score", "instructions": branch.instructions, "criteria": branch.options}


def _distribution(case: dict, answer: dict) -> list[float]:
    if case["branch"].kind == "noul":
        return [1.0 - answer["noul"], answer["noul"]]
    probs = answer["probabilities"]
    keys = case["names"] if case["branch"].kind == "choice" else [str(i) for i in range(len(case["branch"].options))]
    return [float(probs[k]) for k in keys]


def main() -> None:
    key = os.environ["JEV_API_KEY"]
    cases = _cases()
    by_state: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        by_state.setdefault(case["state"], []).append(index)
    rows, tokens = [None] * len(cases), 0
    for state, indices in by_state.items():
        questions = {f"q{i}": _question(cases[i]) for i in indices}
        response = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=120,
                                 json={"model": MODEL, "state": state, "questions": questions})
        response.raise_for_status()
        body = response.json(); tokens += body["usage"]["input_tokens"]
        for i in indices:
            case = cases[i]
            rows[i] = {"case_id": case["case_id"], "kind": case["branch"].kind,
                       "official": _official_distribution(case["branch"].kind, case["names"], case["official"]),
                       "official_raw": case["official"], "jev_now": _distribution(case, body["answers"][f"q{i}"]),
                       "jev_now_raw": body["answers"][f"q{i}"]}
    out = Path("results/artifacts/e2_jev_official.json")
    out.write_text(json.dumps({"model": MODEL, "input_tokens": tokens, "rows": rows}, indent=2) + "\n")
    print("input_tokens", tokens)
    for r in rows:
        print(r["case_id"], "| official", r["official"], "| jev_now", [round(x, 3) for x in r["jev_now"]])


if __name__ == "__main__":
    main()
