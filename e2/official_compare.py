"""公式ドキュメント掲載のJev出力と、E2の学習前Qwen3-4B・選定LoRAを定性的に並べる。

掲載値は正解ラベルではない。checkpoint選定・学習には使わない。
入力はE2評価と同じchat形式・同じ順番平均（Choice/Score 4通り、Noul 2通り）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from scipy.stats import kendalltau

from e2.evaluate_lora import option_orders, predict_request
from e2.train_lora import load_lora_model
from legacy.experiments.official_compare import EXAMPLES
from model.packer import Branch

ADAPTER = Path("checkpoints/e2/lr-1e-4/epoch-3-quarter-3")
NOUL_OPTIONS = ["No", "Yes"]


def _official_distribution(kind: str, names: list[str], official: dict) -> list[float] | None:
    """掲載値をcanonical順の分布へ直す。Noulは[No, Yes]。確率が未掲載のScoreはNone。"""
    if kind == "noul":
        return [1.0 - official["noul"], official["noul"]]
    probs = official.get("probabilities")
    if probs is None:
        return None
    return [probs[name] for name in names] if isinstance(probs, dict) else list(probs)


def _cases() -> list[dict]:
    """全例を (case_id, state, Branch, 選択肢名, 掲載値) へ平らにする。"""
    cases = []
    for example in EXAMPLES:
        if example.get("per_state"):
            branch = example["score_branch"]
            for index, (state, official) in enumerate(example["per_state"]):
                cases.append({"case_id": f"severity_table/{index}", "group": "severity_table", "state": state, "branch": branch,
                              "names": [f"L{i}" for i in range(len(branch.options))], "official": official})
            continue
        for qid, branch, names, official in example["questions"]:
            if branch.kind == "noul":
                branch = Branch("noul", branch.instructions, NOUL_OPTIONS)
                names = ["no", "yes"]
            cases.append({"case_id": f"{example['name']}/{qid}", "group": example["name"], "state": example["state"],
                          "branch": branch, "names": names, "official": official})
    return cases


def _predict_all(model, tokenizer, cases: list[dict], device) -> list[list[float]]:
    """同じstateの設問は1リクエストにpackする（本家の1 call複数設問と同じ）。"""
    results: list[list[float] | None] = [None] * len(cases)
    by_state: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        by_state.setdefault(case["state"], []).append(index)
    for state, indices in by_state.items():
        branches = [cases[i]["branch"] for i in indices]
        orders = [option_orders(b.kind, len(b.options), cases[i]["case_id"]) for i, b in zip(indices, branches)]
        for i, p in zip(indices, predict_request(model, tokenizer, state, branches, orders, device)):
            results[i] = p
    return results


def _summary(kind: str, p: list[float]) -> dict:
    if kind == "noul":
        return {"yes": p[1]}
    if kind == "score":
        return {"score": sum(i * v for i, v in enumerate(p)), "argmax": max(range(len(p)), key=p.__getitem__)}
    return {"argmax": max(range(len(p)), key=p.__getitem__)}


def main() -> None:
    device = torch.device("cuda")
    model, tokenizer = load_lora_model(ADAPTER)
    model.to(device).eval()
    cases = _cases()
    with model.disable_adapter():
        baseline = _predict_all(model, tokenizer, cases, device)
    selected = _predict_all(model, tokenizer, cases, device)

    rows = []
    for case, p_base, p_lora in zip(cases, baseline, selected):
        kind = case["branch"].kind
        official = _official_distribution(kind, case["names"], case["official"])
        row = {"case_id": case["case_id"], "kind": kind, "names": case["names"], "official_raw": case["official"],
               "official": official, "baseline": p_base, "selected": p_lora}
        for label, p in (("baseline", p_base), ("selected", p_lora)):
            row[f"{label}_summary"] = _summary(kind, p)
            if official is not None:
                row[f"{label}_tvd"] = 0.5 * sum(abs(a - b) for a, b in zip(p, official))
                row[f"{label}_argmax_match"] = max(range(len(p)), key=p.__getitem__) == max(range(len(official)), key=official.__getitem__)
        rows.append(row)

    severity = [r for r in rows if r["case_id"].startswith("severity_table/")]
    official_scores = [r["official_raw"]["score"] for r in severity]
    report = {"rows": rows, "severity_table_tau": {}}
    for label in ("baseline", "selected"):
        tau, p_value = kendalltau(official_scores, [r[f"{label}_summary"]["score"] for r in severity])
        report["severity_table_tau"][label] = {"tau": tau, "p": p_value}
        comparable = [r for r in rows if f"{label}_tvd" in r]
        report[f"{label}_totals"] = {
            "n": len(comparable),
            "argmax_match": sum(r[f"{label}_argmax_match"] for r in comparable),
            "mean_tvd": sum(r[f"{label}_tvd"] for r in comparable) / len(comparable),
        }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/e2-results/official_compare.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for r in rows:
        print(r["case_id"], r["kind"], "official", r["official_raw"], "| base", r["baseline_summary"], "| lora", r["selected_summary"])


if __name__ == "__main__":
    main()
