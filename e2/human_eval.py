"""P7 人間テストの集計（仕様§7.3）。

回答 `answers_R{1,2,3}.txt`（`番号: 記号`）を key で正準順の選択肢へ戻し、case ごとに
- 多数決: X 以外の票で 2 人以上一致した選択肢（無ければ一致計算から外す）
- 経験分布 q: X 以外の票の割合（X 以外が 2 票未満の case は Brier/NLL から外す）
を作る。X（判断不能）は票に数えない。予測器ごとに人間との一致・Brier・NLL を出す。

使い方: python3 -m e2.human_eval [--predictions name=path.json ...]
predictions JSON は {case_id: [正準順の確率]}。教師（manifest）と本家Jev（raw jsonl）は常に含める。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from e2.jev_eval import to_distribution

HUMAN = Path("data/e2/human_test")
RATERS = ("R1", "R2", "R3")


def load_votes() -> dict[str, list[int | None]]:
    votes: dict[str, list[int | None]] = {}
    for rater in RATERS:
        answers = {int(n): x for n, x in re.findall(r"^(\d+): ?([A-Z])\s*$", (HUMAN / f"answers_{rater}.txt").read_text(), re.M)}
        for entry in json.loads((HUMAN / f"key_{rater}.json").read_text()):
            letter = answers[entry["number"]]
            canonical = None if letter == "X" else entry["display_to_canonical"][ord(letter) - 65]
            votes.setdefault(entry["case_id"], []).append(canonical)
    return votes


def targets(votes: dict[str, list[int | None]], sizes: dict[str, int]) -> tuple[dict[str, int], dict[str, list[float]]]:
    majority, empirical = {}, {}
    for case_id, vs in votes.items():
        valid = [v for v in vs if v is not None]
        if valid:
            option, count = Counter(valid).most_common(1)[0]
            if count >= 2:
                majority[case_id] = option
        if len(valid) >= 2:
            empirical[case_id] = [valid.count(i) / len(valid) for i in range(sizes[case_id])]
    return majority, empirical


def score(predictions: dict[str, list[float]], majority: dict[str, int], empirical: dict[str, list[float]], kinds: dict[str, str]) -> dict:
    result = {}
    for kind in ("all", "choice", "score", "noul"):
        keep = lambda c: kind == "all" or kinds[c] == kind
        agree = [max(range(len(predictions[c])), key=predictions[c].__getitem__) == m for c, m in majority.items() if keep(c)]
        cases = [c for c in empirical if keep(c)]
        brier = [sum((p - q) ** 2 for p, q in zip(predictions[c], empirical[c])) / len(empirical[c]) for c in cases]
        nll = [-sum(q * math.log(max(p, 1e-12)) for p, q in zip(predictions[c], empirical[c])) for c in cases]
        result[kind] = {"agreement": sum(agree) / len(agree), "n_majority": len(agree),
                        "brier": sum(brier) / len(brier), "nll": sum(nll) / len(nll), "n_empirical": len(cases)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="*", default=[], help="name=path.json")
    parser.add_argument("--out", type=Path, default=Path("results/artifacts/e2_human_eval.json"))
    args = parser.parse_args()

    votes = load_votes()
    rows = {r["case_id"]: r for r in map(json.loads, Path("data/e2/teacher_branches.jsonl").read_text().splitlines()) if r["case_id"] in votes}
    sizes = {c: len(r["options"]) for c, r in rows.items()}
    kinds = {c: r["kind"] for c, r in rows.items()}
    majority, empirical = targets(votes, sizes)

    predictors = {"teacher": {c: r["probabilities"] for c, r in rows.items()}}
    jev = {}
    for record in map(json.loads, Path("results/artifacts/e2_jev_eval_raw.jsonl").read_text().splitlines()):
        for case_id, answer in record["answers"].items():
            if case_id in rows:
                jev[case_id] = to_distribution(rows[case_id], answer)
    predictors["jev"] = jev
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        predictors[name] = json.loads(Path(path).read_text())

    x_counts = Counter(sum(v is None for v in vs) for vs in votes.values())
    unanimous = sum(1 for vs in votes.values() if None not in vs and len(set(vs)) == 1)
    pairwise = [sum(a == b for a, b in ((vs[i], vs[j]) for vs in votes.values() if vs[i] is not None and vs[j] is not None))
                / sum(1 for vs in votes.values() if vs[i] is not None and vs[j] is not None) for i, j in ((0, 1), (0, 2), (1, 2))]
    report = {
        "cases": len(votes), "x_per_case": dict(sorted(x_counts.items())), "all_x_free_unanimous": unanimous,
        "rater_pairwise_agreement": dict(zip(("R1-R2", "R1-R3", "R2-R3"), pairwise)),
        "excluded_no_majority": len(votes) - len(majority), "excluded_lt2_valid_votes": len(votes) - len(empirical),
        "predictors": {name: score(p, majority, empirical, kinds) for name, p in predictors.items()},
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
