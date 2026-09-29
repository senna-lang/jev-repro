"""選定LoRA − 学習前の paired bootstrap（10,000回、仕様§7.2）。

教師テスト: eval 1,200 branch を case 単位で復元抽出し、top-1・Brier・NLL の差の 95% CI を全体と kind 別に出す。
人間テスト: 多数決/経験分布が作れた case で、一致・Brier・NLL の差の 95% CI を出す（`e2.human_eval` と同じ定義）。
改善方向: top-1/一致は +、Brier/NLL は − が改善。

使い方: python3 -m e2.bootstrap
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from e2.human_eval import load_votes, targets

RESAMPLES = 10_000
SEED = 20260928


def per_case(p: list[float], q: list[float], target: int | None = None) -> tuple[float, float, float]:
    k = len(q)
    top = max(range(k), key=q.__getitem__) if target is None else target
    return (float(max(range(k), key=p.__getitem__) == top),
            sum((a - b) ** 2 for a, b in zip(p, q)) / k,
            -sum(b * math.log(max(a, 1e-12)) for a, b in zip(p, q)))


def ci(diffs: np.ndarray, rng: np.random.Generator) -> dict:
    idx = rng.integers(0, len(diffs), size=(RESAMPLES, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(diffs.mean()), "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0), "n": len(diffs)}


def main() -> None:
    base = json.loads(Path("results/e2_pred_baseline.json").read_text())
    lora = json.loads(Path("results/e2_pred_selected.json").read_text())
    rows = {r["case_id"]: r for r in map(json.loads, Path("data/e2/teacher_branches.jsonl").read_text().splitlines()) if r["split"] == "eval"}
    rng = np.random.default_rng(SEED)
    report = {"resamples": RESAMPLES, "seed": SEED, "teacher": {}, "human": {}}

    for kind in ("all", "choice", "score", "noul"):
        cases = [c for c, r in rows.items() if kind == "all" or r["kind"] == kind]
        d = np.array([np.subtract(per_case(lora[c], rows[c]["probabilities"]), per_case(base[c], rows[c]["probabilities"])) for c in cases])
        report["teacher"][kind] = {m: ci(d[:, i], rng) for i, m in enumerate(("top1", "brier", "nll"))}

    votes = load_votes()
    majority, empirical = targets(votes, {c: len(rows[c]["options"]) for c in votes})
    for kind in ("all", "choice", "score", "noul"):
        keep = lambda c: kind == "all" or rows[c]["kind"] == kind
        agree = np.array([per_case(lora[c], [0] * len(lora[c]), m)[0] - per_case(base[c], [0] * len(base[c]), m)[0] for c, m in majority.items() if keep(c)])
        dist = np.array([np.subtract(per_case(lora[c], q)[1:], per_case(base[c], q)[1:]) for c, q in empirical.items() if keep(c)])
        report["human"][kind] = {"agreement": ci(agree, rng), "brier": ci(dist[:, 0], rng), "nll": ci(dist[:, 1], rng)}

    Path("results/e2_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    for section in ("teacher", "human"):
        for kind, metrics in report[section].items():
            print(section, kind, {m: (round(v["diff"], 4), [round(x, 4) for x in v["ci95"]], v["excludes_zero"]) for m, v in metrics.items()})


if __name__ == "__main__":
    main()
