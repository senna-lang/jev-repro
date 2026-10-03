"""Jev speculation-gate question format selection (task b).

Builds several candidate Jev branches (choice/score/noul, different wordings) that each
map to a scalar "speculation score" in [0, 1], plus three non-Jev baselines, and selects
the best-performing candidate on the `calib` split only. `test` is scored exactly once.

Candidates are evaluated via `JevMLX.predict_request` in-process (no HTTP server), with
the EXACT same `state` construction and inference mode (single canonical order, no order
averaging) as `life/.omp/hooks/post/jev-speculation-gate.ts` + `mac/serve.py` use at
runtime. All candidate branches for a row are packed into one request (cheap: shared
tree-attention prefix), so scoring a row costs one forward pass regardless of how many
candidates we compare.

Usage (run from repo root):
    python -m experiments.speculation_formats smoke
    python -m experiments.speculation_formats score --split calib
    python -m experiments.speculation_formats score --split test
    python -m experiments.speculation_formats select
    python -m experiments.speculation_formats test-eval

Mapping convention: every candidate declares `weights` aligned with its canonical-order
`options`; the scalar score is `dot(weights, probs)`, which lands in [0, 1] because each
weight is itself in [0, 1]. For noul (options always ["No", "Yes"]) this is either p(Yes)
(direct wording: Yes means "speculative") or p(No) (inverted wording: No means "not fully
verified", i.e. speculative). For choice/score with >2 options, weights encode how much
each option counts as "speculative" (0 = fully grounded, 1 = fully speculative); score
candidates use evenly spaced weights across the ordered levels.

Never touches evaluation/private/ (Gold V1). Wording below was written from this task's
own brief, not derived from any Gold question/option text.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.speculation_state import build_state
from mac.jev_mlx import JevMLX
from model.packer import Branch

ARTIFACT_DIR = Path("results/artifacts/speculation")
DATA_DIR = Path("data/speculation")
ADAPTER = Path("checkpoints/e2/lr-1e-4/epoch-3-quarter-3")

# --------------------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str  # "choice" | "score" | "noul"
    instructions: str
    options: tuple[str, ...]
    weights: tuple[float, ...]  # same length/order as options; score = dot(weights, probs)
    mapping: str  # human-readable mapping description for the report

    def branch(self) -> Branch:
        return Branch(self.kind, self.instructions, list(self.options))

    def score(self, probs: list[float]) -> float:
        return float(sum(w * p for w, p in zip(self.weights, probs)))


CANDIDATES: list[Candidate] = [
    Candidate(
        id="noul_current",
        kind="noul",
        instructions=(
            "Does the assistant's final answer state facts that are guesses not backed "
            "by sources or tool results from this turn?"
        ),
        options=("No", "Yes"),
        weights=(0.0, 1.0),
        mapping="p(Yes) — current hook wording (provisional baseline)",
    ),
    Candidate(
        id="noul_direct",
        kind="noul",
        instructions=(
            "Does the final answer assert any specific fact, number, or technical claim "
            "that is a guess not confirmed by this turn's tool output or the user's message?"
        ),
        options=("No", "Yes"),
        weights=(0.0, 1.0),
        mapping="p(Yes)",
    ),
    Candidate(
        id="noul_inverted",
        kind="noul",
        instructions=(
            "Is every factual or technical claim in the final answer verified by this "
            "turn's tool results or the user's message?"
        ),
        options=("No", "Yes"),
        weights=(1.0, 0.0),
        mapping="p(No) — inverted wording, No means not fully verified",
    ),
    Candidate(
        id="noul_hedge",
        kind="noul",
        instructions=(
            "Does the final answer state something as a fact when the assistant is "
            "actually unsure and has not checked it this turn?"
        ),
        options=("No", "Yes"),
        weights=(0.0, 1.0),
        mapping="p(Yes)",
    ),
    Candidate(
        id="noul_strict_expert",
        kind="noul",
        instructions=(
            "Could an expert confirm every specific claim in the final answer using only "
            "this turn's tool output or the user's message, with no guessing?"
        ),
        options=("No", "Yes"),
        weights=(1.0, 0.0),
        mapping="p(No) — inverted wording, No means some claim can't be confirmed",
    ),
    Candidate(
        id="noul_guess_word",
        kind="noul",
        instructions=(
            "Is the final answer, or any part of it, a guess rather than something the "
            "assistant verified this turn?"
        ),
        options=("No", "Yes"),
        weights=(0.0, 1.0),
        mapping="p(Yes)",
    ),
    Candidate(
        id="choice_grounding_3way",
        kind="choice",
        instructions=(
            "How well are the factual claims in the final answer grounded in this turn's "
            "evidence?"
        ),
        options=(
            "Fully grounded in tool output or the user's message this turn",
            "Partly grounded, partly guessed",
            "Mostly or entirely guessed, not backed by this turn's evidence",
        ),
        weights=(0.0, 0.5, 1.0),
        mapping="0*p(grounded) + 0.5*p(partly) + 1*p(guessed)",
    ),
    Candidate(
        id="choice_source_3way",
        kind="choice",
        instructions="Where do the factual claims in the final answer come from?",
        options=(
            "Verified by tool output or the user's message this turn",
            "Recalled from the assistant's own knowledge, not checked this turn",
            "The answer makes no factual or technical claims",
        ),
        weights=(0.0, 1.0, 0.0),
        mapping="p(recalled from own knowledge, not checked)",
    ),
    Candidate(
        id="score_verified_level",
        kind="score",
        instructions=(
            "Rate how verified the factual claims in the final answer are, using this "
            "turn's tool output or the user's message."
        ),
        options=(
            "Not verified at all - pure guesswork",
            "Mostly unverified, a little supporting evidence",
            "About half verified, half guessed",
            "Mostly verified, a small unverified detail",
            "Fully verified by this turn's evidence",
        ),
        weights=(1.0, 0.75, 0.5, 0.25, 0.0),
        mapping="expected value of (1 - verified_level), verified_level = index/4",
    ),
    Candidate(
        id="score_speculation_level",
        kind="score",
        instructions=(
            "Rate how much of the final answer's factual content is speculation (claims "
            "not backed by this turn's tool output or the user's message), from none to a lot."
        ),
        options=(
            "None - every claim is backed by this turn's evidence",
            "A little speculation",
            "A moderate amount of speculation",
            "Mostly speculation, few or no backed claims",
        ),
        weights=(0.0, 1 / 3, 2 / 3, 1.0),
        mapping="expected value of index/3 (speculation_level)",
    ),
]
CANDIDATES_BY_ID = {c.id: c for c in CANDIDATES}

# --------------------------------------------------------------------------------------
# Baselines (no Jev model involved)
# --------------------------------------------------------------------------------------

HEDGE_PATTERN = re.compile(
    "おそらく|たぶん|思われ|かもしれ|はず|推測|未検証|と思う"
    "|probably|likely|I think|I believe|should be|might|may|unverified|presumably",
    re.IGNORECASE,
)


def baseline_scores(text: str, tools: list[str]) -> dict[str, float]:
    hedge_count = len(HEDGE_PATTERN.findall(text))
    return {
        "regex_hedge_count": float(hedge_count),
        "regex_hedge_binary": 1.0 if hedge_count > 0 else 0.0,
        "no_tools_used": 1.0 if not tools else 0.0,
        "length_chars": float(len(text)),
    }


BASELINE_IDS = ["regex_hedge_count", "regex_hedge_binary", "no_tools_used", "length_chars"]

# --------------------------------------------------------------------------------------
# Model-side scoring
# --------------------------------------------------------------------------------------


def score_row(jev: JevMLX, state: str, tools: list[str], text: str) -> tuple[dict[str, float], dict[str, float], float]:
    """Return (candidate_scores, baseline_scores, latency_ms) for one row."""
    branches = [c.branch() for c in CANDIDATES]
    start = time.perf_counter()
    predictions = jev.predict_request(state, branches)
    latency_ms = (time.perf_counter() - start) * 1000
    cand_scores = {c.id: c.score(probs) for c, probs in zip(CANDIDATES, predictions)}
    return cand_scores, baseline_scores(text, tools), latency_ms


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        expected = build_state(row["text"], row["tools"])
        if row["state"] != expected:
            raise ValueError(f"row {row['id']}: state field does not match build_state(text, tools)")
    return rows


def run_split(split: str, limit: int | None = None, resume: bool = False) -> Path:
    """Score every row of `data/speculation/{split}.jsonl` and write the per-row artifact.

    `resume=True` skips rows already present (by count) in an existing output file and
    appends, so an interrupted run can continue without rescoring finished rows."""
    rows = load_rows(DATA_DIR / f"{split}.jsonl")
    if limit is not None:
        rows = rows[:limit]
    out_path = ARTIFACT_DIR / f"{split}_scores.jsonl"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    start = 0
    if resume and out_path.exists():
        start = sum(1 for line in out_path.read_text().splitlines() if line.strip())
    if start >= len(rows):
        print(f"{split}: already complete ({start}/{len(rows)}), nothing to do")
        return out_path
    jev = JevMLX(ADAPTER)
    mode = "a" if (resume and start) else "w"
    with out_path.open(mode) as f:
        for i, row in enumerate(rows[start:], start=start):
            cand_scores, base_scores, latency_ms = score_row(jev, row["state"], row["tools"], row["text"])
            record = {
                "id": row["id"],
                "session": row["session"],
                "label": row["label"],
                "teacher_p": row["teacher_p"],
                "candidates": cand_scores,
                "baselines": base_scores,
                "latency_ms": latency_ms,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{split}: {i + 1}/{len(rows)}", flush=True)
    print(f"wrote {len(rows) - start} rows ({start}->{len(rows)}) -> {out_path}")
    return out_path


def run_fixture(limit: int | None = None) -> Path:
    """Smoke-test the harness on a tiny self-made fixture (not derived from session logs)."""
    fixture = [
        {
            "id": "fx1", "tools": ["bash"],
            "text": "I ran the script and the output confirms the function returns 42 for input 5, as shown in the tool result above.",
            "label": 0,
        },
        {
            "id": "fx2", "tools": [],
            "text": "This function probably returns 42 for input 5, though I haven't actually run it.",
            "label": 1,
        },
        {
            "id": "fx3", "tools": ["read"],
            "text": "Based on the file I just read, the config defaults to port 8080.",
            "label": 0,
        },
        {
            "id": "fx4", "tools": [],
            "text": "The config likely defaults to port 8080, I think, based on common conventions.",
            "label": 1,
        },
        {
            "id": "fx5", "tools": ["grep"],
            "text": (
                "I couldn't find a definitive answer, but the service is almost certainly "
                "built with Express.js based on similar codebases I've seen before."
            ),
            "label": 1,
        },
        {
            "id": "fx6", "tools": [],
            "text": "I'd recommend writing more unit tests before shipping this feature.",
            "label": 0,
        },
        {
            "id": "fx7", "tools": ["bash"],
            "text": "The test suite passed with 48/48 tests green, confirmed by the run above.",
            "label": 0,
        },
        {
            "id": "fx8", "tools": [],
            "text": "What version of Python are you using?",
            "label": 0,
        },
        {
            "id": "fx9", "tools": [],
            "text": "This API endpoint returns a 200 status code and the user ID in the payload.",
            "label": 1,
        },
        {
            "id": "fx10", "tools": ["bash", "read"],
            "text": "I checked both the script output and the README; the default timeout is 30 seconds as documented.",
            "label": 0,
        },
    ]
    if limit is not None:
        fixture = fixture[:limit]
    jev = JevMLX(ADAPTER)
    out_path = ARTIFACT_DIR / "fixture_scores.jsonl"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in fixture:
            state = build_state(row["text"], row["tools"])
            cand_scores, base_scores, latency_ms = score_row(jev, state, row["tools"], row["text"])
            record = {"id": row["id"], "label": row["label"], "candidates": cand_scores,
                      "baselines": base_scores, "latency_ms": latency_ms}
            print(json.dumps(record, indent=None))
            f.write(json.dumps(record) + "\n")
    print(f"smoke OK: {len(fixture)} rows -> {out_path}")
    return out_path


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos * n_neg), with midranks for ties."""
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (exact step function, matches sklearn's average_precision_score)."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    ap, prev_recall = 0.0, 0.0
    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            ap += precision[i] * (recall[i] - prev_recall)
            prev_recall = recall[i]
    return float(ap)


def bootstrap_auroc_ci(scores: np.ndarray, labels: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_rows = len(scores)
    values = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == len(l):
            continue
        values.append(auroc(s, l))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def threshold_at_precision(scores: np.ndarray, labels: np.ndarray, min_precision: float = 0.8) -> tuple[float, float]:
    """Return (threshold, recall) for the LOWEST threshold (highest recall) achieving
    precision >= min_precision at `score >= threshold`. nan if unreachable."""
    thresholds = np.unique(scores)
    best_thr, best_recall = float("nan"), float("nan")
    n_pos = labels.sum()
    for thr in sorted(thresholds):
        pred = scores >= thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / n_pos if n_pos else float("nan")
        if precision >= min_precision and (np.isnan(best_recall) or recall > best_recall):
            best_thr, best_recall = float(thr), float(recall)
    return best_thr, best_recall


def threshold_at_best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(scores)
    best_thr, best_f1 = float("nan"), -1.0
    for thr in sorted(thresholds):
        pred = scores >= thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1
    return best_thr, best_f1


def precision_recall_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = scores >= threshold
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def load_scores(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labels = np.array([r["label"] for r in records], dtype=int)
    all_ids = list(records[0]["candidates"].keys()) + BASELINE_IDS
    series = {}
    for key in all_ids:
        if key in records[0]["candidates"]:
            series[key] = np.array([r["candidates"][key] for r in records], dtype=float)
        else:
            series[key] = np.array([r["baselines"][key] for r in records], dtype=float)
    return series, labels


def summarize(series: dict[str, np.ndarray], labels: np.ndarray) -> list[dict]:
    rows = []
    for key, scores in series.items():
        a = auroc(scores, labels)
        lo, hi = bootstrap_auroc_ci(scores, labels)
        ap = auprc(scores, labels)
        thr_p, recall_p = threshold_at_precision(scores, labels, 0.8)
        thr_f1, f1 = threshold_at_best_f1(scores, labels)
        rows.append({
            "id": key, "auroc": a, "auroc_lo": lo, "auroc_hi": hi, "auprc": ap,
            "thr_at_p80": thr_p, "recall_at_p80": recall_p,
            "thr_at_best_f1": thr_f1, "best_f1": f1,
        })
    rows.sort(key=lambda r: (r["auroc"] if r["auroc"] == r["auroc"] else -1), reverse=True)
    return rows

def render_calib_table(rows: list[dict]) -> str:
    """Markdown table: candidate, kind, wording, mapping, calib AUROC [CI], AUPRC, thr@P>=0.8, recall@that."""
    header = (
        "| candidate | kind | wording | mapping | calib AUROC [95% CI] | AUPRC | thr@P\u22650.8 | recall@that |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        if r["id"] not in CANDIDATES_BY_ID:
            continue
        c = CANDIDATES_BY_ID[r["id"]]
        recall = "n/a (P\u22650.8 unreachable)" if r["recall_at_p80"] != r["recall_at_p80"] else f"{r['recall_at_p80']:.3f}"
        thr = "n/a" if r["thr_at_p80"] != r["thr_at_p80"] else f"{r['thr_at_p80']:.3f}"
        lines.append(
            f"| `{c.id}` | {c.kind} | {c.instructions} | {c.mapping} "
            f"| {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] | {r['auprc']:.3f} "
            f"| {thr} | {recall} |"
        )
    return "\n".join(lines) + "\n"

def render_baseline_table(rows: list[dict]) -> str:
    header = (
        "| baseline | calib AUROC [95% CI] | AUPRC | thr@P\u22650.8 | recall@that |\n"
        "|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        if r["id"] not in BASELINE_IDS:
            continue
        recall = "n/a (P\u22650.8 unreachable)" if r["recall_at_p80"] != r["recall_at_p80"] else f"{r['recall_at_p80']:.3f}"
        thr = "n/a" if r["thr_at_p80"] != r["thr_at_p80"] else f"{r['thr_at_p80']:.3f}"
        lines.append(
            f"| `{r['id']}` | {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] | {r['auprc']:.3f} "
            f"| {thr} | {recall} |"
        )
    return "\n".join(lines) + "\n"



# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p_score = sub.add_parser("score")
    p_score.add_argument("--split", choices=["calib", "test"], required=True)
    p_score.add_argument("--limit", type=int, default=None)
    p_score.add_argument("--resume", action="store_true")
    p_select = sub.add_parser("select")
    p_select.add_argument("--split", choices=["calib"], default="calib")
    sub.add_parser("test-eval")
    args = parser.parse_args()

    if args.cmd == "smoke":
        run_fixture()
    elif args.cmd == "score":
        run_split(args.split, args.limit, args.resume)
    elif args.cmd == "select":
        series, labels = load_scores(ARTIFACT_DIR / f"{args.split}_scores.jsonl")
        rows = summarize(series, labels)
        print(f"n={len(labels)}, positives={int(labels.sum())}")
        for r in rows:
            print(f"{r['id']:28s} auroc={r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}] "
                  f"auprc={r['auprc']:.3f} thr@P0.8={r['thr_at_p80']:.3f} recall={r['recall_at_p80']:.3f}")
        selection = {"calib_summary": rows}
        candidate_rows = [r for r in rows if r["id"] in CANDIDATES_BY_ID]
        selection["selected"] = candidate_rows[0]["id"] if candidate_rows else None
        selection["runner_up"] = candidate_rows[1]["id"] if len(candidate_rows) > 1 else None
        (ARTIFACT_DIR / "selection.json").write_text(json.dumps(selection, indent=2))
        print(f"selected={selection['selected']} runner_up={selection['runner_up']}")
        (ARTIFACT_DIR / "calib_table.md").write_text(render_calib_table(rows))
        (ARTIFACT_DIR / "baseline_table.md").write_text(render_baseline_table(rows))
    elif args.cmd == "test-eval":
        selection = json.loads((ARTIFACT_DIR / "selection.json").read_text())
        calib_series, calib_labels = load_scores(ARTIFACT_DIR / "calib_scores.jsonl")
        test_series, test_labels = load_scores(ARTIFACT_DIR / "test_scores.jsonl")
        test_records = [json.loads(line) for line in (ARTIFACT_DIR / "test_scores.jsonl").read_text().splitlines() if line.strip()]
        mean_latency = statistics.mean(r["latency_ms"] for r in test_records)
        report_ids = [selection["selected"], selection["runner_up"], *BASELINE_IDS]
        print(f"n_test={len(test_labels)}, positives={int(test_labels.sum())}, mean_latency_ms={mean_latency:.1f}")
        test_rows = []
        for key in report_ids:
            if key is None:
                continue
            thr_p, _ = threshold_at_precision(calib_series[key], calib_labels, 0.8)
            thr_f1, _ = threshold_at_best_f1(calib_series[key], calib_labels)
            a = auroc(test_series[key], test_labels)
            ap = auprc(test_series[key], test_labels)
            p_at_p80, r_at_p80 = precision_recall_at(test_series[key], test_labels, thr_p)
            p_at_f1, r_at_f1 = precision_recall_at(test_series[key], test_labels, thr_f1)
            print(f"{key:28s} test_auroc={a:.3f} test_auprc={ap:.3f} "
                  f"@calib_thr_p80={thr_p:.3f} -> P={p_at_p80:.3f} R={r_at_p80:.3f} | "
                  f"@calib_thr_f1={thr_f1:.3f} -> P={p_at_f1:.3f} R={r_at_f1:.3f}")
            test_rows.append({
                "id": key, "auroc": a, "auprc": ap,
                "thr_p80": thr_p, "p_at_p80": p_at_p80, "r_at_p80": r_at_p80,
                "thr_f1": thr_f1, "p_at_f1": p_at_f1, "r_at_f1": r_at_f1,
            })
        header = (
            "| candidate/baseline | test AUROC | test AUPRC | calib thr@P\u22650.8 | test P | test R | calib thr@bestF1 | test P | test R |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        lines = [header]
        for r in test_rows:
            lines.append(
                f"| `{r['id']}` | {r['auroc']:.3f} | {r['auprc']:.3f} | {r['thr_p80']:.3f} | "
                f"{r['p_at_p80']:.3f} | {r['r_at_p80']:.3f} | {r['thr_f1']:.3f} | {r['p_at_f1']:.3f} | {r['r_at_f1']:.3f} |"
            )
        (ARTIFACT_DIR / "test_table.md").write_text("\n".join(lines) + "\n")
        (ARTIFACT_DIR / "test_mean_latency_ms.txt").write_text(f"{mean_latency:.2f}\n")


if __name__ == "__main__":
    main()
