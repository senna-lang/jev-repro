"""Mac MLX latency and teacher-eval metrics for the E2 LoRA.

  python -m mac.bench latency [--limit 10] [--single]
  python -m mac.bench eval    [--limit 60] [--single] [--out results/artifacts/mac/eval_mlx_fp16.json]

Uses the teacher eval split (data/e2). Fixed-candidate evaluation only; never for selection, never Gold V1.
Default is order averaging (e2.evaluate_lora.option_orders); --single uses the canonical order only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from e2.evaluate_lora import option_orders, teacher_metrics
from e2.packed_data import load_requests
from mac.jev_mlx import JevMLX
from model.packer import Branch


def eval_requests(limit: int | None):
    requests = [r for r in load_requests(Path("data/e2/teacher_branches.jsonl"), Path("data/e2/states.jsonl")) if r[0][0]["split"] == "eval"]
    for rows, request in requests[:limit]:
        branches = [Branch(b.kind, b.instructions, row["options"]) for row, b in zip(rows, request.branches)]
        yield rows, request.state, branches


def orders_for(rows, single: bool):
    if single:
        return None
    return [option_orders(row["kind"], len(row["options"]), row["case_id"]) for row in rows]


def latency(jev: JevMLX, limit: int, single: bool) -> dict:
    items = list(eval_requests(limit))
    for rows, state, branches in items:  # warm up every input shape
        jev.predict_request(state, branches, orders_for(rows, single))
    times: list[list[float]] = [[] for _ in items]
    for _ in range(5):  # round-robin so drift affects every request equally
        for i, (rows, state, branches) in enumerate(items):
            start = time.perf_counter()
            jev.predict_request(state, branches, orders_for(rows, single))
            times[i].append((time.perf_counter() - start) * 1000)
    per_request = [round(statistics.median(t), 1) for t in times]
    return {"median_ms": round(statistics.median(per_request), 1), "per_request_ms": per_request}


def evaluate(jev: JevMLX, limit: int | None, single: bool) -> dict:
    pairs, predictions, start = [], [], time.time()
    for rows, state, branches in eval_requests(limit):
        for row, p in zip(rows, jev.predict_request(state, branches, orders_for(rows, single))):
            pairs.append((row["kind"], p, row["probabilities"]))
            predictions.append({"case_id": row["case_id"], "kind": row["kind"], "p": p})
    return {"seconds": round(time.time() - start), "branches": len(pairs), "metrics": teacher_metrics(pairs), "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["latency", "eval"])
    parser.add_argument("--limit", type=int, help="number of eval requests (default: 10 latency / all eval)")
    parser.add_argument("--single", action="store_true", help="canonical order only (no order averaging)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    jev = JevMLX()
    if args.mode == "latency":
        result = latency(jev, args.limit or 10, args.single)
    else:
        result = evaluate(jev, args.limit, args.single)
    result = {"mode": args.mode, "single": args.single, "limit": args.limit, **result}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("predictions", "per_request_ms")}, indent=1))


if __name__ == "__main__":
    main()
