"""Public zero-shot accuracy on MLX, following the fixed protocol of qwen3-4b-system-one docs/BENCHMARKS.md.

  python -m mac.bench_public --backbone bonsai-4b --adapter checkpoints/bonsai/candidate \
      --output results/artifacts/public_bench/bonsai4b [--limit 500] [--task "AG News"]

500 rows per task from `dataset.shuffle(seed=0).select(range(500))`, one canonical option order, one forward per
request, accuracy of the highest-probability option with a 95% bootstrap CI (2,000 resamples, seed 0).
`base` = the same packed model with the LoRA sidecars bypassed. The gold sequence is checked against the published
Qwen predictions so that both systems are measured on the same 500 rows. Never used for training or selection.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear

from mac.bench import _sidecars
from mac.jev_mlx import JevMLX
from model.packer import Branch

SAMPLE_SIZE, SEED, BOOTSTRAP = 500, 0, 2000
SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
MASSIVE_SCENARIOS = ["alarm", "audio", "calendar", "cooking", "datetime", "email", "general", "iot", "lists",
                     "music", "news", "play", "qa", "recommendation", "social", "takeaway", "transport", "weather"]


def _ag_news(row):
    return row["text"], Branch("choice", "What is the topic of this news article?", ["World", "Sports", "Business", "Sci/Tech"]), row["label"]


def _massive(row):
    question = "Which scenario does this request to a voice assistant belong to?"
    return row["text"], Branch("choice", question, MASSIVE_SCENARIOS), MASSIVE_SCENARIOS.index(row["label_text"])


def _mnli(row):
    state = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
    question = "What is the relationship between the premise and the hypothesis?"
    return state, Branch("choice", question, ["entailment", "neutral", "contradiction"]), row["label"]


def _boolq(row):
    question = row["question"].strip()
    question = question[0].upper() + question[1:] + ("" if question.endswith("?") else "?")
    return row["passage"], Branch("noul", question, ["No", "Yes"]), int(row["answer"])


def _sst5(row):
    return row["text"], Branch("score", "How positive is the sentiment of this text?", SST5_LEVELS), row["label"]


# (name, file slug, hf args, split, builder, kind)
TASKS = [
    ("AG News", "ag", ("fancyzhx/ag_news",), "test", _ag_news, "choice"),
    ("MASSIVE scenario (en-US)", "massive", ("mteb/amazon_massive_scenario", "en"), "test", _massive, "choice"),
    ("MNLI (matched)", "mnli", ("nyu-mll/glue", "mnli"), "validation_matched", _mnli, "choice"),
    ("BoolQ", "boolq", ("google/boolq",), "validation", _boolq, "noul"),
    ("SST-5", "sst5", ("SetFit/sst5",), "test", _sst5, "score"),
]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def bootstrap_ci(values: list[float], rng: random.Random) -> list[float]:
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return [round(means[int(0.025 * BOOTSTRAP)], 4), round(means[int(0.975 * BOOTSTRAP) - 1], 4)]


class Bypass:
    """Temporarily replace every LoRA sidecar by its frozen base linear."""

    def __init__(self, jev: JevMLX):
        self.jev, self.saved = jev, []

    def __enter__(self):
        layers = self.jev.model.model.layers
        self.saved = [(i, p, leaf, module) for i, p, leaf, module in _sidecars(self.jev)]
        for i, parent_name, leaf, module in self.saved:
            setattr(getattr(layers[i], parent_name), leaf, module.linear)

    def __exit__(self, *exc):
        layers = self.jev.model.model.layers
        for i, parent_name, leaf, module in self.saved:
            setattr(getattr(layers[i], parent_name), leaf, module)


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="bonsai-4b")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=SAMPLE_SIZE, help="rows per task (smoke tests only)")
    parser.add_argument("--task", help="run one task by name")
    parser.add_argument("--reference", type=Path, default=Path("results/artifacts/public_bench/qwen"))
    parser.add_argument("--memory-limit-gib", type=float, default=8.0)
    args = parser.parse_args()

    def guard() -> None:
        while True:
            if mx.get_active_memory() > args.memory_limit_gib * 2**30:
                print(f"[guard] MLX active memory above {args.memory_limit_gib} GiB; aborting", flush=True)
                os._exit(3)
            time.sleep(0.05)

    threading.Thread(target=guard, daemon=True).start()
    jev = JevMLX(args.adapter, backbone=args.backbone)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, slug, hf_args, split, build, kind in TASKS:
        if args.task and args.task != name:
            continue
        dataset = load_dataset(*hf_args, split=split).shuffle(seed=SEED).select(range(args.limit))
        items = [build(row) for row in dataset]
        reference = [json.loads(line) for line in (args.reference / f"{slug}.predictions.jsonl").read_text().splitlines()][: args.limit]
        same_rows = [g for _, _, g in items] == [r["gold"] for r in reference]
        rows, start = [], time.perf_counter()
        for index, (state, branch, gold) in enumerate(items):
            rows.append({"index": index, "gold": gold, "lora": jev.predict_request(state, [branch])[0]})
        with Bypass(jev):
            for row, (state, branch, _) in zip(rows, items):
                row["base"] = jev.predict_request(state, [branch])[0]
        rng = random.Random(SEED)
        out = {"task": name, "kind": kind, "n": len(rows), "options": len(rows[0]["lora"]), "same_rows_as_qwen_reference": same_rows}
        correct = {}
        for system in ("base", "lora"):
            correct[system] = [float(argmax(r[system]) == r["gold"]) for r in rows]
            out[f"{system}_accuracy"] = round(sum(correct[system]) / len(rows), 4)
            out[f"{system}_accuracy_ci95"] = bootstrap_ci(correct[system], rng)
            if kind == "score":
                out[f"{system}_mae"] = round(sum(abs(argmax(r[system]) - r["gold"]) for r in rows) / len(rows), 4)
        delta = [a - b for a, b in zip(correct["lora"], correct["base"])]
        out["delta_accuracy"] = round(sum(delta) / len(delta), 4)
        out["delta_accuracy_ci95"] = bootstrap_ci(delta, rng)
        out["seconds"] = round(time.perf_counter() - start, 1)
        out["peak_gib"] = round(mx.get_peak_memory() / 2**30, 2)
        summaries.append(out)
        (args.output / f"{slug}.predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps(out), flush=True)
    if not args.task:
        meta = {"backbone": args.backbone, "adapter": str(args.adapter), "sample_size": args.limit, "seed": SEED, "device": "mlx"}
        (args.output / "summary.json").write_text(json.dumps({"meta": meta, "results": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
