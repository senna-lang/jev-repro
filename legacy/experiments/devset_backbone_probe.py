"""Evaluate an instruction-following causal LM on dev set B without training.

The prompt uses the model's chat template. Every option is teacher-forced in
the numbered answer form it is asked to generate (``"1. option text"``), so
tokenizer boundaries and the answer format match.

Usage:
    python3 -m experiments.devset_backbone_probe --model Qwen/Qwen3-1.7B
    python3 -m experiments.devset_backbone_probe --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = Path("data/devset/devset_b_v0.jsonl")


def option_log_probs(model, tokenizer, prompt: str, options: list[str]) -> list[float]:
    """Mean log probability of each numbered answer continuation."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    device = model.device
    scores = []
    with torch.inference_mode():
        for option in options:
            ids = tokenizer(prompt + option, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            option_ids = ids[0, len(prompt_ids) :]
            if not len(option_ids):
                raise ValueError(f"empty option continuation: {option!r}")
            logits = model(ids).logits[0]
            start = len(prompt_ids) - 1
            token_log_probs = torch.log_softmax(logits[start : start + len(option_ids)], dim=-1)
            scores.append(sum(token_log_probs[i, token] for i, token in enumerate(option_ids)).item() / len(option_ids))
    return scores


def prompt_for(row: dict, tokenizer) -> str:
    options = "\n".join(f"{i + 1}. {option}" for i, option in enumerate(row["options"]))
    user = (
        "Read the record and answer the question using exactly one listed answer.\n\n"
        f"Record:\n{row['state']}\n\nQuestion: {row['question']}\n"
        f"Answers:\n{options}\n\nAnswer:"
    )
    messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def softmax(values: list[float]) -> list[float]:
    highest = max(values)
    weights = [math.exp(value - highest) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def evaluate(model, tokenizer, rows: list[dict]) -> tuple[dict, list[dict]]:
    correct = defaultdict(int)
    total = defaultdict(int)
    outputs = []
    for n, row in enumerate(rows, 1):
        scored_options = [f"{i + 1}. {option}" for i, option in enumerate(row["options"])]
        probabilities = softmax(option_log_probs(model, tokenizer, prompt_for(row, tokenizer), scored_options))
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        correct[row["kind"]] += prediction == row["label"]
        total[row["kind"]] += 1
        outputs.append({"case_id": row["case_id"], "kind": row["kind"], "label": row["label"], "prediction": prediction, "probabilities": probabilities})
        if n % 20 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    return {kind: {"n": total[kind], "top1_accuracy": correct[kind] / total[kind]} for kind in sorted(total)}, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("data/devset/runs"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.open()]
    if args.dry_run:
        assert len(rows) == 198
        assert {row["kind"] for row in rows} == {"choice", "score", "noul"}
        print(json.dumps({"dataset": str(args.dataset), "rows": len(rows), "model": args.model}))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(args.device)
    model.eval()
    quality, outputs = evaluate(model, tokenizer, rows)
    tag = args.model.rsplit("/", 1)[-1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{tag}_standalone.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    print(json.dumps({"model": args.model, "quality": quality, "outputs": str(path)}, indent=2))


if __name__ == "__main__":
    main()
