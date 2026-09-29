"""Evaluate an instruction-following causal LM by its answer-label probabilities.

Each branch is given through the model's chat template. Choice and Score options
are labelled A, B, ...; Noul is labelled A. No / B. Yes. At the assistant answer
position only those labels' next-token logits are normalized. This preserves the
answer format while avoiding continuation-length and option-wording bias.

This intentionally does not replace ``devset_backbone_probe.py``: that script is
the historical mean-option-continuation baseline.

Usage:
  python3 -m experiments.devset_label_probe --model Qwen/Qwen3-1.7B
  python3 -m experiments.devset_label_probe --dry-run
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = Path("data/devset/devset_b_v0.jsonl")
LABELS = tuple(chr(ord("A") + index) for index in range(20))


def prompt_for(row: dict, tokenizer) -> str:
    options = row["options"] if row["kind"] != "noul" else ["No", "Yes"]
    if len(options) > len(LABELS):
        raise ValueError(f"too many options: {len(options)}")
    labelled = "\n".join(f"{LABELS[index]}. {option}" for index, option in enumerate(options))
    user = (
        "Read the record and answer the question. Respond with exactly one listed "
        "letter and no other text.\n\n"
        f"Record:\n{row['state']}\n\nQuestion: {row['question']}\nAnswers:\n{labelled}\n\nAnswer:"
    )
    messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def label_token_ids(tokenizer, prompt: str, option_count: int) -> list[int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    ids = []
    for label in LABELS[:option_count]:
        continuation = tokenizer(prompt + label, add_special_tokens=False).input_ids[len(prompt_ids) :]
        if len(continuation) != 1:
            raise ValueError(f"answer label {label!r} is not one continuation token: {continuation}")
        ids.append(continuation[0])
    return ids


def probabilities(model, tokenizer, row: dict) -> list[float]:
    prompt = prompt_for(row, tokenizer)
    option_count = 2 if row["kind"] == "noul" else len(row["options"])
    tokens = label_token_ids(tokenizer, prompt, option_count)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, -1, tokens]
    return torch.softmax(logits, dim=-1).tolist()


def evaluate(model, tokenizer, rows: list[dict]) -> tuple[dict, list[dict]]:
    correct = defaultdict(int)
    total = defaultdict(int)
    predictions = defaultdict(Counter)
    outputs = []
    for number, row in enumerate(rows, 1):
        values = probabilities(model, tokenizer, row)
        prediction = max(range(len(values)), key=values.__getitem__)
        correct[row["kind"]] += prediction == row["label"]
        total[row["kind"]] += 1
        predictions[row["kind"]][prediction] += 1
        outputs.append(
            {
                "case_id": row["case_id"],
                "kind": row["kind"],
                "label": row["label"],
                "prediction": prediction,
                "probabilities": values,
            }
        )
        if number % 20 == 0:
            print(f"{number}/{len(rows)}", flush=True)
    quality = {
        kind: {
            "n": total[kind],
            "top1_accuracy": correct[kind] / total[kind],
            "prediction_counts": dict(sorted(predictions[kind].items())),
        }
        for kind in sorted(total)
    }
    return quality, outputs


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
    model.to(args.device).eval()
    quality, outputs = evaluate(model, tokenizer, rows)
    tag = args.model.rsplit("/", 1)[-1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{tag}_label_tokens.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    print(json.dumps({"model": args.model, "quality": quality, "outputs": str(output)}, indent=2))


if __name__ == "__main__":
    main()
