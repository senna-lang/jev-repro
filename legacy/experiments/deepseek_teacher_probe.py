"""Measure DeepSeek top-20 answer-label coverage on development set B.

The API returns at most 20 output-token logprobs after sampling temperature.
``temperature=1`` is used so the returned values are the model's native
relative logprobs. Labels absent from the top-20 have probability strictly
below ``exp(min_top_logprob)``.

Usage:
  set -a && . ./.env && set +a && python3 -m experiments.deepseek_teacher_probe --model deepseek-flash
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

DATASET = Path("data/devset/devset_b_v0.jsonl")
GROK = Path("data/devset/grok_devset_b_v0.jsonl")
LABELS = tuple(chr(ord("A") + index) for index in range(20))
API_URL = "https://api.deepseek.com/chat/completions"


def option_count(row: dict) -> int:
    return 2 if row["kind"] == "noul" else len(row["options"])


def prompt_for(row: dict) -> str:
    options = row["options"] if row["kind"] != "noul" else ["No", "Yes"]
    labelled = "\n".join(f"{LABELS[index]}. {option}" for index, option in enumerate(options))
    return (
        "Read the record and answer the question. Respond with exactly one listed "
        "letter and no other text.\n\n"
        f"Record:\n{row['state']}\n\nQuestion: {row['question']}\nAnswers:\n{labelled}\n\nAnswer:"
    )


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def request_choice(session: requests.Session, api_key: str, model: str, row: dict) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(row)}],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
        "thinking": {"type": "disabled"},
    }
    last_error = None
    for attempt in range(5):
        try:
            response = session.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            return response.json()["choices"][0]
        except (requests.RequestException, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed for {row['case_id']}: {last_error}") from last_error


def label_logprobs(alternatives: list[dict], allowed: tuple[str, ...]) -> dict[str, list[float]]:
    by_label = {label: [] for label in allowed}
    for alternative in alternatives:
        label = alternative["token"].strip()
        if label in by_label:
            by_label[label].append(float(alternative["logprob"]))
    return by_label


def grok_predictions(rows: list[dict]) -> dict[str, int]:
    wanted = {row["case_id"] for row in rows}
    by_case = {row["case_id"]: row for row in rows}
    predictions = {}
    for line in GROK.open():
        record = json.loads(line)
        if record["case_id"] not in wanted:
            continue
        row = by_case[record["case_id"]]
        answer = record["answer"]
        if row["kind"] == "choice":
            predictions[row["case_id"]] = row["options"].index(answer["choice"])
        elif row["kind"] == "score":
            predictions[row["case_id"]] = int(answer["score"])
        else:
            predictions[row["case_id"]] = int(answer["bool"])
    return predictions


def brier(probabilities: list[float], label: int) -> float:
    return sum((value - float(index == label)) ** 2 for index, value in enumerate(probabilities)) / len(probabilities)


def evaluate(model: str, rows: list[dict], api_key: str) -> tuple[dict, list[dict]]:
    grok = grok_predictions(rows)
    sums = defaultdict(lambda: {
        "n": 0,
        "correct": 0,
        "brier": 0.0,
        "nll": 0.0,
        "grok": 0,
        "complete": 0,
        "gold_missing": 0,
        "any_missing": 0,
        "max_missing_bound": 0.0,
    })
    predictions = defaultdict(Counter)
    outputs = []
    with requests.Session() as session:
        for number, row in enumerate(rows, 1):
            choice = request_choice(session, api_key, model, row)
            alternatives = choice["logprobs"]["content"][0]["top_logprobs"]
            observed = [float(item["logprob"]) for item in alternatives]
            min_top = min(observed)
            bound = math.exp(min_top)
            allowed = LABELS[: option_count(row)]
            by_label = label_logprobs(alternatives, allowed)
            missing = [label for label, values in by_label.items() if not values]
            complete = not missing
            gold_label = allowed[row["label"]]
            gold_missing = gold_label in missing
            values = None
            prediction = None
            if complete:
                label_lp = [logsumexp(by_label[label]) for label in allowed]
                normalizer = logsumexp(label_lp)
                values = [math.exp(value - normalizer) for value in label_lp]
                prediction = max(range(len(values)), key=values.__getitem__)
            total = sums[row["kind"]]
            total["n"] += 1
            total["any_missing"] += bool(missing)
            total["gold_missing"] += gold_missing
            total["max_missing_bound"] = max(total["max_missing_bound"], bound if missing else 0.0)
            if complete:
                total["complete"] += 1
                total["correct"] += prediction == row["label"]
                total["brier"] += brier(values, row["label"])
                total["nll"] -= math.log(values[row["label"]])
                total["grok"] += prediction == grok[row["case_id"]]
                predictions[row["kind"]][prediction] += 1
            outputs.append(
                {
                    "case_id": row["case_id"],
                    "kind": row["kind"],
                    "label": row["label"],
                    "generated": choice["message"]["content"],
                    "prediction": prediction,
                    "probabilities": values,
                    "missing_labels": missing,
                    "gold_missing": gold_missing,
                    "min_top_logprob": min_top,
                    "missing_prob_bound": bound if missing else None,
                    "complete": complete,
                }
            )
            if number % 20 == 0:
                print(f"{number}/{len(rows)}", flush=True)
    quality = {}
    for kind, total in sorted(sums.items()):
        complete = total["complete"]
        quality[kind] = {
            "n": total["n"],
            "complete": complete,
            "any_missing": total["any_missing"],
            "gold_missing": total["gold_missing"],
            "max_missing_prob_bound": total["max_missing_bound"],
            "top1_accuracy_complete": (total["correct"] / complete) if complete else None,
            "brier_complete": (total["brier"] / complete) if complete else None,
            "nll_complete": (total["nll"] / complete) if complete else None,
            "grok_top1_agreement_complete": (total["grok"] / complete) if complete else None,
            "prediction_counts": dict(sorted(predictions[kind].items())),
        }
    return quality, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("deepseek-flash", "deepseek-v4-pro"), required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("data/devset/runs"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.open()]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.dry_run:
        print(json.dumps({"dataset": str(args.dataset), "rows": len(rows), "model": args.model, "temperature": 1.0}))
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; source .env before running")
    quality, outputs = evaluate(args.model, rows, api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.model}_label_tokens.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in outputs))
    print(json.dumps({"model": args.model, "temperature": 1.0, "quality": quality, "outputs": str(output)}, indent=2))


if __name__ == "__main__":
    main()
