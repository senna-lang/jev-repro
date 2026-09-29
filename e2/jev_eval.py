"""本家Jev（TypeSafe API）をE2の教師テスト（eval 1,200 branch）にかけ、教師分布との指標を出す。

学習前Qwen3-4B・選定LoRAと同じ `teacher_metrics` で集計する。1 state＝1 API call（設問は全部まとめる）。
Jevは選択肢記号を使わないので順番平均はしない（正準順の1回だけ）。
結果は比較のためだけに使い、checkpoint選定・学習には使わない。

使い方: set -a && . ./.env && set +a && python3 -m e2.jev_eval [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from e2.evaluate_lora import teacher_metrics
from e2.packed_data import load_requests

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
RETRY_STATUSES = {429, 529, 500, 502, 503, 504}


def build_questions(rows: list[dict]) -> dict:
    questions = {}
    for row in rows:
        if row["kind"] == "noul":
            questions[row["case_id"]] = {"type": "noul", "instructions": row["question"]}
        elif row["kind"] == "choice":
            if len(set(row["options"])) != len(row["options"]):
                raise ValueError(f"duplicate choice option: {row['case_id']}")
            questions[row["case_id"]] = {"type": "choice", "instructions": row["question"],
                                         "criteria": {option: None for option in row["options"]}}
        else:
            questions[row["case_id"]] = {"type": "score", "instructions": row["question"], "criteria": row["options"]}
    return questions


def to_distribution(row: dict, answer: dict) -> list[float]:
    """Jevの回答をmanifestの正準順の分布へ直す（Noulは[No, Yes]）。"""
    if row["kind"] == "noul":
        return [1.0 - answer["noul"], answer["noul"]]
    probs = answer["probabilities"]
    keys = row["options"] if row["kind"] == "choice" else [str(i) for i in range(len(row["options"]))]
    return [float(probs[key]) for key in keys]


def call(session: requests.Session, key: str, state: str, questions: dict) -> dict:
    for attempt in range(8):
        response = session.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=120,
                                json={"model": MODEL, "state": state, "questions": questions})
        if response.status_code not in RETRY_STATUSES:
            response.raise_for_status()
            return response.json()
        time.sleep(float(response.headers.get("retry-after", 2 ** attempt)))
    response.raise_for_status()
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=Path, default=Path("data/e2/teacher_branches.jsonl"))
    parser.add_argument("--states", type=Path, default=Path("data/e2/states.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path("results/e2_jev_eval_raw.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/e2_jev_eval.json"))
    parser.add_argument("--limit", type=int, default=None, help="先頭N stateだけ（smoke用）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    key = os.environ["JEV_API_KEY"]

    groups = [(rows, request.state) for rows, request in load_requests(args.branches, args.states) if rows[0]["split"] == "eval"]
    groups = groups[: args.limit] if args.limit else groups
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.raw.exists():
        for line in args.raw.read_text().splitlines():
            record = json.loads(line)
            done[record["state_key"]] = record
    todo = [(rows, state) for rows, state in groups if rows[0]["case_id"] not in done]

    def work(rows, state):
        session = requests.Session()
        return rows, call(session, key, state, build_questions(rows))

    with args.raw.open("a") as sink, ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(work, rows, state) for rows, state in todo]):
            rows, response = future.result()
            record = {"state_key": rows[0]["case_id"], "model": response["model"], "answers": response["answers"], "usage": response["usage"]}
            sink.write(json.dumps(record) + "\n"); sink.flush()
            done[record["state_key"]] = record

    pairs, input_tokens, models = [], 0, set()
    for rows, _ in groups:
        record = done[rows[0]["case_id"]]
        input_tokens += record["usage"]["input_tokens"]; models.add(record["model"])
        for row in rows:
            pairs.append((row["kind"], to_distribution(row, record["answers"][row["case_id"]]), row["probabilities"]))
    report = {"models": sorted(models), "states": len(groups), "input_tokens": input_tokens,
              "cost_usd": input_tokens * 0.042 / 1e6, "metrics": teacher_metrics(pairs)}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
