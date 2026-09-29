"""Gold V1（英語版）の最終試験。選定済みcheckpointを1回だけ評価し、報告にだけ使う。

学習・選定・ハイパーパラメータには使わない（ROADMAP / GOAL.md の Gold ルール）。
採点は既存の `evaluation.run.score_quality`（Choice/Score top-1、Noul threshold accuracy、Brier、NLL）。

使い方:
  GPU:  python -m e2.gold_eval local checkpoints/e2/lr-1e-4/epoch-3-quarter-3 results/
  API:  set -a && . ./.env && set +a && python3 -m e2.gold_eval jev results/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from evaluation.run import score_quality
from evaluation.schema import load_cases
from model.packer import Branch

GOLD = Path("evaluation/private/gold_v1_adjudicated_en_mt.jsonl")
NOUL_OPTIONS = ["No", "Yes"]


def _to_output(kind: str, p: list[float]) -> dict:
    return {"kind": "noul", "p_true": p[1]} if kind == "noul" else {"kind": kind, "probabilities": p}


def run_local(adapter: Path, out_dir: Path) -> None:
    import torch

    from e2.evaluate_lora import option_orders, predict_request
    from e2.train_lora import load_lora_model

    cases = load_cases(GOLD)
    device = torch.device("cuda")
    model, tokenizer = load_lora_model(adapter)
    model.to(device).eval()

    def predict() -> dict:
        outputs = {}
        for case in cases:
            branches = [Branch(b.kind, b.instructions, NOUL_OPTIONS if b.kind == "noul" else list(b.options)) for b in case.branches]
            orders = [option_orders(b.kind, len(b.options), f"gold:{case.id}/{src.id}") for b, src in zip(branches, case.branches)]
            for src, p in zip(case.branches, predict_request(model, tokenizer, case.state, branches, orders, device)):
                outputs[(case.id, src.id)] = _to_output(src.kind, p)
        return outputs

    results = {"selected": predict()}
    with model.disable_adapter():
        results["baseline"] = predict()
    for name, outputs in results.items():
        _write(out_dir, name, cases, outputs)


def run_jev(out_dir: Path) -> None:
    import requests

    from e2.jev_eval import MODEL, URL

    key = os.environ["JEV_API_KEY"]
    cases = load_cases(GOLD)
    outputs, tokens = {}, 0
    for case in cases:
        questions = {}
        for b in case.branches:
            if b.kind == "noul":
                questions[b.id] = {"type": "noul", "instructions": b.instructions}
            elif b.kind == "choice":
                questions[b.id] = {"type": "choice", "instructions": b.instructions, "criteria": {o: None for o in b.options}}
            else:
                questions[b.id] = {"type": "score", "instructions": b.instructions, "criteria": list(b.options)}
        response = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=120,
                                 json={"model": MODEL, "state": case.state, "questions": questions})
        response.raise_for_status()
        body = response.json(); tokens += body["usage"]["input_tokens"]
        for b in case.branches:
            answer = body["answers"][b.id]
            if b.kind == "noul":
                outputs[(case.id, b.id)] = {"kind": "noul", "p_true": float(answer["noul"])}
            else:
                keys = list(b.options) if b.kind == "choice" else [str(i) for i in range(len(b.options))]
                outputs[(case.id, b.id)] = {"kind": b.kind, "probabilities": [float(answer["probabilities"][k]) for k in keys]}
    print("jev input_tokens", tokens)
    _write(out_dir, "jev", cases, outputs)


def _write(out_dir: Path, name: str, cases, outputs: dict) -> None:
    quality = score_quality(cases, outputs)
    raw = {f"{c}/{b}": v for (c, b), v in outputs.items()}
    (out_dir / f"e2_gold_{name}.json").write_text(json.dumps({"quality": quality, "outputs": raw}, indent=2) + "\n")
    print(name, json.dumps(quality))


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "local":
        run_local(Path(rest[0]), Path(rest[1]))
    else:
        run_jev(Path(rest[0]))
