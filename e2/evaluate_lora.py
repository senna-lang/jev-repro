"""E2 teacher-distribution metrics with prescribed option-order averaging."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from e2.chat_packer import LETTERS, pack_chat_request
from e2.packed_data import load_requests
from model.packer import Branch, PackedRequest
from model.tree_mask import build_position_ids, build_tree_attention_mask


def option_orders(kind: str, option_count: int, key: str) -> list[list[int]]:
    """Canonical plus fixed permutations: Choice/Score 4 orders (fewer if K! < 4), Noul canonical and reverse."""
    canonical = list(range(option_count))
    if kind == "noul":
        return [canonical, list(reversed(canonical))]
    rng = random.Random(f"e2-eval-v1:{key}")
    result = [canonical]
    while len(result) < min(4, math.factorial(option_count)):
        candidate = canonical.copy(); rng.shuffle(candidate)
        if candidate not in result:
            result.append(candidate)
    return result


def _label_ids(tokenizer, count, device):
    tokens = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS[:count]]
    if any(len(token) != 1 for token in tokens):
        raise ValueError("answer labels must each be one token")
    return torch.tensor([token[0] for token in tokens], device=device)


@torch.inference_mode()
def predict_request(model, tokenizer, state: str, branches: list[Branch], orders: list[list[list[int]]], device) -> list[list[float]]:
    """Pack every branch under every order into one tree and return canonical-order averaged distributions."""
    expanded = [
        (index, order, Branch(branch.kind, branch.instructions, [branch.options[i] for i in order]))
        for index, (branch, branch_orders) in enumerate(zip(branches, orders))
        for order in branch_orders
    ]
    packed = pack_chat_request(PackedRequest(state, [branch for _, _, branch in expanded]), tokenizer)
    decisions = torch.tensor([span.decision_index for span in packed.branch_spans], device=device)
    output = model(
        input_ids=torch.tensor([packed.input_ids], device=device),
        attention_mask=build_tree_attention_mask(packed).to(device)[None, None],
        position_ids=torch.tensor([build_position_ids(packed)], device=device),
        logits_to_keep=decisions,
        use_cache=False,
    ).logits[0]
    predictions: list[list[list[float]]] = [[] for _ in branches]
    for logits, (index, order, _) in zip(output, expanded):
        labels = _label_ids(tokenizer, len(order), device)
        ordered = torch.softmax(logits[labels].float(), dim=-1).tolist()
        canonical = [0.0] * len(order)
        for position, original in enumerate(order):
            canonical[original] = ordered[position]
        predictions[index].append(canonical)
    return [[sum(values) / len(values) for values in zip(*per_order)] for per_order in predictions]


def teacher_metrics(pairs: list[tuple[str, list[float], list[float]]]) -> dict:
    """(kind, p, q) の列から teacher-distribution top-1, Brier, NLL, 10-bin ECE を全体とkind別に返す。"""
    totals = defaultdict(float); by_kind = defaultdict(lambda: defaultdict(float)); count = defaultdict(int)
    for kind, p, q in pairs:
        k = len(q)
        correct = int(max(range(k), key=p.__getitem__) == max(range(k), key=q.__getitem__))
        brier = sum((a - b) ** 2 for a, b in zip(p, q)) / k
        nll = -sum(a * math.log(max(b, 1e-12)) for a, b in zip(q, p))
        confidence = max(p)
        bucket = min(int(confidence * 10), 9)
        for target in (totals, by_kind[kind]):
            target["top1"] += correct; target["brier"] += brier; target["nll"] += nll
            target[f"ece_conf_{bucket}"] += confidence; target[f"ece_correct_{bucket}"] += correct
        count["all"] += 1; count[kind] += 1
    def finalize(values, n):
        result = {metric: values[metric] / n for metric in ("top1", "brier", "nll")}
        result["ece"] = sum(abs(values[f"ece_conf_{i}"] - values[f"ece_correct_{i}"]) for i in range(10)) / n
        result["n"] = n
        return result
    return {"all": finalize(totals, count["all"]), **{kind: finalize(by_kind[kind], count[kind]) for kind in by_kind}}


@torch.inference_mode()
def evaluate(model, tokenizer, branches_path: Path, states_path: Path, split: str, device) -> dict:
    """Return teacher-distribution top-1, Brier, NLL, and 10-bin ECE."""
    model.eval()
    pairs = []
    for rows, request in load_requests(branches_path, states_path):
        if rows[0]["split"] != split:
            continue
        branches = [Branch(branch.kind, branch.instructions, row["options"]) for row, branch in zip(rows, request.branches)]
        orders = [option_orders(row["kind"], len(row["options"]), row["case_id"]) for row in rows]
        for row, p in zip(rows, predict_request(model, tokenizer, request.state, branches, orders, device)):
            pairs.append((row["kind"], p, row["probabilities"]))
    return teacher_metrics(pairs)


def fixed_request() -> tuple[str, str, list[Branch]]:
    """First `val` request in canonical option order: the shared fixed request of the regression and footprint checks."""
    for rows, request in load_requests(Path("data/e2/teacher_branches.jsonl"), Path("data/e2/states.jsonl")):
        if rows[0]["split"] == "val":
            return rows[0]["state_id"], request.state, [Branch(b.kind, b.instructions, row["options"]) for row, b in zip(rows, request.branches)]
    raise ValueError("no val request")


def regression_snapshot(tokenizer, predict, adapter_record: dict) -> dict:
    """Backend-independent record of the fixed request. `predict(state, branches)` returns canonical single-order probabilities."""
    state_id, state, branches = fixed_request()
    packed = pack_chat_request(PackedRequest(state, branches), tokenizer)
    return {
        "state_id": state_id,
        "input_ids": [int(t) for t in packed.input_ids],
        "position_ids": [int(p) for p in build_position_ids(packed)],
        "decision_indices": [span.decision_index for span in packed.branch_spans],
        "probabilities": predict(state, branches),
        "adapter": adapter_record,
    }


def compare_snapshots(base: dict, now: dict, tolerance: float) -> dict:
    exact = {k: base[k] == now[k] for k in ("state_id", "input_ids", "position_ids", "decision_indices", "adapter")}
    max_delta = max(abs(a - b) for pa, pb in zip(base["probabilities"], now["probabilities"]) for a, b in zip(pa, pb))
    return {"exact_match": exact, "max_probability_delta": max_delta, "passed": all(exact.values()) and max_delta <= tolerance}


def main() -> None:
    """python -m e2.evaluate_lora {record|compare|val} --adapter DIR [--out FILE] [--baseline FILE]"""
    import argparse
    import json

    from e2.train_lora import load_lora_model

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["record", "compare", "val"])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--backbone", default="qwen")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("VCPU", "8")))
    device = torch.device("cuda")
    model, tokenizer = load_lora_model(args.adapter, args.backbone)
    model.to(device).eval()
    if args.mode == "val":
        result = evaluate(model, tokenizer, Path("data/e2/teacher_branches.jsonl"), Path("data/e2/states.jsonl"), "val", device)
    else:
        config = json.loads((args.adapter / "adapter_config.json").read_text())
        canonical = lambda state, branches: predict_request(model, tokenizer, state, branches, [[list(range(len(b.options)))] for b in branches], device)
        now = regression_snapshot(tokenizer, canonical, {"r": config["r"], "lora_alpha": config["lora_alpha"]})
        result = now if args.mode == "record" else compare_snapshots(json.loads(args.baseline.read_text()), now, args.tolerance)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_ids", "position_ids")}, indent=1))
    if args.mode == "compare":
        raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
