"""Mac MLX latency and teacher-eval metrics for the E2 LoRA.

  python -m mac.bench latency [--limit 10] [--single]
  python -m mac.bench eval    [--limit 60] [--single] [--out results/artifacts/mac/eval_mlx_fp16.json]
  python -m mac.bench record  --out results/artifacts/regression/qwen_mlx_baseline.json
  python -m mac.bench compare --baseline results/artifacts/regression/qwen_mlx_baseline.json

Uses the teacher eval split (data/e2). Fixed-candidate evaluation only; never for selection, never Gold V1.
Default is order averaging (e2.evaluate_lora.option_orders); --single uses the canonical order only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.tuner.lora import LoRALinear

from e2.backbone import get_backbone
from e2.bonsai_parity import TARGETS
from e2.chat_packer import pack_chat_request
from e2.evaluate_lora import compare_snapshots, option_orders, teacher_metrics
from e2.evaluate_lora import regression_snapshot as base_snapshot
from e2.packed_data import load_requests
from mac.jev_mlx import DEFAULT_ADAPTER, JevMLX
from model.packer import Branch, PackedRequest


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


def regression_snapshot(jev: JevMLX) -> dict:
    adapter = json.loads((DEFAULT_ADAPTER / "adapter_config.json").read_text())
    q_proj = jev.model.model.layers[0].self_attn.q_proj.weight
    record = {"r": adapter["r"], "lora_alpha": adapter["lora_alpha"], "layer0_q_proj_sha256": hashlib.sha256(np.array(q_proj.view(mx.uint16)).tobytes()).hexdigest()}
    return base_snapshot(jev.tokenizer, jev.predict_request, record)


def compare(jev: JevMLX, baseline_path: Path, tolerance: float = 1e-4) -> dict:
    return compare_snapshots(json.loads(baseline_path.read_text()), regression_snapshot(jev), tolerance)


def _sha(array: mx.array) -> str:
    return hashlib.sha256(np.array(array).tobytes()).hexdigest()


def _sidecars(jev: JevMLX):
    for i, layer in enumerate(jev.model.model.layers):
        for parent_name, leaf in (t.split(".") for t in TARGETS):
            module = getattr(getattr(layer, parent_name), leaf)
            if isinstance(module, LoRALinear):
                yield i, parent_name, leaf, module


def eval_requests_for(split: str, state_id: str):
    for rows, request in load_requests(Path("data/e2/teacher_branches.jsonl"), Path("data/e2/states.jsonl")):
        if rows[0]["split"] == split and rows[0]["state_id"] == state_id:
            yield request.state, [Branch(b.kind, b.instructions, row["options"]) for row, b in zip(rows, request.branches)]


def parity(args) -> dict:
    """Spec 6.2 checks for a packed (Bonsai) backbone. Returns {check: {passed, ...}}; every check must pass."""
    from huggingface_hub import snapshot_download

    spec = get_backbone(args.backbone)
    jev = JevMLX(args.adapter, backbone=args.backbone)
    cuda = json.loads(args.cuda.read_text())["requests"][: args.limit]
    results: dict[str, dict] = {}

    token_mismatch, argmax_agree, branches, max_delta = [], 0, 0, 0.0
    for ref in cuda:
        state, branches_in = next(eval_requests_for("val", ref["state_id"]))
        packed = pack_chat_request(PackedRequest(state, branches_in), jev.tokenizer)
        if [int(t) for t in packed.input_ids] != ref["input_ids"]:
            token_mismatch.append(ref["state_id"])
        for mine, theirs in zip(jev.predict_request(state, branches_in), ref["probabilities"]):
            branches += 1
            argmax_agree += int(max(range(len(mine)), key=mine.__getitem__) == max(range(len(theirs)), key=theirs.__getitem__))
            max_delta = max(max_delta, max(abs(a - b) for a, b in zip(mine, theirs)))
    results["token_match"] = {"passed": not token_mismatch, "requests": len(cuda), "mismatched": token_mismatch}
    results["prediction_agreement"] = {
        "passed": argmax_agree / branches >= 0.99 and max_delta <= 0.03,
        "branches": branches, "argmax_agreement": argmax_agree / branches, "max_probability_delta": max_delta,
    }

    print(f"[mem] after predictions: active {mx.get_active_memory()/2**30:.2f} GiB peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
    sidecars = list(_sidecars(jev))
    non_fp32 = [(i, p, l) for i, p, l, m in sidecars if m.lora_a.dtype != mx.float32 or m.lora_b.dtype != mx.float32]
    results["sidecar_fp32"] = {"passed": len(sidecars) == 252 and not non_fp32, "sidecars": len(sidecars), "non_fp32": len(non_fp32)}

    disk = mx.load(str(Path(snapshot_download(spec.infer_repo, revision=spec.infer_revision, allow_patterns=["*.safetensors", "*.json"])) / "model.safetensors"))
    changed = []
    for i, parent_name, leaf, module in sidecars:
        for part, array in (("weight", module.linear.weight), ("scales", module.linear.scales), ("biases", module.linear.biases)):
            if _sha(array) != _sha(disk[f"model.layers.{i}.{parent_name}.{leaf}.{part}"]):
                changed.append(f"layers.{i}.{parent_name}.{leaf}.{part}")
    results["packed_unchanged"] = {"passed": not changed, "changed": changed[:5], "changed_count": len(changed)}

    del disk, sidecars, jev
    mx.clear_cache()
    if args.zero_adapter:
        zero = JevMLX(args.zero_adapter, backbone=args.backbone)
        state, branches_in = next(eval_requests_for("val", cuda[0]["state_id"]))
        with_sidecars = zero.predict_request(state, branches_in)
        for i, parent_name, leaf, module in list(_sidecars(zero)):
            setattr(getattr(zero.model.model.layers[i], parent_name), leaf, module.linear)
        results["zero_init_bit_exact"] = {"passed": with_sidecars == zero.predict_request(state, branches_in)}

    results["adapter_validation"] = _validation_checks(args)
    return results


def _validation_checks(args) -> dict:
    """Each tampered copy of the adapter directory must be rejected before any weights are loaded."""
    import shutil
    import tempfile

    def rejected(mutate, expected) -> bool:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("adapter_config.json", "bonsai_base.json"):
                shutil.copy(args.adapter / name, directory / name)
            (directory / "adapter_model.safetensors").symlink_to((args.adapter / "adapter_model.safetensors").resolve())
            mutate(directory)
            try:
                JevMLX(directory, backbone=args.backbone)
            except expected:
                return True
            return False

    def edit(name, **changes):
        def apply(directory: Path) -> None:
            data = json.loads((directory / name).read_text())
            data.update(changes)
            (directory / name).write_text(json.dumps(data))
        return apply

    cases = {
        "missing_base_record": (lambda d: (d / "bonsai_base.json").unlink(), FileNotFoundError),
        "wrong_infer_revision": (edit("bonsai_base.json", infer_revision="0" * 40), ValueError),
        "wrong_train_repo": (edit("bonsai_base.json", train_repo="prism-ml/other"), ValueError),
        "unknown_training_path": (edit("bonsai_base.json", training_path="bogus"), ValueError),
        "wrong_rank": (edit("adapter_config.json", r=8), ValueError),
        "wrong_alpha": (edit("adapter_config.json", lora_alpha=16), ValueError),
        "dora": (edit("adapter_config.json", use_dora=True), ValueError),
        "missing_target": (edit("adapter_config.json", target_modules=["q_proj", "k_proj"]), ValueError),
    }
    outcome = {name: rejected(mutate, expected) for name, (mutate, expected) in cases.items()}
    return {"passed": all(outcome.values()), **outcome}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["latency", "eval", "record", "compare", "parity"])
    parser.add_argument("--backbone", default="qwen")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--cuda", type=Path, help="parity: CUDA predictions from `e2.evaluate_lora predict`")
    parser.add_argument("--zero-adapter", type=Path, help="parity: B=0 adapter for the zero-init check")
    parser.add_argument("--limit", type=int, help="number of eval requests (default: 10 latency / all eval)")
    parser.add_argument("--single", action="store_true", help="canonical order only (no order averaging)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.mode == "parity":
        import os
        import threading

        def guard(limit_gib: float = 12.0) -> None:
            while True:
                if mx.get_active_memory() > limit_gib * 2**30:
                    print(f"[guard] MLX active memory above {limit_gib} GiB; aborting", flush=True)
                    os._exit(3)
                time.sleep(0.05)

        threading.Thread(target=guard, daemon=True).start()
        results = parity(args)
        print(json.dumps(results, indent=1))
        raise SystemExit(0 if all(r["passed"] for r in results.values()) else 1)
    jev = JevMLX(args.adapter, backbone=args.backbone)
    if args.mode == "record":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(regression_snapshot(jev)) + "\n")
        print(f"recorded {args.out}")
        return
    if args.mode == "compare":
        result = compare(jev, args.baseline)
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if result["passed"] else 1)
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
