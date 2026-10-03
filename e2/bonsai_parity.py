"""Bit-exact check: unpacked FP16 Bonsai weights vs the dequantized MLX pack, on every LoRA target.

  python -m e2.bonsai_parity [--backbone bonsai-4b] [--out results/artifacts/bonsai_parity.json]

Needs `mlx` (CPU build is enough). Exit code 1 if any layer differs; then train on the dequantized
packed weights instead (`training_path = packed_dequantized`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

from e2.backbone import get_backbone

TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def load_tensors(directory: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    """Read only `wanted` tensors from every safetensors shard (numpy; bf16 not expected)."""
    found: dict[str, np.ndarray] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), framework="np") as f:
            for name in set(f.keys()) & wanted:
                found[name] = f.get_tensor(name)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="bonsai-4b")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    backbone = get_backbone(args.backbone)
    config = json.loads((Path(snapshot_download(backbone.infer_repo, revision=backbone.infer_revision, allow_patterns=["config.json"])) / "config.json").read_text())
    quantization = config["quantization"]
    unpacked_dir = Path(snapshot_download(backbone.train_repo, revision=backbone.train_revision, allow_patterns=["*.safetensors", "*.json"]))
    packed_dir = Path(snapshot_download(backbone.infer_repo, revision=backbone.infer_revision, allow_patterns=["*.safetensors", "*.json"]))

    modules = [f"model.layers.{i}.{t}" for i in range(config["num_hidden_layers"]) for t in TARGETS]
    unpacked = load_tensors(unpacked_dir, {f"{m}.weight" for m in modules})
    packed = load_tensors(packed_dir, {f"{m}.{p}" for m in modules for p in ("weight", "scales", "biases")})

    differing, rows = [], []
    for m in modules:
        w, s, b = (mx.array(packed[f"{m}.{p}"]) for p in ("weight", "scales", "biases"))
        dequantized = np.array(mx.dequantize(w, s, b, group_size=quantization["group_size"], bits=quantization["bits"]).astype(mx.float16))
        reference = unpacked[f"{m}.weight"].astype(np.float16)
        same = dequantized.shape == reference.shape and dequantized.tobytes() == reference.tobytes()
        numeric = dequantized.astype(np.float32) - reference.astype(np.float32)
        rows.append({"module": m, "bit_exact": bool(same), "differing_elements": int((numeric != 0).sum()), "max_abs_diff": float(np.abs(numeric).max())})
        if not same:
            differing.append(m)
    result = {
        "backbone": backbone.name, "targets": len(modules), "bit_exact": len(modules) - len(differing),
        "training_path": "unpacked" if not differing else "packed_dequantized", "differing": differing,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**result, "modules": rows}, indent=1) + "\n")
    print(json.dumps(result if len(differing) < 20 else {**result, "differing": differing[:20] + ["..."]}, indent=1))
    raise SystemExit(1 if differing else 0)


if __name__ == "__main__":
    main()
