"""Base-model definitions shared by training, evaluation and MLX inference.

One place for repo names, pinned commits and the training dtype. `bonsai_base.json` records which
base an adapter was trained on so the MLX loader can refuse a mismatched pairing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Backbone:
    name: str
    train_repo: str
    train_revision: str
    infer_repo: str  # MLX pack; for the Qwen reference the same repo is merged at load time
    infer_revision: str | None
    train_dtype: torch.dtype
    io_layers: bool = False  # also check/overwrite embed_tokens (+ untied lm_head) with the packed dequantized values


BACKBONES: dict[str, Backbone] = {
    "qwen": Backbone(
        "qwen", "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554",
        "Qwen/Qwen3-4B-Instruct-2507", None, torch.bfloat16,
    ),
    "bonsai-4b": Backbone(
        "bonsai-4b",
        "prism-ml/Ternary-Bonsai-4B-unpacked", "4485fae7a00129467b9329b738110d88b2942a1a",
        "prism-ml/Ternary-Bonsai-4B-mlx-2bit", "e1374ad6bf9b1b56afd743936b8faa33c409a75f",
        torch.float16,
    ),
    "bonsai-8b": Backbone(
        "bonsai-8b",
        "prism-ml/Ternary-Bonsai-8B-unpacked", "ac20f03fc62e872399218b659c8e949dfca05769",
        "prism-ml/Ternary-Bonsai-8B-mlx-2bit", "9260b24298e4211e804663e9f519962cf59f34be",
        torch.float16, io_layers=True,
    ),
}

TRAINING_PATHS = ("unpacked", "packed_dequantized")
BASE_RECORD = "bonsai_base.json"


def get_backbone(name: str) -> Backbone:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name]


def write_base_record(directory: Path, backbone: Backbone, training_path: str) -> None:
    if training_path not in TRAINING_PATHS:
        raise ValueError(f"training_path must be one of {TRAINING_PATHS}")
    record = {
        "backbone": backbone.name,
        "train_repo": backbone.train_repo, "train_revision": backbone.train_revision,
        "infer_repo": backbone.infer_repo, "infer_revision": backbone.infer_revision,
        "training_path": training_path,
    }
    (directory / BASE_RECORD).write_text(json.dumps(record, indent=1) + "\n")


def read_base_record(directory: Path, backbone: Backbone) -> dict:
    """Load `bonsai_base.json` and require it to match `backbone`; raise otherwise."""
    path = directory / BASE_RECORD
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; this adapter was not trained on {backbone.name}")
    record = json.loads(path.read_text())
    expected = {
        "backbone": backbone.name,
        "train_repo": backbone.train_repo, "train_revision": backbone.train_revision,
        "infer_repo": backbone.infer_repo, "infer_revision": backbone.infer_revision,
    }
    mismatched = {k: (record.get(k), v) for k, v in expected.items() if record.get(k) != v}
    if mismatched:
        raise ValueError(f"{path} does not match {backbone.name}: {mismatched}")
    if record.get("training_path") not in TRAINING_PATHS:
        raise ValueError(f"{path}: training_path must be one of {TRAINING_PATHS}")
    return record
