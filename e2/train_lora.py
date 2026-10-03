"""E2 packed chat-template LoRA training.

One optimizer step accumulates 16 packed state rows (64 branches).  Rows are kept
separate because each has a distinct tree mask and sequence length.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2.backbone import BASE_RECORD, TRAINING_PATHS, Backbone, get_backbone, read_base_record, write_base_record
from e2.chat_packer import LETTERS, pack_chat_request
from e2.evaluate_lora import evaluate
from e2.packed_data import load_requests
from model.packer import Branch, PackedRequest
from model.tree_mask import build_position_ids, build_tree_attention_mask

ROWS_PER_STEP = 16


class NonFiniteError(RuntimeError):
    """Loss, logits or trainable parameters became NaN/inf; the run is invalid."""



@dataclass
class TrainingRow:
    request: PackedRequest
    probabilities: list[list[float]]


def permuted_row(rows: list[dict], request: PackedRequest, rng: random.Random) -> TrainingRow:
    """Randomize option order and carry each teacher distribution to that order."""
    branches: list[Branch] = []
    probabilities: list[list[float]] = []
    for row, branch in zip(rows, request.branches):
        order = list(range(len(row["options"])))
        if row["kind"] == "noul":
            if rng.random() < 0.5:
                order.reverse()
        else:
            rng.shuffle(order)
        options = [row["options"][index] for index in order]
        branches.append(Branch(kind=branch.kind, instructions=branch.instructions, options=options))
        probabilities.append([row["probabilities"][index] for index in order])
    return TrainingRow(PackedRequest(request.state, branches), probabilities)
def _label_token_ids(tokenizer, count: int, device: torch.device) -> torch.Tensor:
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS[:count]]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("answer labels must each be exactly one token")
    return torch.tensor([tokens[0] for tokens in ids], device=device)


def packed_kl_loss(model, tokenizer, row: TrainingRow, device: torch.device) -> torch.Tensor:
    """Mean KL(q || p) across a four-branch packed state row."""
    packed = pack_chat_request(row.request, tokenizer)
    decision_indices = torch.tensor(
        [span.decision_index for span in packed.branch_spans], device=device, dtype=torch.long
    )
    input_ids = torch.tensor([packed.input_ids], device=device, dtype=torch.long)
    position_ids = torch.tensor([build_position_ids(packed)], device=device, dtype=torch.long)
    attention_mask = build_tree_attention_mask(packed).to(device)[None, None]
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_to_keep=decision_indices,
        use_cache=False,
    )
    if not torch.isfinite(output.logits).all():
        raise NonFiniteError("non-finite logits")
    losses = []
    for logits, target in zip(output.logits[0], row.probabilities):
        labels = _label_token_ids(tokenizer, len(target), device)
        log_probs = F.log_softmax(logits[labels].float(), dim=-1)
        q = torch.tensor(target, device=device, dtype=torch.float32)
        losses.append(F.kl_div(log_probs, q, reduction="sum"))
    return torch.stack(losses).mean()


def _load_base(backbone: Backbone, training_path: str | None):
    base = AutoModelForCausalLM.from_pretrained(
        backbone.train_repo, revision=backbone.train_revision, dtype=backbone.train_dtype, attn_implementation="sdpa"
    )
    if training_path == "packed_dequantized":
        from e2.bonsai_parity import packed_dequantized_weights
        for name, array in packed_dequantized_weights(backbone).items():
            parameter = base.get_parameter(name)
            parameter.data.copy_(torch.from_numpy(array).to(parameter.dtype))
    elif training_path not in (None, "unpacked"):
        raise ValueError(f"unknown training_path {training_path!r}")
    return base


def load_lora_model(adapter: Path | None = None, backbone: str = "qwen", training_path: str | None = None):
    """Frozen base + PEFT LoRA. Non-Qwen backbones need `training_path` (or an adapter with `bonsai_base.json`)."""
    spec = get_backbone(backbone)
    if spec.name != "qwen":
        if adapter:
            recorded = read_base_record(adapter, spec)["training_path"]
            if training_path not in (None, recorded):
                raise ValueError(f"adapter was trained on {recorded}, not {training_path}")
            training_path = recorded
        if training_path is None:
            raise ValueError(f"{spec.name} needs training_path")
    else:
        if adapter and (adapter / BASE_RECORD).exists():
            raise ValueError(f"{adapter} was trained on a non-Qwen base ({BASE_RECORD} present)")
        training_path = None
    tokenizer = AutoTokenizer.from_pretrained(spec.train_repo, revision=spec.train_revision)
    base = _load_base(spec, training_path)
    base.config.use_cache = False
    base.gradient_checkpointing_enable({"use_reentrant": False})
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
        ))
    if any("lm_head" in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("lm_head must not be a LoRA target")
    wrong = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.dtype != torch.float32]
    if wrong or any(parameter.dtype != spec.train_dtype for parameter in base.parameters() if not parameter.requires_grad):
        raise RuntimeError(f"dtype contract violated: trainable non-fp32 {wrong[:3]}, or base not {spec.train_dtype}")
    model.training_path = training_path
    return model, tokenizer


def train_epoch(model, tokenizer, train_path: Path, states_path: Path, optimizer, scheduler, rng: random.Random, device, on_quarter):
    rows = [(rows, request) for rows, request in load_requests(train_path, states_path) if rows[0]["split"] == "train"]
    rng.shuffle(rows)
    model.train()
    total_loss = 0.0
    steps = 0
    quarter_rows = (len(rows) + 3) // 4
    optimizer.zero_grad(set_to_none=True)
    for index, (manifest_rows, request) in enumerate(rows, start=1):
        loss = packed_kl_loss(model, tokenizer, permuted_row(manifest_rows, request, rng), device)
        if not torch.isfinite(loss):
            raise NonFiniteError(f"non-finite loss at row {index}")
        (loss / ROWS_PER_STEP).backward()
        total_loss += loss.item()
        if index % ROWS_PER_STEP == 0 or index == len(rows):
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                raise NonFiniteError(f"non-finite parameter after step {steps + 1}")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        if index % quarter_rows == 0 or index == len(rows):
            on_quarter(index / len(rows))
            model.train()
    return {"train_kl": total_loss / len(rows), "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=Path, default=Path("data/e2/teacher_branches.jsonl"))
    parser.add_argument("--states", type=Path, default=Path("data/e2/states.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--backbone", default="qwen")
    parser.add_argument("--training-path", choices=TRAINING_PATHS)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, tokenizer = load_lora_model(args.resume, args.backbone, args.training_path)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0)
    train_rows = sum(1 for rows, _ in load_requests(args.branches, args.states) if rows[0]["split"] == "train")
    total_steps = args.epochs * ((train_rows + ROWS_PER_STEP - 1) // ROWS_PER_STEP)
    warmup_steps = max(1, round(total_steps * 0.03))
    scheduler = LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    start_epoch = 1
    best_nll = float("inf")
    stalled = 0
    if args.resume:
        resume_state = torch.load(args.resume / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            quarter = 0
            def validate(fraction):
                nonlocal quarter, best_nll, stalled
                quarter += 1
                metrics = evaluate(model, tokenizer, args.branches, args.states, "val", device)["all"]
                if not math.isfinite(metrics["nll"]):
                    raise NonFiniteError("non-finite val NLL")
                checkpoint = args.output / f"epoch-{epoch}-quarter-{quarter}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint)
                if args.backbone != "qwen":
                    write_base_record(checkpoint, get_backbone(args.backbone), model.training_path)
                torch.save(
                    {"epoch": epoch, "quarter": quarter, "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "seed": args.seed},
                    checkpoint / "optimizer.pt",
                )
                (checkpoint / "val_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
                stalled = stalled + 1 if metrics["nll"] >= best_nll else 0
                best_nll = min(best_nll, metrics["nll"])
                print(json.dumps({"epoch": epoch, "quarter": quarter, "fraction": fraction, "val": metrics}))
            metrics = train_epoch(model, tokenizer, args.branches, args.states, optimizer, scheduler, rng, device, validate)
            print(json.dumps({"epoch": epoch, **metrics}))
            if stalled >= 4:
                break
    except NonFiniteError as error:
        (args.output / "INVALID.json").write_text(json.dumps({"seed": args.seed, "reason": str(error)}) + "\n")
        print(f"INVALID run (seed {args.seed}): {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
