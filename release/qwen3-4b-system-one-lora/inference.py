"""Standalone canonical-order inference for the Qwen3-4B System One LoRA adapter."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from packing import Branch, LETTERS, PackedRequest, pack_request, position_ids as build_position_ids

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def load_model(adapter: str | Path = ".", device: str | None = None):
    """Download the base model as needed, then load the adapter from this directory."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(adapter)).to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def predict_request(model, tokenizer, request: PackedRequest) -> list[list[float]]:
    """Return one probability vector per branch, in the original option order.

    A noul branch uses [No, Yes]. All branches share the chat prompt's common
    prefix; each branch sees only that prefix and its own suffix.
    """
    ids, state_len, spans = pack_request(request, tokenizer)

    branch_ids = torch.full((len(ids),), -1, dtype=torch.long)
    for index, (start, end) in enumerate(spans):
        branch_ids[start:end + 1] = index
    t = len(ids)
    mask = torch.tril(torch.ones((t, t), dtype=torch.bool)) & (
        (branch_ids == -1).unsqueeze(0) | (branch_ids[:, None] == branch_ids[None, :])
    )
    device = next(model.parameters()).device
    decisions = torch.tensor([end for _, end in spans], device=device)
    logits = model(
        input_ids=torch.tensor([ids], device=device),
        attention_mask=mask.to(device)[None, None],
        position_ids=torch.tensor([build_position_ids(state_len, spans)], device=device),
        logits_to_keep=decisions,
        use_cache=False,
    ).logits[0]
    counts = [len(branch.options or ["No", "Yes"]) for branch in request.branches]
    label_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS[:max(counts)]]
    if any(len(token) != 1 for token in label_ids):
        raise ValueError("answer labels must each be one token")
    labels = torch.tensor([token[0] for token in label_ids], device=device)
    return [torch.softmax(row[labels[:count]].float(), dim=-1).tolist() for row, count in zip(logits, counts)]


if __name__ == "__main__":
    model, tokenizer = load_model(Path(__file__).resolve().parent)
    request = PackedRequest(
        state="A customer received a replacement device after the first one stopped charging.",
        branches=[
            Branch("choice", "Which issue is described?", ["Delivery delay", "Charging failure", "Billing error"]),
            Branch("noul", "Does the record mention a replacement?"),
        ],
    )
    for branch, probabilities in zip(request.branches, predict_request(model, tokenizer, request)):
        print(branch.kind, probabilities)
