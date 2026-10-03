"""Shared request format and Qwen chat-template packing for PyTorch and MLX."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LETTERS = "ABCDEFGHIJKLMNOPQRST"


@dataclass
class Branch:
    kind: Literal["choice", "score", "noul"]
    instructions: str
    options: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind == "noul":
            if self.options and self.options != ["No", "Yes"]:
                raise ValueError("noul options must be empty or ['No', 'Yes']")
        elif self.kind in ("choice", "score"):
            if not 2 <= len(self.options) <= len(LETTERS):
                raise ValueError(f"{self.kind} requires 2–{len(LETTERS)} options")
        else:
            raise ValueError(f"unknown branch kind: {self.kind}")


@dataclass
class PackedRequest:
    state: str
    branches: list[Branch]


def pack_request(request: PackedRequest, tokenizer) -> tuple[list[int], int, list[tuple[int, int]]]:
    """Return input IDs, shared prefix length, and each branch's inclusive span."""
    if not request.branches:
        raise ValueError("at least one branch is required")
    prompts = []
    for branch in request.branches:
        options = branch.options or ["No", "Yes"]
        answers = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(options))
        text = (
            "Read the record and answer the question. Respond with exactly one listed letter and no other text.\n\n"
            f"Record:\n{request.state}\n\nQuestion: {branch.instructions}\nAnswers:\n{answers}\n\nAnswer:"
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        ids = rendered.input_ids if hasattr(rendered, "input_ids") else rendered
        prompts.append(ids[0] if ids and isinstance(ids[0], list) else ids)
    state_len = 0
    for tokens in zip(*prompts):
        if len(set(tokens)) != 1:
            break
        state_len += 1
    # A single question shares its whole prompt with itself; keep at least one suffix token per question.
    state_len = min(state_len, min(map(len, prompts)) - 1)
    if state_len <= 0:
        raise ValueError("chat template prompts do not have a usable shared state prefix")
    input_ids = prompts[0][:state_len]
    spans = []
    for prompt in prompts:
        start = len(input_ids)
        input_ids.extend(prompt[state_len:])
        spans.append((start, len(input_ids) - 1))
    return input_ids, state_len, spans


def position_ids(state_len: int, spans: list[tuple[int, int]]) -> list[int]:
    positions = list(range(state_len))
    for start, end in spans:
        positions.extend(range(state_len, state_len + end - start + 1))
    return positions
