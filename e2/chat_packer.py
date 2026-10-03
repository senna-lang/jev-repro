"""E2の公式chat template入力をstate共有tree入力へpackする。"""
from __future__ import annotations

from model.packer import Branch, BranchSpan, PackedRequest, PackedSequence


LETTERS = "ABCDEFGHIJKLMNOPQRST"


def scoring_text(state: str, branch: Branch) -> str:
    """教師採点と同じ、user message内の本文を作る。"""
    options = branch.options or ["No", "Yes"]
    answers = "\n".join(f"{LETTERS[index]}. {option}" for index, option in enumerate(options))
    return (
        "Read the record and answer the question. Respond with exactly one listed letter and no other text.\n\n"
        f"Record:\n{state}\n\nQuestion: {branch.instructions}\nAnswers:\n{answers}\n\nAnswer:"
    )


def chat_prompt_ids(tokenizer, state: str, branch: Branch) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": scoring_text(state, branch)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = rendered.input_ids if hasattr(rendered, "input_ids") else rendered
    return input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids

def _common_prefix_length(sequences: list[list[int]]) -> int:
    if not sequences:
        raise ValueError("at least one branch is required")
    for index, tokens in enumerate(zip(*sequences)):
        if len(set(tokens)) != 1:
            return index
    return min(map(len, sequences))


def pack_chat_request(request: PackedRequest, tokenizer) -> PackedSequence:
    """単独chat-template promptと完全に同じtoken列を共有prefix + suffixへ変換する。"""
    prompts = [chat_prompt_ids(tokenizer, request.state, branch) for branch in request.branches]
    # A single branch shares its whole prompt with itself; keep at least one suffix token per branch.
    state_len = min(_common_prefix_length(prompts), min(map(len, prompts)) - 1)
    if state_len <= 0:
        raise ValueError("chat template prompts do not have a usable shared state prefix")

    input_ids = prompts[0][:state_len]
    spans = []
    for prompt, branch in zip(prompts, request.branches):
        start = len(input_ids)
        input_ids.extend(prompt[state_len:])
        spans.append(BranchSpan(
            kind=branch.kind,
            start=start,
            option_start_indices=[],
            option_end_indices=[],
            decision_index=len(input_ids) - 1,
        ))
    return PackedSequence(input_ids=input_ids, state_len=state_len, branch_spans=spans)
