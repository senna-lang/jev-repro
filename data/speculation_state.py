"""Shared state-string builder for the Jev speculation-gate dev set.

Mirrors exactly what life/.omp/hooks/post/jev-speculation-gate.ts feeds to
JevMLX.predict_request as `state`. Both the devset builder (data/speculation_devset.py)
and anything that consumes the resulting jsonl rows MUST use this function so the
`state` field always matches what the hook itself would construct.
"""
from __future__ import annotations


def build_state(text: str, tools: list[str]) -> str:
    """Build the Jev `state` input from a turn's final assistant text and tool names.

    `text` is the stripped text of the last assistant message in the turn that has
    non-empty text. `tools` is the unique list of tool names used in the turn, in
    first-seen order when scanning backwards from the end of the turn to the
    preceding user message.
    """
    tool_list = ", ".join(tools) if tools else "none"
    return f"Tools used this turn: {tool_list}\n\nAssistant final answer:\n{text[-6000:]}"
