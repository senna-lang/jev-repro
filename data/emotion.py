"""dair-ai/emotionを読み込み、感情分類をChoice branchの`PackedRequest`へ変換する。

公式Jev比較例の`tone`（calm/frustrated/angry）は態度・感情の判定であり、既存の訓練データ
（AG Newsのニューストピック、banking77の銀行intent）には態度判定タスクが一切なかった
（実測: `tone`問題は不一致）。emotionはこのタスク型を直接埋める。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample, pick_instruction
from model.packer import Branch, PackedRequest

INSTRUCTION_VARIANTS = [
    "What emotion is being expressed in this message? Choose the single best answer.",
    "Identify the dominant emotion conveyed by the text below.",
    "Which emotion best describes how the writer of this message feels?",
]

MIN_K = 3
MAX_K = 6


def load_emotion(split: str = "train") -> Dataset:
    """HF `datasets`経由でdair-ai/emotionをロードする。"""
    return load_dataset("dair-ai/emotion", split=split)


def to_packed_example(
    text: str, label: int, label_names: list[str], rng: random.Random
) -> ChoiceExample:
    """1件（本文＋正解ラベル）を、正解1つ＋ランダムdistractor（K可変）のPackedRequestに変換する。"""
    k = rng.randint(MIN_K, MAX_K)
    other_indices = [i for i in range(len(label_names)) if i != label]
    distractors = rng.sample(other_indices, min(k - 1, len(other_indices)))

    option_indices = distractors + [label]
    rng.shuffle(option_indices)
    options = [label_names[i] for i in option_indices]
    correct_pos = option_indices.index(label)

    instructions = pick_instruction(INSTRUCTION_VARIANTS, rng)
    branch = Branch(kind="choice", instructions=instructions, options=options)
    request = PackedRequest(state=text, branches=[branch])
    return ChoiceExample(request=request, label=correct_pos)


def _smoke_check() -> None:
    from model.packer import load_packer_tokenizer, pack

    dataset = load_emotion("train")
    label_names = dataset.features["label"].names
    print(f"labels: {label_names}")
    assert label_names == ["sadness", "joy", "love", "anger", "fear", "surprise"]

    rng = random.Random(0)
    example = dataset[0]
    packed_example = to_packed_example(example["text"], example["label"], label_names, rng)

    print(f"state: {packed_example.request.state!r}")
    branch = packed_example.request.branches[0]
    print(f"instructions: {branch.instructions!r}")
    print(f"options ({len(branch.options)}): {branch.options}")
    print(f"label: {packed_example.label}")

    assert MIN_K <= len(branch.options) <= MAX_K
    assert 0 <= packed_example.label < len(branch.options)
    assert branch.options[packed_example.label] == label_names[example["label"]]
    assert branch.instructions in INSTRUCTION_VARIANTS

    tokenizer = load_packer_tokenizer()
    packed = pack(packed_example.request, tokenizer)
    assert len(packed.branch_spans[0].option_end_indices) == len(branch.options)

    print("OK: emotionの1件がK可変(3-6)・instructions可変のChoice PackedRequestに変換でき、"
          "packerでトークン化までできることを確認。")


if __name__ == "__main__":
    _smoke_check()
