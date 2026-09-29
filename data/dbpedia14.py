"""DBpedia-14を読み込み、記事分類をChoice branchの`PackedRequest`へ変換する。

AG News（ニュース4クラス）・banking77（銀行intent 77クラス）に、Wikipedia由来の
エンティティ分類14クラスを加える。目的は選択肢語彙の多様性を増やすこと——訓練済み
readout（`W_q,W_k`のΔの実効ランクが1.83だった実測）が特定データセットの少数の
文字列パターンに過学習するのを防ぐため、readoutが目にする選択肢テキストの種類自体を
増やす。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample, pick_instruction
from model.packer import Branch, PackedRequest

INSTRUCTION_VARIANTS = [
    "Which category best describes this Wikipedia article? Choose the single best answer.",
    "Classify the following article into the most appropriate category.",
    "What type of subject does this article describe?",
]

MIN_K = 4
MAX_K = 8


def load_dbpedia14(split: str = "train") -> Dataset:
    """HF `datasets`経由でDBpedia-14をロードする。"""
    return load_dataset("fancyzhx/dbpedia_14", split=split)


def to_packed_example(
    title: str, content: str, label: int, label_names: list[str], rng: random.Random
) -> ChoiceExample:
    """1件（タイトル＋本文＋正解ラベル）を、正解1つ＋ランダムdistractor（K可変）のPackedRequestに変換する。"""
    text = f"{title}\n\n{content}"

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

    dataset = load_dbpedia14("train")
    label_names = dataset.features["label"].names
    print(f"num labels: {len(label_names)}")
    assert len(label_names) == 14

    rng = random.Random(0)
    example = dataset[0]
    packed_example = to_packed_example(
        example["title"], example["content"], example["label"], label_names, rng
    )

    branch = packed_example.request.branches[0]
    print(f"state (先頭80字): {packed_example.request.state[:80]!r}")
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

    print("OK: DBpedia-14の1件がK可変(4-8)・instructions可変のChoice PackedRequestに変換でき、"
          "packerでトークン化までできることを確認。")


if __name__ == "__main__":
    _smoke_check()
