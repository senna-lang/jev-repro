"""banking77を読み込み、意図分類をChoice branchの`PackedRequest`へ変換する。

AG News（ニュース記事4クラス・固定）とはドメイン（銀行サポートチケット）も選択肢数K
（正解1つ＋ランダムなdistractor、可変K）も異なる。目的はreadoutが「AG Newsの4文字列
パターンマッチ」に過学習するのを防ぎ、state・選択肢テキストの意味的マッチングを学習する
圧力をかけること。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample
from model.packer import Branch, PackedRequest

QUESTION = "How should this customer support message be classified? Choose the single best intent."

# インテント数77は1 branchの選択肢としては非現実的に多い（公式Jevの実例も選択肢は2〜5個程度）ため、
# 正解1つ＋distractorをランダムサンプルしてKを可変にする。
MIN_K = 3
MAX_K = 6


def _label_text(name: str) -> str:
    """snake_caseのラベル名を読みやすいフレーズに変換する（例: `card_arrival` -> `card arrival`）。"""
    return name.replace("_", " ")


def load_banking77(split: str = "train") -> Dataset:
    """HF `datasets`経由でbanking77をロードする。"""
    return load_dataset("legacy-datasets/banking77", split=split)


def to_packed_example(
    text: str, label: int, label_names: list[str], rng: random.Random
) -> ChoiceExample:
    """1件（本文＋正解ラベル）を、正解1つ＋ランダムdistractor（K可変）のPackedRequestに変換する。"""
    k = rng.randint(MIN_K, MAX_K)
    other_indices = [i for i in range(len(label_names)) if i != label]
    distractors = rng.sample(other_indices, min(k - 1, len(other_indices)))

    option_indices = distractors + [label]
    rng.shuffle(option_indices)
    options = [_label_text(label_names[i]) for i in option_indices]
    correct_pos = option_indices.index(label)

    branch = Branch(kind="choice", instructions=QUESTION, options=options)
    request = PackedRequest(state=text, branches=[branch])
    return ChoiceExample(request=request, label=correct_pos)


def _smoke_check() -> None:
    from model.packer import load_packer_tokenizer, pack

    dataset = load_banking77("train")
    label_names = dataset.features["label"].names
    print(f"num labels: {len(label_names)}")
    assert len(label_names) == 77

    rng = random.Random(0)
    example = dataset[0]
    packed_example = to_packed_example(example["text"], example["label"], label_names, rng)

    print(f"state: {packed_example.request.state!r}")
    print(f"options ({len(packed_example.request.branches[0].options)}): "
          f"{packed_example.request.branches[0].options}")
    print(f"label: {packed_example.label}")

    branch = packed_example.request.branches[0]
    assert MIN_K <= len(branch.options) <= MAX_K
    assert 0 <= packed_example.label < len(branch.options)
    assert branch.options[packed_example.label] == _label_text(label_names[example["label"]])

    tokenizer = load_packer_tokenizer()
    packed = pack(packed_example.request, tokenizer)
    assert len(packed.branch_spans[0].option_end_indices) == len(branch.options)

    print("OK: banking77の1件がK可変(3-6)のChoice PackedRequestに変換でき、"
          "packerでトークン化までできることを確認。")


if __name__ == "__main__":
    _smoke_check()
