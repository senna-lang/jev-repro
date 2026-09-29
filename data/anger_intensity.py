"""怒りの強度（low/medium/high）を、Gold V1のfrustration Scoreと同じ構造に変換する。

`stepp1/tweet_emotion_intensity`（WASSA-2017 EmoInt、英語tweet）の`class == "anger"`行を使い、
`sentiment_intensity`（low/medium/high）を順序尺度のレベルにする。Gold V1のfrustration
（冷静→不満だが礼儀的→非常に怒っている、K=3）と同じ向き・同じKで、レベル1=最も穏やか。
選択肢はGold V1と同じく説明付きの文にする。Gold V1のデータ・文面は一切使わない（分離規則）。

train split=3960件（anger 941件: low 227 / medium 499 / high 215）。test splitは
`sentiment_intensity`が全てNoneのため使わない。train内でoffset分割する。
レベルごとに均等サンプルする。訓練データとして使う前に`experiments/support_task_probe.py`で
zero-shot信号を確認すること。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample, pick_instruction
from model.packer import Branch, PackedRequest

INSTRUCTION_VARIANTS = [
    "How angry does the writer of this message seem?",
    "Rate how strongly the author expresses anger.",
    "Judge the intensity of the anger in this message.",
]

# ラベルindexの昇順がそのままレベル順（0=最も弱い .. 2=最も強い）。
LEVEL_ORDER = [
    "slightly irritated, mostly calm",
    "clearly annoyed or angry",
    "furious, hostile, or aggressive",
]
_INTENSITY_TO_LABEL = {"low": 0, "medium": 1, "high": 2}


def load_anger_intensity_stratified(n_per_level: int, seed: int = 0, offset: int = 0) -> Dataset:
    """trainのanger行から、レベルごとに`[offset, offset+n_per_level)`を取る。

    test splitは強度ラベルが無い（全てNone）ので使わない。offsetをずらせばtrain/held-outが重複しない。
    """
    ds = load_dataset("stepp1/tweet_emotion_intensity", split="train")
    ds = ds.filter(lambda r: r["class"] == "anger").shuffle(seed=seed)
    selected: list[int] = []
    for level in _INTENSITY_TO_LABEL:
        indices = [i for i, value in enumerate(ds["sentiment_intensity"]) if value == level]
        taken = indices[offset : offset + n_per_level]
        if len(taken) != n_per_level:
            raise ValueError(f"{level}: n_per_level={n_per_level}, offset={offset}に対し{len(indices)}件しかない")
        selected.extend(taken)
    random.Random(seed).shuffle(selected)
    return ds.select(selected)


def to_packed_example(tweet: str, intensity: str, rng: random.Random) -> ChoiceExample:
    """1件（tweet＋強度）を、K=3固定・説明付きレベルのScore PackedRequestに変換する。"""
    branch = Branch(kind="score", instructions=pick_instruction(INSTRUCTION_VARIANTS, rng), options=list(LEVEL_ORDER))
    return ChoiceExample(request=PackedRequest(state=tweet, branches=[branch]), label=_INTENSITY_TO_LABEL[intensity])


def _smoke_check() -> None:
    from collections import Counter

    from model.packer import load_packer_tokenizer, pack

    tokenizer = load_packer_tokenizer()
    rng = random.Random(0)
    ds = load_anger_intensity_stratified(5, seed=0)
    assert Counter(ds["sentiment_intensity"]) == {"low": 5, "medium": 5, "high": 5}
    assert set(ds["class"]) == {"anger"}
    for row in ds:
        ex = to_packed_example(row["tweet"], row["sentiment_intensity"], rng)
        assert ex.request.branches[0].kind == "score"
        assert ex.label == _INTENSITY_TO_LABEL[row["sentiment_intensity"]]
        assert len(pack(ex.request, tokenizer).branch_spans[0].option_end_indices) == 3
    held_out = load_anger_intensity_stratified(5, seed=0, offset=5)
    assert not set(ds["id"]) & set(held_out["id"])
    print("OK: anger行のみ・レベル均等、K=3のScoreに変換しtokenize可、offset=5のheld-outとid重複0件。")


if __name__ == "__main__":
    _smoke_check()
