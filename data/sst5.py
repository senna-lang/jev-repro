"""SST-5（Stanford Sentiment Treebank, 5段階）を読み込み、Score branchの`PackedRequest`へ変換する。

公式Jevの`severity`/`frustration`のようなScore（順序尺度）タスクは、`NoulReadout`と同じく
訓練データが存在しなかった（`Choice`用のAG News/banking77/emotion/dbpedia_14はいずれも
名義尺度）。SST-5のラベルは`very negative`(0) < `negative`(1) < `neutral`(2) < `positive`(3)
< `very positive`(4)という自然な順序を持つため、選択肢の並び順をレベル順に固定すれば
そのままScoreの教師データになる（`data/common.py`の`reshuffled`はScore branchを
シャッフルしないため、この順序は訓練中も保たれる）。

レベル数Kは固定せず、正解を含む長さKの連続部分区間をサンプルする。5レベル固定で訓練した
readoutは、同じドメイン・同じテキストでもK=2/3/4ではKendall tauがほぼ0まで落ちることを
`experiments/score_diagnose.py`で確認した（K=5のみtau=+0.107, p=0.04、他はp>0.14）。
つまり「順序尺度上での比較」ではなく「5択レイアウト固有の手掛かり」を覚えてしまっていた。
公式JevのScoreはK=3（severity/frustration）やK=4（report_quality）で使われるので、
訓練時にKを振ってK不変な比較を学ばせる必要がある。

連続区間に限るのは、順序尺度としての意味を壊さないため（`negative`と`very positive`だけを
並べた飛び飛びの選択肢は「隣接レベル」の距離が均一でなくなり、`score_expectation`の
線形な期待値計算と整合しない）。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample, pick_instruction
from model.packer import Branch, PackedRequest

INSTRUCTION_VARIANTS = [
    "Rate the sentiment of this movie review on a scale from very negative to very positive.",
    "How positive or negative is this review? Pick the level that best matches.",
    "Judge the overall sentiment expressed in the review below.",
]

# ラベルindexの昇順がそのままレベル順（0=very negative .. 4=very positive）。
LEVEL_ORDER = ["very negative", "negative", "neutral", "positive", "very positive"]


def load_sst5(split: str = "train") -> Dataset:
    """HF `datasets`経由でSST-5をロードする。"""
    return load_dataset("SetFit/sst5", split=split)


def to_packed_example(
    text: str,
    label: int,
    rng: random.Random,
    k_range: tuple[int, int] = (2, 5),
) -> ChoiceExample:
    """1件（本文＋正解ラベル0-4）を、レベル順を保ったままK可変のScore PackedRequestに変換する。

    `k_range`から選んだKに対し、正解ラベルを含む長さKの連続区間を一様サンプルし、
    その区間内での相対位置を新しいラベルにする。`k_range=(5, 5)`で全5レベル固定になる。
    """
    low, high = k_range
    k = rng.randint(low, high)
    # 正解を含む長さkの窓 [start, start+k) の取りうる範囲。
    start = rng.randint(max(0, label - k + 1), min(label, len(LEVEL_ORDER) - k))
    options = LEVEL_ORDER[start:start + k]

    instructions = pick_instruction(INSTRUCTION_VARIANTS, rng)
    branch = Branch(kind="score", instructions=instructions, options=options)
    request = PackedRequest(state=text, branches=[branch])
    return ChoiceExample(request=request, label=label - start)


def _smoke_check() -> None:
    from model.packer import load_packer_tokenizer, pack

    dataset = load_sst5("train")
    rng = random.Random(0)
    example = dataset[0]
    assert dataset.features["label_text"] is not None

    packed_example = to_packed_example(example["text"], example["label"], rng)
    branch = packed_example.request.branches[0]

    print(f"state: {packed_example.request.state!r}")
    print(f"instructions: {branch.instructions!r}")
    print(f"options (レベル順を保った連続区間): {branch.options}")
    print(f"label: {packed_example.label} (元のレベル: {example['label_text']})")

    assert branch.kind == "score"
    assert branch.instructions in INSTRUCTION_VARIANTS

    # 100件サンプルして、K可変化の不変条件を確かめる。
    seen_k: set[int] = set()
    for row in dataset.select(range(100)):
        ex = to_packed_example(row["text"], row["label"], rng)
        options = ex.request.branches[0].options
        k = len(options)
        seen_k.add(k)

        assert 2 <= k <= 5, f"Kが範囲外: {k}"
        # 選択肢はLEVEL_ORDERの連続部分列（順序尺度としての等間隔性を保つ）
        start = LEVEL_ORDER.index(options[0])
        assert options == LEVEL_ORDER[start:start + k], f"連続区間でない: {options}"
        # ラベルは窓内の相対位置を指し、元のレベルのテキストと一致する
        assert 0 <= ex.label < k, f"ラベルが窓外: {ex.label} (K={k})"
        assert options[ex.label] == row["label_text"], (
            f"ラベルの付け替えがずれている: {options[ex.label]!r} != {row['label_text']!r}"
        )

    assert seen_k == {2, 3, 4, 5}, f"Kが振れていない: {sorted(seen_k)}"

    # k_range=(5, 5)で全5レベル固定（ablationの対照条件）になること
    fixed = to_packed_example(example["text"], example["label"], rng, k_range=(5, 5))
    assert fixed.request.branches[0].options == LEVEL_ORDER
    assert fixed.label == example["label"]

    tokenizer = load_packer_tokenizer()
    packed = pack(packed_example.request, tokenizer)
    assert len(packed.branch_spans[0].option_end_indices) == len(branch.options)

    print(f"OK: 100件でK={sorted(seen_k)}が振れ、選択肢は常にLEVEL_ORDERの連続区間、"
          "ラベルは窓内の相対位置に正しく付け替わり、packerでトークン化までできることを確認。")


if __name__ == "__main__":
    _smoke_check()
