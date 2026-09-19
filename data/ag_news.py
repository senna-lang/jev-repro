"""AG Newsを読み込み、記事分類をChoice branchの`PackedRequest`へ変換する。

各記事本文をstate、4クラスを選択肢、分類指示とクラス基準をbranchに設定する。
再現可能なshuffle後の訓練サブセットとheld-out評価splitも提供する。
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, load_dataset

from model.packer import Branch, PackedRequest

QUESTION = "この記事のトピックを、選択肢の中から1つ選べ。"

# criteria説明はAG Newsの一般的なクラス定義に基づく短い説明文。
CRITERIA = {
    "World": "国際情勢・外交・戦争・各国の政治など、特定の国に限らない一般的な国際ニュース",
    "Sports": "スポーツの試合結果・選手・大会・チームに関するニュース",
    "Business": "企業・経済・金融・株式市場・貿易に関するニュース",
    "Sci/Tech": "科学研究・技術・IT・宇宙開発に関するニュース",
}


def _format_option(label: str) -> str:
    """選択肢の文字列を`ラベル — criteria`の形にする。

    `Branch.options`はpacker側ではただの文字列リストなので、ラベルとcriteria説明を
    1つの文字列に畳み込む（Jevの実際のAPIがラベルとcriteriaを別フィールドで送るとしても、
    packerが受け取るのは最終的に1つのテキストなので、この畳み込みで表現として等価）。
    """
    return f"{label} — {CRITERIA[label]}"


@dataclass
class AGNewsExample:
    """1記事分の`PackedRequest`と正解ラベルのindex。

    `label`はAG Newsの元のインデックスを使う
    （0=World、1=Sports、2=Business、3=Sci/Tech）。正解ラベルは訓練と評価に使う。
    """

    request: PackedRequest
    label: int


def load_ag_news(split: str = "train") -> Dataset:
    """HF `datasets`経由でAG Newsをロードする。"""
    return load_dataset("ag_news", split=split)


# データ量別の実験で使う訓練サブセットのサイズ。
TRAIN_SUBSET_SIZES = [500, 2000, 10000, 20000]
EVAL_SIZE = 2000
SHUFFLE_SEED = 0  # 再現性のため固定


def train_subset(size: int, seed: int = SHUFFLE_SEED) -> Dataset:
    """train splitから、クラスに偏りなくsize件を切り出す。

    AG Newsのtrain splitは元々クラスごとに固まって並んでいる（実測：先頭500件中370件がSci/Techに
    偏る一方、全体では4クラスとも30,000件で均等）。そのまま`select(range(size))`すると偏った
    サブセットになるため、`shuffle`してから先頭size件を取る。

    同じ`seed`でshuffleした同一の並び替えから先頭size件を取るだけなので、`train_subset(500)`は
    `train_subset(2000)`の先頭500件と一致する（サイズを変えても既存サブセットが入れ子になる）。
    """
    return load_ag_news("train").shuffle(seed=seed).select(range(size))


def eval_split(size: int = EVAL_SIZE, seed: int = SHUFFLE_SEED) -> Dataset:
    """test splitから、held-out評価用にsize件を切り出す（train_subsetと同じ理由でshuffleする）。"""
    return load_ag_news("test").shuffle(seed=seed).select(range(size))


def to_packed_example(text: str, label: int, label_names: list[str]) -> AGNewsExample:
    """1記事（本文＋正解ラベル）を、Jevリクエスト形式の`PackedRequest`に変換する。"""
    options = [_format_option(name) for name in label_names]
    branch = Branch(kind="choice", instructions=QUESTION, options=options)
    request = PackedRequest(state=text, branches=[branch])
    return AGNewsExample(request=request, label=label)


def _smoke_check() -> None:
    from model.packer import load_packer_tokenizer, pack

    dataset = load_ag_news("train")
    label_names = dataset.features["label"].names
    print(f"label_names: {label_names}")
    assert label_names == ["World", "Sports", "Business", "Sci/Tech"], "想定と異なるラベル名"

    for name in label_names:
        assert name in CRITERIA, f"{name}のcriteria説明が用意されていない"

    example = dataset[0]
    packed_example = to_packed_example(example["text"], example["label"], label_names)

    print(f"state (記事本文の先頭80字): {packed_example.request.state[:80]!r}")
    print(f"question: {packed_example.request.branches[0].instructions}")
    print(f"options: {packed_example.request.branches[0].options}")
    print(f"label: {packed_example.label} ({label_names[packed_example.label]})")

    assert len(packed_example.request.branches) == 1
    assert packed_example.request.branches[0].kind == "choice"
    assert len(packed_example.request.branches[0].options) == 4
    assert 0 <= packed_example.label < 4

    # 実際にpackerでトークン化まで通ることを確認（Jev形式として最終的に使える状態か）
    tokenizer = load_packer_tokenizer()
    packed = pack(packed_example.request, tokenizer)
    print(f"packされたトークン数: {len(packed.input_ids)}")
    assert len(packed.branch_spans) == 1
    assert len(packed.branch_spans[0].option_end_indices) == 4

    print("OK: AG Newsの1件がstate＋Choice(4択, criteria付き)のPackedRequestに変換でき、"
          "実際にpackerでトークン化までできることを確認。")

    print(f"\n=== train_subset / eval_split ({TRAIN_SUBSET_SIZES}, eval={EVAL_SIZE}) ===")
    from collections import Counter

    subset_2000 = train_subset(2000)
    subset_500 = train_subset(500)
    assert len(subset_2000) == 2000
    assert len(subset_500) == 500

    counts_2000 = Counter(subset_2000["label"])
    print(f"train_subset(2000)のクラス分布: {dict(sorted(counts_2000.items()))}")
    # 完全に均等ではないが、偏りが極端（先頭500件で370/500がSci/Techのような状態）でないことを確認
    for label_idx, count in counts_2000.items():
        assert 300 < count < 800, (
            f"label={label_idx}の件数{count}が偏りすぎ。shuffleせずに先頭から切り出している可能性"
        )

    for size in TRAIN_SUBSET_SIZES:
        assert len(train_subset(size)) == size, f"train_subset({size})の件数が一致しない"

    # train_subset(500)がtrain_subset(2000)の先頭500件と一致すること（サイズ違いで入れ子）を確認
    assert subset_500["text"] == subset_2000["text"][:500], (
        "train_subset(500)がtrain_subset(2000)の先頭500件と一致しない（shuffleの再現性が壊れている）"
    )

    eval_examples = eval_split()
    assert len(eval_examples) == EVAL_SIZE
    counts_eval = Counter(eval_examples["label"])
    print(f"eval_split()のクラス分布: {dict(sorted(counts_eval.items()))}")
    for label_idx, count in counts_eval.items():
        assert 300 < count < 800, f"eval側もlabel={label_idx}の件数{count}が偏りすぎ"

    # eval用の1件もAGNewsExampleに変換できることを確認
    eval_label_names = eval_examples.features["label"].names
    eval_example0 = to_packed_example(eval_examples[0]["text"], eval_examples[0]["label"], eval_label_names)
    assert 0 <= eval_example0.label < 4

    print("OK: train_subsetは4サイズともクラス偏りなく切り出せ、サイズ違いでも入れ子になっている／"
          "eval_splitもheld-outとして偏りなく切り出せている。")


if __name__ == "__main__":
    _smoke_check()
