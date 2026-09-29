"""カスタマーサポートチケットの`priority`ラベル（low/medium/high）を読み込み、
Score branchの`PackedRequest`へ変換する。

`EXP_HEAD_VS_BACKBONE.md` Gate 1-b: 公式Jevのseverity/frustrationはバグ報告・サポート
チケット風の重大度尺度だが、これまでの訓練データ（AG News/banking77/emotion/sst5/
dbpedia14）にはこのドメインが一切含まれていなかった。Gate 0で「severity_tableの順序信号は
凍結backboneに既に存在する」ことが確認されたため、不足しているのはreadoutの訓練データ分布。

`Tobi-Bueck/customer-support-tickets`（HF Hub, 61,765件, 独語/英語）を使用。英語のみに絞り、
`priority`が`low`/`medium`/`high`の3値（`critical`/`very_low`は英語サブセットに存在しない
ため対象外）。人手ラベルの実データであり、公式Jevの掲載値とは完全に独立（模倣ではない）。

severity_tableとは異なるドメイン（実際の企業サポートチケット、業務優先度ベースのラベル）
だが、公式Jevが訓練データを持たないのと同じ理由でこの近似を採用する: 完全に一致するpublicな
Jev訓練データは存在しない。`priority`は技術的深刻度そのものではなく業務優先度だが、
「対応の緊急さ」という共通の順序軸を持ち、severityの代理として妥当な範囲にある。

train/held-outの分割はこのモジュール内で完結させる（`load_dataset`のtrain splitのみ存在する
ため、`split="train"`をシャッフルしてから先頭をtrain、末尾をheld-outとして使う）。held-out側は
`EXP_HEAD_VS_BACKBONE.md` §3.2の定義する「out-of-domain」評価に使う。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample, pick_instruction
from model.packer import Branch, PackedRequest

INSTRUCTION_VARIANTS = [
    "How severe is the issue described in this support ticket?",
    "Rate the severity of this customer's problem.",
    "Judge how severe or urgent the issue in this ticket is.",
]

# ラベルindexの昇順がそのままレベル順（0=low .. 2=high）。公式severity（K=3）と同じ選択肢数。
LEVEL_ORDER = [
    "low; minor issue with little urgency",
    "medium; noticeable issue needing attention",
    "high; urgent issue blocking the customer",
]
_PRIORITY_TO_LABEL = {"low": 0, "medium": 1, "high": 2}

_MAX_BODY_CHARS = 600  # 長文チケットのトークン数を抑える（他データセットのstate長と揃える）


def _load_all(seed: int = 0) -> Dataset:
    """英語・priorityがlow/medium/highの行だけに絞り、シャッフル済みで返す。"""
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    ds = ds.filter(lambda r: r["language"] == "en" and r["priority"] in _PRIORITY_TO_LABEL)
    return ds.shuffle(seed=seed)


def load_support_tickets(split: str, n: int, seed: int = 0) -> Dataset:
    """`split="train"`は先頭から、`split="held_out"`は末尾からn件を取る（重複しない）。"""
    ds = _load_all(seed=seed)
    if split == "train":
        return ds.select(range(n))
    if split == "held_out":
        return ds.select(range(len(ds) - n, len(ds)))
    raise ValueError(f"未知のsplit: {split!r}（'train'または'held_out'）")


def to_packed_example(subject: str, body: str, priority: str, rng: random.Random) -> ChoiceExample:
    """1件（件名＋本文＋priority）を、K=3固定のScore PackedRequestに変換する。"""
    label = _PRIORITY_TO_LABEL[priority]
    subject = subject or ""
    body = body or ""
    text = f"{subject}\n\n{body}"[: _MAX_BODY_CHARS + len(subject) + 2]

    instructions = pick_instruction(INSTRUCTION_VARIANTS, rng)
    branch = Branch(kind="score", instructions=instructions, options=list(LEVEL_ORDER))
    request = PackedRequest(state=text, branches=[branch])
    return ChoiceExample(request=request, label=label)


def _smoke_check() -> None:
    from model.packer import load_packer_tokenizer, pack

    rng = random.Random(0)
    tokenizer = load_packer_tokenizer()
    ds = load_support_tickets("train", 20)

    seen_labels: set[int] = set()
    for row in ds:
        ex = to_packed_example(row["subject"], row["body"], row["priority"], rng)
        packed = pack(ex.request, tokenizer)
        assert len(packed.branch_spans) == 1
        assert len(packed.branch_spans[0].option_end_indices) == 3
        assert 0 <= ex.label < 3
        seen_labels.add(ex.label)

    train_ds = load_support_tickets("train", 100)
    held_out_ds = load_support_tickets("held_out", 100)
    train_ids = set(train_ds["subject"])
    held_out_ids = set(held_out_ds["subject"])
    # 完全に厳密ではない（件名重複がありうる）が、大きくは重ならないはずの簡易チェック。
    overlap = train_ids & held_out_ids
    assert len(overlap) < 5, f"train/held_outの重複が多すぎる: {len(overlap)}件"

    print(f"OK: 20件でlabel={sorted(seen_labels)}、K=3固定のScore PackedRequestに変換でき、"
          f"packerでトークン化までできることを確認。train/held_out(各100件)の件名重複={len(overlap)}件。")


if __name__ == "__main__":
    _smoke_check()
