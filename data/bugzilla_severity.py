"""Eclipse/Mozilla Bugzillaの実バグ報告のseverityラベル（技術的深刻度）を読み込む。

`support_tickets`のpriorityラベル（業務優先度、テキストとの相関が実質ゼロ、
`results/logs/2026-09-23_gate1_pooling_and_severity_data.md`参照）に代わる、より適切な
severity代理データの候補。BugzillaのSeverity Labelはバグトリアージ担当者が付ける
技術的深刻度そのもので、公式Score severity_tableが表す軸（cosmetic〜blocking）に近い。

公式Bugzillaの順序（重い順）: blocker > critical > major > normal > minor > trivial。
本モジュールは軽い順に並べる（`SEVERITY_ORDER`のindexが大きいほど重い）。

訓練データとして使う前に、`experiments/bugzilla_severity_probe.py`でzero-shot信号
（Q1: テキストとラベルの相関が凍結backboneに既に見えるか）を先に確認すること
（`support_tickets`で同じ手順を省略し、無駄な訓練を回した反省: Gate 1-b）。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample, pick_instruction
from model.packer import Branch, PackedRequest

INSTRUCTION_VARIANTS = [
    "How severe is the bug described in this report?",
    "Rate the severity of this bug report.",
    "Judge how severe or minor the bug in this report is.",
]

# 軽い順（公式Bugzillaの重い順 blocker>critical>major>normal>minor>trivial を反転）。
SEVERITY_ORDER = ["trivial", "minor", "normal", "major", "critical", "blocker"]
_LABEL_TO_INDEX = {name: i for i, name in enumerate(SEVERITY_ORDER)}


def _load_all(seed: int = 0) -> Dataset:
    """有効なseverityラベル・非空の説明文の行だけに絞り、シャッフル済みで返す。"""
    ds = load_dataset("AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset", split="train")
    ds = ds.filter(
        lambda r: r["Severity Label"] in _LABEL_TO_INDEX
        and r["Short Description"] is not None
        and len(r["Short Description"].strip()) >= 10
    )
    return ds.shuffle(seed=seed)


def load_bugzilla_severity(split: str, n: int, seed: int = 0) -> Dataset:
    """`split="train"`は先頭から、`split="held_out"`は末尾からn件を取る（重複しない）。"""
    ds = _load_all(seed=seed)
    if split == "train":
        return ds.select(range(n))
    if split == "held_out":
        return ds.select(range(len(ds) - n, len(ds)))
    raise ValueError(f"未知のsplit: {split!r}（'train'または'held_out'）")


def load_bugzilla_severity_stratified(n_per_level: int, seed: int = 0, half: str = "second") -> Dataset:
    """severityレベルごとに均等にn_per_level件サンプルする（severity分布が'normal'に
    大きく偏っている— 81%—ため、学習・tau測定のバランスを取るのに使う）。

    `half="first"`は前半、`half="second"`（既定）は後半からのみ取り、`half`を変えて
    呼べば訓練用とheld-out用が重ならない（例: 訓練=`half="first"`, held-out=`half="second"`）。
    """
    ds = _load_all(seed=seed)
    mid = len(ds) // 2
    pool = ds.select(range(0, mid)) if half == "first" else ds.select(range(mid, len(ds)))
    by_level: dict[str, list[int]] = {name: [] for name in SEVERITY_ORDER}
    for i, label in enumerate(pool["Severity Label"]):
        if len(by_level[label]) < n_per_level:
            by_level[label].append(i)
        if all(len(v) >= n_per_level for v in by_level.values()):
            break
    indices = [i for idxs in by_level.values() for i in idxs]
    return pool.select(indices)


def to_packed_example(description: str, severity: str, rng: random.Random) -> ChoiceExample:
    """1件（Short Description＋severity）を、K=6固定のScore PackedRequestに変換する。"""
    label = _LABEL_TO_INDEX[severity]
    instructions = pick_instruction(INSTRUCTION_VARIANTS, rng)
    branch = Branch(kind="score", instructions=instructions, options=list(SEVERITY_ORDER))
    request = PackedRequest(state=description, branches=[branch])
    return ChoiceExample(request=request, label=label)


def _smoke_check() -> None:
    from model.packer import load_packer_tokenizer, pack

    rng = random.Random(0)
    tokenizer = load_packer_tokenizer()
    ds = load_bugzilla_severity("train", 20)

    seen_labels: set[int] = set()
    for row in ds:
        ex = to_packed_example(row["Short Description"], row["Severity Label"], rng)
        packed = pack(ex.request, tokenizer)
        assert len(packed.branch_spans) == 1
        assert len(packed.branch_spans[0].option_end_indices) == 6
        assert 0 <= ex.label < 6
        seen_labels.add(ex.label)

    strat = load_bugzilla_severity_stratified(5)
    from collections import Counter
    counts = Counter(strat["Severity Label"])
    print(f"OK: 20件でlabel={sorted(seen_labels)}、K=6固定のScore PackedRequestに変換でき、"
          f"packerでトークン化までできることを確認。stratifiedサンプル(5/level)のlevel分布: "
          f"{dict(counts)}")


if __name__ == "__main__":
    _smoke_check()
