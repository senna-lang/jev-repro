"""BoolQ（read-comprehension型yes/no質問応答）を読み込む、Noul（yes/no）訓練データの候補。

state=passage、instructions=question（先頭大文字化＋"?"補完）とすることで、
`model/packer.py::pack()`のnoul branchテンプレート（"{instructions} Answer yes or no.\\nAnswer:"）
にそのまま乗る。ラベルはanswer（bool）。

train/validation合わせて62.3%がTrue（不均衡）なので、`load_boolq_stratified`で
yes/noを均等にサンプルする。

訓練データとして使う前に、`experiments/boolq_probe.py`でzero-shot信号（Q1: yes/noの
判別に必要な手がかりが凍結backboneに既に見えるか）を先に確認すること
（`support_tickets`で同じ手順を省略し、無駄な訓練を回した反省: Gate 1-b）。
"""

from __future__ import annotations

import random

from datasets import Dataset, concatenate_datasets, load_dataset

from model.packer import Branch, PackedRequest


def _question_to_instructions(question: str) -> str:
    q = question.strip()
    q = q[0].upper() + q[1:] if q else q
    if not q.endswith("?"):
        q += "?"
    return q


def _load_all(seed: int = 0) -> Dataset:
    """train+validationを結合し、passage/questionが非空の行だけに絞り、シャッフル済みで返す。"""
    ds = concatenate_datasets(
        [load_dataset("google/boolq", split="train"), load_dataset("google/boolq", split="validation")]
    )
    ds = ds.filter(lambda ex: bool(ex["passage"]) and bool(ex["question"]))
    return ds.shuffle(seed=seed)


def load_boolq_stratified(n_per_label: int, seed: int = 0, offset: int = 0) -> Dataset:
    """yes/noそれぞれn_per_label件を均等にサンプルする（BoolQは62.3%がTrueで不均衡）。

    各ラベルプールの`[offset, offset+n_per_label)`区間を取る。train/held_outで
    サイズが異なっていても、offsetを`train_n`（またはそれ以上）にすれば確実に重複しない。
    """
    pool = _load_all(seed=seed)
    true_idx = [i for i, ex in enumerate(pool) if ex["answer"] is True]
    false_idx = [i for i, ex in enumerate(pool) if ex["answer"] is False]

    def _take(indices: list[int]) -> list[int]:
        return indices[offset : offset + n_per_label]

    selected = _take(true_idx) + _take(false_idx)
    if len(selected) != 2 * n_per_label:
        raise ValueError(
            f"要求されたn_per_label={n_per_label}, offset={offset}に対しデータが不足"
            f"（true={len(true_idx)}, false={len(false_idx)}）"
        )
    rng = random.Random(seed)
    rng.shuffle(selected)
    return pool.select(selected)


def to_packed_example(question: str, passage: str, answer: bool) -> tuple[PackedRequest, bool]:
    """1件（question＋passage）を、Noul branch1つのPackedRequestに変換する。"""
    instructions = _question_to_instructions(question)
    branch = Branch(kind="noul", instructions=instructions, options=[])
    request = PackedRequest(state=passage, branches=[branch])
    return request, answer


def _smoke_check() -> None:
    ds = load_boolq_stratified(n_per_label=5, seed=0, offset=0)
    counts: dict[bool, int] = {}
    for ex in ds:
        counts[ex["answer"]] = counts.get(ex["answer"], 0) + 1
        request, label = to_packed_example(ex["question"], ex["passage"], ex["answer"])
        assert request.branches[0].kind == "noul"
        assert request.branches[0].options == []
        assert isinstance(label, bool)
    print(f"OK: n={len(ds)}, label分布={counts}")


if __name__ == "__main__":
    _smoke_check()
