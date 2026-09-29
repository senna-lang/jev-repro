"""サポートチケットの担当部署（`queue`）を、Gold V1の部署振り分けChoiceと同じ構造に変換する。

`Tobi-Bueck/customer-support-tickets`（`data/support_tickets.py`と同じソース、英語サブセット）の
`queue`列を使う。Gold V1のChoice（例: 請求担当/技術担当/営業担当 — 各部署の担当範囲付き）と
タスク構造を揃えるため、選択肢は「部署名 — 担当範囲」の説明付き文字列、K=3〜4
（正解1つ＋ランダムdistractor）にする。Gold V1のデータ・文面は一切使わない（分離規則）。

部署は互いに担当範囲が重ならない5つに限定する。`Product Support`/`IT Support`は
`Technical Support`と、`Customer Service`/`General Inquiry`は他の全部署と範囲が重なり、
正解が1つに定まらないため除外する。

訓練データとして使う前に`experiments/support_task_probe.py`でzero-shot信号を確認すること
（Gate 1-bの反省: 同ソースの`priority`はテキストと弱くしか相関していなかった）。
"""

from __future__ import annotations

import random

from datasets import Dataset, load_dataset

from data.common import ChoiceExample, pick_instruction
from model.packer import Branch, PackedRequest

INSTRUCTION_VARIANTS = [
    "Which team should handle this customer's request?",
    "Which department is responsible for resolving this ticket?",
    "Route this support ticket to the right team.",
]

QUEUE_OPTIONS = {
    "Billing and Payments": "Billing — charges, invoices, and payments",
    "Technical Support": "Technical support — product malfunctions and access problems",
    "Sales and Pre-Sales": "Sales — plans and purchases before buying",
    "Returns and Exchanges": "Returns — sending items back or exchanging them",
    "Human Resources": "Human resources — employment, payroll, and staff matters",
}
QUEUES = list(QUEUE_OPTIONS)
MIN_K = 3
MAX_K = 4
_MAX_BODY_CHARS = 600  # `data/support_tickets.py`と同じ上限


def _load_all(seed: int = 0) -> Dataset:
    """英語・対象queueの行だけに絞り、(subject, body)重複を除いてシャッフル済みで返す。

    同一チケットが複数行あると、offset分割してもtrain/held-outがテキストで重なる。
    """
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    ds = ds.filter(lambda r: r["language"] == "en" and r["queue"] in QUEUE_OPTIONS)
    seen: set[tuple[str, str]] = set()
    keep: list[int] = []
    for i, row in enumerate(ds):
        key = (row["subject"] or "", row["body"] or "")
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return ds.select(keep).shuffle(seed=seed)


def load_ticket_queue_stratified(n_per_queue: int, seed: int = 0, offset: int = 0) -> Dataset:
    """queueごとに`[offset, offset+n_per_queue)`区間を取る（offsetをずらせばtrain/held-outが重複しない）。"""
    pool = _load_all(seed=seed)
    by_queue: dict[str, list[int]] = {queue: [] for queue in QUEUES}
    for i, queue in enumerate(pool["queue"]):
        by_queue[queue].append(i)
    selected: list[int] = []
    for queue, indices in by_queue.items():
        taken = indices[offset : offset + n_per_queue]
        if len(taken) != n_per_queue:
            raise ValueError(f"{queue}: n_per_queue={n_per_queue}, offset={offset}に対し{len(indices)}件しかない")
        selected.extend(taken)
    random.Random(seed).shuffle(selected)
    return pool.select(selected)


def state_text(subject: str | None, body: str | None) -> str:
    subject = subject or ""
    body = body or ""
    return f"{subject}\n\n{body}"[: _MAX_BODY_CHARS + len(subject) + 2]


def to_packed_example(subject: str | None, body: str | None, queue: str, rng: random.Random) -> ChoiceExample:
    """正解queue1つ＋distractorでK=3〜4の説明付きChoiceにする（順序はランダム）。"""
    k = rng.randint(MIN_K, MAX_K)
    distractors = rng.sample([q for q in QUEUES if q != queue], k - 1)
    chosen = distractors + [queue]
    rng.shuffle(chosen)
    branch = Branch(
        kind="choice",
        instructions=pick_instruction(INSTRUCTION_VARIANTS, rng),
        options=[QUEUE_OPTIONS[q] for q in chosen],
    )
    return ChoiceExample(request=PackedRequest(state=state_text(subject, body), branches=[branch]), label=chosen.index(queue))


def _smoke_check() -> None:
    from collections import Counter

    from model.packer import load_packer_tokenizer, pack

    tokenizer = load_packer_tokenizer()
    rng = random.Random(0)
    ds = load_ticket_queue_stratified(4, seed=0)
    assert Counter(ds["queue"]) == {q: 4 for q in QUEUES}
    for row in ds:
        ex = to_packed_example(row["subject"], row["body"], row["queue"], rng)
        branch = ex.request.branches[0]
        assert MIN_K <= len(branch.options) <= MAX_K
        assert branch.options[ex.label] == QUEUE_OPTIONS[row["queue"]]
        assert len(pack(ex.request, tokenizer).branch_spans[0].option_end_indices) == len(branch.options)
    held_out = load_ticket_queue_stratified(4, seed=0, offset=4)
    overlap = set(zip(ds["subject"], ds["body"])) & set(zip(held_out["subject"], held_out["body"]))
    assert not overlap, overlap
    print(f"OK: queueごとに4件（{len(QUEUES)}部署）、K=3〜4の説明付きChoiceに変換、正解テキスト一致、"
          "packerでtokenize可、offset=4のheld-outと重複0件。")


if __name__ == "__main__":
    _smoke_check()
