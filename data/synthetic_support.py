"""LLMで合成した英語サポートメッセージを、Choice / Scoreの訓練データに変換する。

GOAL.md §6「学習データ」に従う: Gold V1（原本・英語版）の文面は生成プロンプトに入れない。
ここにあるタスク仕様（設問・ラベル）はこのリポジトリで独自に書いたもの。Choiceは各タスクの
ラベルプール（5〜7個）から正解1つ＋distractorでK=3〜4を毎回サンプルし、固定の選択肢集合に
癒着しないようにする（`data/banking77.py`と同じ方式）。Scoreは順序付きレベルを固定で使う。

生成済みデータは`data/synthetic/support_v{N}.jsonl`（1行=1メッセージ、`task`/`label`/`text`/`idx`）、
生成条件は同名の`.manifest.json`。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from data.common import ChoiceExample
from model.packer import Branch, PackedRequest

DATA_DIR = Path(__file__).with_name("synthetic")


@dataclass(frozen=True)
class TaskSpec:
    kind: str  # "choice" | "score"
    question: str
    labels: tuple[str, ...]  # scoreは低い順


TASKS: dict[str, TaskSpec] = {
    "team_routing": TaskSpec(
        "choice",
        "Which team should own resolving this message?",
        (
            "Accounts and billing — charges, refunds, invoices, payment methods",
            "Engineering support — bugs, errors, outages, login or access failures",
            "Sales — pricing, plan comparisons, and upgrades before buying",
            "Logistics — shipment tracking, delivery, and couriers",
            "Returns desk — sending products back and exchanges",
            "Account security — suspicious activity, hacked accounts, 2FA",
        ),
    ),
    "return_cause": TaskSpec(
        "choice",
        "What is the main reason the customer wants to send the product back?",
        (
            "it matches the order but the size or fit is wrong",
            "a different product, size, or color was delivered than ordered",
            "it arrived broken or stopped working",
            "they simply changed their mind",
            "it does not match the description or photos",
            "it arrived too late to be useful",
        ),
    ),
    "delivery_problem": TaskSpec(
        "choice",
        "Which delivery problem is the customer describing?",
        (
            "the package is delayed but still in transit",
            "tracking says delivered but nothing was received",
            "the package went to the wrong address",
            "the package was damaged during shipping",
            "items are missing from an otherwise delivered package",
            "the tracking number does not work or shows no updates",
        ),
    ),
    "desired_outcome": TaskSpec(
        "choice",
        "What outcome does the customer most want?",
        (
            "a replacement of the correct product",
            "a refund of their money",
            "the same item sent again",
            "only an answer or information, no action",
            "a discount or store credit",
            "cancellation of the order or subscription",
            "to talk to a supervisor or manager",
        ),
    ),
    "customer_tone": TaskSpec(
        "choice",
        "How would you describe the customer's tone?",
        (
            "neutral and matter-of-fact",
            "friendly and appreciative",
            "annoyed but still polite",
            "angry, hostile, or threatening",
            "anxious or worried",
            "confused and unsure what to do",
        ),
    ),
    "info_topic": TaskSpec(
        "choice",
        "What information is the customer asking for?",
        (
            "whether a product is eligible to be returned",
            "the progress of a return they already shipped",
            "how to swap an item for a different size or model",
            "who pays for return shipping",
            "when a refund will arrive",
            "what the warranty covers",
        ),
    ),
    "issue_severity": TaskSpec(
        "score",
        "How serious is the problem the customer reports?",
        (
            "cosmetic only; everything still works",
            "something is broken but an easy workaround exists",
            "broken, and the only workaround is slow or painful",
            "completely blocked with no workaround",
        ),
    ),
    "report_detail": TaskSpec(
        "score",
        "How much does this report give an engineer to act on?",
        (
            "only says something does not work",
            "names the affected feature or page",
            "gives concrete steps to reproduce",
            "gives steps to reproduce plus exact environment details such as app version, OS, or device",
        ),
    ),
    "anger_level": TaskSpec(
        "score",
        "How upset is the customer?",
        (
            "calm, no sign of frustration",
            "mildly irritated",
            "clearly upset but still civil",
            "furious, insulting, or threatening",
        ),
    ),
}

MIN_K = 3
MAX_K = 4

# 学習時にrngで選ぶ設問の言い換え（固定文面への癒着を防ぐ）。先頭は`TaskSpec.question`と同一。
QUESTION_VARIANTS: dict[str, tuple[str, ...]] = {
    "team_routing": (TASKS["team_routing"].question, "Route this message to the right team.", "Who is responsible for handling this request?"),
    "return_cause": (TASKS["return_cause"].question, "Why is this item being returned?", "What is behind the customer's return request?"),
    "delivery_problem": (TASKS["delivery_problem"].question, "What went wrong with the shipment?", "Classify the shipping issue in this message."),
    "desired_outcome": (TASKS["desired_outcome"].question, "What resolution is the customer asking for?", "What does the customer want us to do?"),
    "customer_tone": (TASKS["customer_tone"].question, "What is the emotional tone of this message?", "How does the customer come across?"),
    "info_topic": (TASKS["info_topic"].question, "What does the customer want to know?", "Which question is the customer asking?"),
    "issue_severity": (TASKS["issue_severity"].question, "Rate the severity of the reported issue.", "How badly is the customer affected?"),
    "report_detail": (TASKS["report_detail"].question, "How actionable is this bug report?", "Rate how much detail the report gives for debugging."),
    "anger_level": (TASKS["anger_level"].question, "How frustrated does the customer seem?", "Rate the customer's level of anger."),
}


def load_rows(version: int = 0) -> list[dict]:
    path = DATA_DIR / f"support_v{version}.jsonl"
    return [json.loads(line) for line in path.open()]


def split_rows(rows: list[dict], held_out_fraction: float = 0.3) -> tuple[list[dict], list[dict]]:
    """task×labelごとに`idx`の末尾側をheld-outにする（生成順と独立な`idx`で決定的に分ける）。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["label"]), []).append(row)
    train: list[dict] = []
    held: list[dict] = []
    for members in groups.values():
        members = sorted(members, key=lambda r: r["idx"])
        cut = len(members) - max(1, round(len(members) * held_out_fraction))
        train.extend(members[:cut])
        held.extend(members[cut:])
    return train, held


def to_packed_example(row: dict, rng: random.Random) -> ChoiceExample:
    spec = TASKS[row["task"]]
    label_index = spec.labels.index(row["label"])
    question = rng.choice(QUESTION_VARIANTS[row["task"]])
    if spec.kind == "score":
        branch = Branch(kind="score", instructions=question, options=list(spec.labels))
        return ChoiceExample(PackedRequest(state=row["text"], branches=[branch]), label=label_index)
    k = rng.randint(MIN_K, min(MAX_K, len(spec.labels)))
    others = [i for i in range(len(spec.labels)) if i != label_index]
    chosen = rng.sample(others, k - 1) + [label_index]
    rng.shuffle(chosen)
    branch = Branch(kind="choice", instructions=question, options=[spec.labels[i] for i in chosen])
    return ChoiceExample(PackedRequest(state=row["text"], branches=[branch]), label=chosen.index(label_index))
