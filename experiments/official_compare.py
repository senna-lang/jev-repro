"""TypeSafe公式ドキュメント掲載のJev実例（state＋questions＋公式出力）を同一入力として
`JevModel`に流し、出力を並べて比較する。

比較の位置づけ: バックボーンも訓練も異なるため数値の一致は期待しない。見るのは
(1) 同一入力から同じ形のtyped出力（選択肢集合内の確率分布・レベル間の位置・[0,1]の
noul）が出るか、(2) 1リクエスト複数questionが1回のforwardで全部答えられるか、
(3) 実際に出てくる値と公式掲載値の定性的な違い。

公式実例は https://docs.typesafe.ai の掲載値をそのまま使用する（live APIは使わない）。
表記の畳み込み: Choice criteriaは「名前 — 説明」、Noul criteriaと構造化criteriaは
instructions/option文字列に展開する（packerの入力は文字列リストのため）。
Scoreの公式値は0基準の位置。この実装の`score_expectation`は1基準なので、
比較時は-1して0基準に揃える。
"""

from __future__ import annotations

import sys

import torch
from model.forward import JevModel
from model.packer import Branch, PackedRequest

QUICKSTART_STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. "
    "I'm losing sales. Please help ASAP."
)
DEPARTMENT_CRITERIA = [
    ("billing", "Payment or subscription issues"),
    ("technical", "Bugs or integration problems"),
    ("sales", "Pricing or account questions"),
]
SEVERITY_LEVELS = [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists",
]
FRUSTRATION_LEVELS = [
    "Calm, just stating facts",
    "Frustrated but civil",
    "Very angry, strong language",
]


def _opt(name: str, desc: str | None) -> str:
    """公式のcriteria（名前＋説明）をpacker用の1つの選択肢文字列に畳み込む。説明なしは名前のみ。"""
    return name if desc is None else f"{name} — {desc}"


# 各要素: (question_id, Branch, 表示用の選択肢名リスト, 公式掲載値)
# 公式値: choice={"choice","probabilities","confidence"} / score={"score","probabilities","confidence"} /
#         noul={"noul"}。掲載されていないフィールドは省略。
EXAMPLES: list[dict] = [
    {
        "name": "quickstart_ticket",
        "source": "docs.typesafe.ai/introduction/quickstart",
        "state": QUICKSTART_STATE,
        "questions": [
            (
                "department",
                Branch(
                    kind="choice",
                    instructions="Which team should handle this",
                    options=[_opt(n, d) for n, d in DEPARTMENT_CRITERIA],
                ),
                [n for n, _ in DEPARTMENT_CRITERIA],
                {"choice": "billing", "probabilities": {"billing": 0.84, "technical": 0.159, "sales": 0.001}, "confidence": 0.596},
            ),
            (
                "frustration",
                Branch(kind="score", instructions="How frustrated the customer appears", options=FRUSTRATION_LEVELS),
                ["L0", "L1", "L2"],
                {"score": 1.035, "confidence": 0.842},
            ),
            (
                "is_urgent",
                Branch(kind="noul", instructions="The message conveys urgency or time-sensitivity"),
                [],
                {"noul": 0.999},
            ),
        ],
    },
    {
        "name": "easy_ticket",
        "source": "docs.typesafe.ai/primitives/choice",
        "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
        "questions": [
            (
                "department",
                Branch(
                    kind="choice",
                    instructions="Which team should handle this?",
                    options=[
                        _opt("returns", "Exchanges, refunds, wrong or damaged items"),
                        _opt("shipping", "Delivery status, delays, lost packages"),
                        _opt("billing", "Charges, invoices, payment problems"),
                    ],
                ),
                ["returns", "shipping", "billing"],
                {"choice": "returns", "probabilities": {"returns": 1.0, "shipping": 0.0, "billing": 0.0}, "confidence": 1.0},
            ),
        ],
    },
    {
        "name": "ambiguous_ticket (5 questions in 1 call)",
        "source": "docs.typesafe.ai/primitives/choice",
        "state": (
            "Shoes arrived two weeks late and in the wrong size. "
            "Also I see two charges on my card. What are you going to do about this?"
        ),
        "questions": [
            (
                "department",
                Branch(
                    kind="choice",
                    instructions="Which team should handle this?",
                    options=[
                        _opt("returns", "Exchanges, refunds, wrong or damaged items"),
                        _opt("shipping", "Delivery status, delays, lost packages"),
                        _opt("billing", "Charges, invoices, payment problems"),
                    ],
                ),
                ["returns", "shipping", "billing"],
                {"choice": "returns", "probabilities": {"returns": 0.6, "billing": 0.38, "shipping": 0.02}, "confidence": 0.39},
            ),
            (
                "return_reason",
                Branch(
                    kind="choice",
                    instructions="If the customer wants to return something, why?",
                    options=[
                        _opt("wrong_size", "The item doesn't fit"),
                        _opt("wrong_item", "A different product was delivered"),
                        _opt("damaged", "The item arrived broken or faulty"),
                        _opt("changed_mind", "The item is fine, the customer no longer wants it"),
                        _opt("other", "A return reason that fits none of the above"),
                    ],
                ),
                ["wrong_size", "wrong_item", "damaged", "changed_mind", "other"],
                {"choice": "wrong_size", "probabilities": {"wrong_size": 1.0, "wrong_item": 0.0, "damaged": 0.0, "changed_mind": 0.0, "other": 0.0}, "confidence": 1.0},
            ),
            (
                "shipping_issue",
                Branch(
                    kind="choice",
                    instructions="If this is a shipping problem, which kind is it?",
                    options=[
                        _opt("not_delivered", "The package never arrived"),
                        _opt("delayed", "The package is late but still on its way"),
                        _opt("wrong_address", "The package went to the wrong place"),
                        _opt("damaged_in_transit", "The package arrived damaged"),
                        _opt("other", "A shipping problem that fits none of the above"),
                    ],
                ),
                ["not_delivered", "delayed", "wrong_address", "damaged_in_transit", "other"],
                {"choice": "delayed", "probabilities": {"delayed": 0.63, "other": 0.37, "not_delivered": 0.0, "wrong_address": 0.0, "damaged_in_transit": 0.0}, "confidence": 0.53},
            ),
            (
                "requested_resolution",
                Branch(
                    kind="choice",
                    instructions="What does the customer want to happen?",
                    options=[
                        _opt("exchange", "Swap the item for a different one"),
                        _opt("refund", "Money back"),
                        _opt("replacement", "The same item sent again"),
                        _opt("information", "Just an answer, no action needed"),
                    ],
                ),
                ["exchange", "refund", "replacement", "information"],
                {"choice": "exchange", "probabilities": {"exchange": 0.37, "refund": 0.29, "replacement": 0.24, "information": 0.1}, "confidence": 0.16},
            ),
            (
                "tone",
                Branch(
                    kind="choice",
                    instructions="What is the customer's tone?",
                    options=["calm", "frustrated", "angry"],
                ),
                ["calm", "frustrated", "angry"],
                {"choice": "frustrated", "probabilities": {"frustrated": 0.92, "angry": 0.08, "calm": 0.0}, "confidence": 0.88},
            ),
        ],
    },
    {
        "name": "noul_ticket",
        "source": "docs.typesafe.ai/primitives/noul",
        "state": "I have asked three times now. Can I please just talk to a real person?",
        "questions": [
            (
                "is_human_escalation",
                Branch(kind="noul", instructions="Is the customer asking for a human agent?"),
                [],
                {"noul": 0.99},
            ),
            (
                "is_repeat_contact",
                Branch(
                    kind="noul",
                    instructions=(
                        "Has the customer contacted support about this before? "
                        "(yes: Mentions a prior attempt, ticket, or that they have asked before / "
                        "no: No sign of any previous contact)"
                    ),
                ),
                [],
                {"noul": 0.93},
            ),
        ],
    },
    {
        "name": "return_topic (structured criteria)",
        "source": "docs.typesafe.ai/primitives/choice",
        "state": "I sent the shoes back a week ago. When do I get my money?",
        "questions": [
            (
                "return_topic",
                Branch(
                    kind="choice",
                    instructions=(
                        "Which returns topic is the customer asking about? "
                        "(focus: Classify the information the customer wants.)"
                    ),
                    options=[
                        "return_policy — what: Whether and how an item can be returned; "
                        "not for: Progress of a return already sent; "
                        "examples: Can I return shoes I've worn once? / How long do I have to return an order?",
                        "return_status — what: Progress of a return already sent; "
                        "not for: Whether and how an item can be returned; "
                        "examples: Has my return arrived yet? / When will my refund be paid?",
                    ],
                ),
                ["return_policy", "return_status"],
                {"choice": "return_status", "probabilities": {"return_policy": 0.0, "return_status": 1.0}, "confidence": 1.0},
            ),
        ],
    },
    {
        "name": "severity_table (same Score question, 5 states)",
        "source": "docs.typesafe.ai/primitives/score",
        "state": None,  # 下のquestionsをstateごとに1問ずつ実行する
        "per_state": [
            (
                "The export button is misaligned by a few pixels on the settings page.",
                {"score": 0.0, "probabilities": [1.0, 0.0, 0.0], "confidence": 1.0},
            ),
            (
                "The PDF export button does nothing when clicked. I can still export to CSV and convert it myself, but that takes ages.",
                {"score": 1.0, "probabilities": [0.0, 1.0, 0.0], "confidence": 1.0},
            ),
            (
                "Export to PDF fails with a spinner that never finishes. Some of our team say CSV export still works for them, others say it fails too.",
                {"score": 1.12, "probabilities": [0.0, 0.88, 0.12], "confidence": 0.81},
            ),
            (
                "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.",
                {"score": 1.3, "probabilities": [0.0, 0.7, 0.3], "confidence": 0.54},
            ),
            (
                "Nobody on our team can log in since this morning. We get a 500 error on every attempt.",
                {"score": 2.0, "probabilities": [0.0, 0.0, 1.0], "confidence": 1.0},
            ),
        ],
        "score_branch": Branch(kind="score", instructions="How severe is the reported issue?", options=SEVERITY_LEVELS),
    },
    {
        "name": "spinner_ticket (3 Score questions in 1 call)",
        "source": "docs.typesafe.ai/primitives/score",
        "state": (
            "Export to PDF fails with a spinner that never finishes. Some of our team say CSV export "
            "still works for them, others say it fails too. This is the third time I'm writing in and "
            "honestly I'm done. Steps: open any report, click Export, choose PDF. Chrome 128 on macOS."
        ),
        "questions": [
            (
                "severity",
                Branch(kind="score", instructions="How severe is the reported issue?", options=SEVERITY_LEVELS),
                ["L0", "L1", "L2"],
                {"score": 1.24, "probabilities": [0.0, 0.76, 0.24], "confidence": 0.63},
            ),
            (
                "frustration",
                Branch(
                    kind="score",
                    instructions="How frustrated is the customer?",
                    options=[
                        "Calm, just stating facts",
                        "Frustrated but civil",
                        "Very angry, strong language or threatening to leave",
                    ],
                ),
                ["L0", "L1", "L2"],
                {"score": 1.45, "probabilities": [0.0, 0.55, 0.45], "confidence": 0.33},
            ),
            (
                "report_quality",
                Branch(
                    kind="score",
                    instructions="How much does the report give an engineer to work with?",
                    options=[
                        "No detail; just says something is broken",
                        "Names the feature but no steps or environment",
                        "Steps to reproduce or environment, but not both",
                        "Steps to reproduce and environment",
                    ],
                ),
                ["L0", "L1", "L2", "L3"],
                {"score": 3.0, "probabilities": [0.0, 0.0, 0.0, 1.0], "confidence": 1.0},
            ),
        ],
    },
]


def _fmt_probs_official(official: dict, names: list[str]) -> str:
    probs = official.get("probabilities")
    if probs is None:
        return ""
    if isinstance(probs, dict):
        return " ".join(f"{n}={probs[n]:.3f}" for n in names if n in probs)
    return " ".join(f"L{i}={p:.3f}" for i, p in enumerate(probs))


def _fmt_conf(official: dict) -> str:
    return f"  confidence={official['confidence']}" if "confidence" in official else ""


def run_example(model: JevModel, ex: dict) -> list[float]:
    """通常の例（1 state × 複数questionを1リクエスト）。各questionの出力を公式掲載値と並べて表示する。

    戻り値は構造チェック用の確率リスト（branchごとにflat化）。
    """
    branches = [q[1] for q in ex["questions"]]
    results = model.forward(PackedRequest(state=ex["state"], branches=branches))
    all_probs: list[float] = []

    print(f"\n=== {ex['name']}  (source: {ex['source']}) ===")
    print(f"state: {ex['state'][:72]}...")
    print(f"1 request / {len(branches)} questions -> 1 forward")

    for (qid, branch, names, official), result in zip(ex["questions"], results):
        print(f"  [{qid}] {branch.kind}")
        if branch.kind == "choice":
            ps = result["probabilities"]
            ours_choice = names[ps.index(max(ps))]
            print(f"    official: choice={official['choice']}  {_fmt_probs_official(official, names)}{_fmt_conf(official)}")
            print(f"    ours    : choice={ours_choice}  " + " ".join(f"{n}={p:.3f}" for n, p in zip(names, ps)))
            all_probs.extend(ps)
        elif branch.kind == "score":
            position = result["score"] - 1  # 公式の0基準位置に揃える
            ps = result["probabilities"]
            off_probs = _fmt_probs_official(official, names)
            print(f"    official: score={official['score']}  {off_probs}{_fmt_conf(official)}")
            print(f"    ours    : score={position:.3f}  " + " ".join(f"L{i}={p:.3f}" for i, p in enumerate(ps)))
            all_probs.extend(ps)
        else:  # noul
            print(f"    official: noul={official['noul']}")
            print(f"    ours    : noul={result['p_true']:.3f}")
            all_probs.append(result["p_true"])

    return all_probs


def run_severity_table(model: JevModel, ex: dict) -> list[float]:
    """Scoreドキュメントの5-state表: 同じscore questionをstateごとに1回ずつforwardする。"""
    print(f"\n=== {ex['name']}  (source: {ex['source']}) ===")
    all_probs: list[float] = []
    for state, official in ex["per_state"]:
        result = model.forward(PackedRequest(state=state, branches=[ex["score_branch"]]))[0]
        position = result["score"] - 1
        ps = result["probabilities"]
        print(f"  state: {state[:60]}...")
        print(f"    official: score={official['score']}  {_fmt_probs_official(official, [])}{_fmt_conf(official)}")
        print(f"    ours    : score={position:.3f}  " + " ".join(f"L{i}={p:.3f}" for i, p in enumerate(ps)))
        all_probs.extend(ps)
    return all_probs


def _main() -> None:
    # 再現性: readoutの初期化を固定する。argvにcheckpointパスを渡すと訓練済みreadoutで比較する
    # （例: python3 -m experiments.official_compare checkpoints/pointer_agnews_s1000_e3.pt）。
    torch.manual_seed(0)
    model = JevModel()

    trained = len(sys.argv) > 1
    if trained:
        ckpt = torch.load(sys.argv[1], map_location="cpu")
        model.pointer.load_state_dict(ckpt["state_dict"])
        readout_label = f"訓練済みreadout ({sys.argv[1]}, train_size={ckpt.get('train_size')}, epochs={ckpt.get('epochs')})"
    else:
        readout_label = "未学習readout（ランダム初期化, seed=0）"
    print(f"readout: {readout_label}")

    examples = EXAMPLES
    if trained:
        # 訓練済みcheckpointモードではNoulタスクを比較から除外する。
        # NoulReadout（線形層）はAG Newsにyes/noの訓練データがなく未訓練のため、
        # その出力を訓練済み扱いで並べるのは誤解を招く。noulのみの例はスキップ、
        # 混在する例はnoul questionを除いて残りを1リクエストで評価する。
        examples = []
        for ex in EXAMPLES:
            if ex.get("state") is None:
                examples.append(ex)
                continue
            questions = [q for q in ex["questions"] if q[1].kind != "noul"]
            if not questions:
                continue
            examples.append({**ex, "questions": questions})
        print("NoulReadoutは未訓練（訓練データにyes/noなし）のため、Noulタスクはこの比較から除外")

    all_probs: list[float] = []
    n_requests = 0
    n_questions = 0
    for ex in examples:
        if ex.get("state") is None:
            all_probs.extend(run_severity_table(model, ex))
            n_requests += len(ex["per_state"])
            n_questions += len(ex["per_state"])
        else:
            all_probs.extend(run_example(model, ex))
            n_requests += 1
            n_questions += len(ex["questions"])

    # 構造チェック: 全branchの確率が[0,1]内で和が1（noulは単一値）、typed出力の形が保たれているか
    in_range = all(0.0 <= p <= 1.0 for p in all_probs)
    print(f"\n=== 構造チェック ===")
    print(f"requests={n_requests}, question outputs={n_questions}, probability values={len(all_probs)}")
    print(f"全確率が[0,1]内: {'OK' if in_range else 'NG'}")
    print(f"（choice/scoreの各分布の和はsoftmax/sigmoidの構造上1。readout: {readout_label}。")
    print(f"  バックボーンも訓練domainも公式Jevと異なるため、数値の一致は比較の目的ではない。）")


if __name__ == "__main__":
    _main()
