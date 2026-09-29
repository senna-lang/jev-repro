"""共有stateと複数branchを1本のトークン列へ直列化する。

2026-09-22の診断（`results/logs/2026-09-22_score_diagnosis.md`）で、特殊トークン（`<STATE_END>`/
`<SEP>`/`<DECISION>`）による構造化フォーマットは、凍結backboneが持つ信号（LM head
zero-shotで最遠ペア0.900・公式severity 4/5一致）をほぼ完全に潰すことが分かった
（同条件で0.50前後まで低下）。選択肢を地の文の中にコンマ区切りで埋め込み「次の1語で
答えよ」と明示する自然文形式に替えると、信号は0.910・tau+0.600まで回復する
（`experiments/readout_probe.py`の`pack_with_prose_cue`で検証）。この形式を採用する。

ChoiceとScoreは選択肢または基準の説明文を instructions の地の文中にコンマ区切りで
列挙し、末尾の"Answer:"の直後（decision token）でreadoutを評価する。Noulは選択肢を
持たず、instructionと"Answer:"だけで表す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from legacy.model_readout.backbone import MODEL_NAME

BranchKind = Literal["choice", "score", "noul"]


@dataclass
class Branch:
    """1つのquestion（branch）の入力。

    - choice: `options` に選択肢の文字列を並べる（2つ以上）
    - score: `options` に `criteria` 配列の各レベルの説明文を並べる（Choiceと同じ扱い）
    - noul: 選択肢を持たない。`options` は空でなければならない
    """

    kind: BranchKind
    instructions: str
    options: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind == "noul" and self.options and set(self.options) != {"No", "Yes"}:
            raise ValueError("noul branch の options は No/Yes のみ")
        if self.kind in ("choice", "score") and len(self.options) < 2:
            raise ValueError(f"{self.kind} branch は options が2つ以上必要（実際: {len(self.options)}）")


@dataclass
class PackedRequest:
    """1リクエスト分の入力（state＋複数branch）。"""

    state: str
    branches: list[Branch]


@dataclass
class BranchSpan:
    """packされたトークン列上での、1つのbranchの位置情報（すべてグローバルindex）。"""

    kind: BranchKind
    start: int  # このbranchの最初のトークンのindex
    option_start_indices: list[int]  # 各選択肢の最初のトークンのindex（順序どおり）。noulなら空
    option_end_indices: list[int]  # 各選択肢の最後のトークンのindex（順序どおり）。noulなら空
    decision_index: int  # "Answer:"直後のトークンのindex（このbranchの最後）


@dataclass
class PackedSequence:
    """sequence packerの出力。"""

    input_ids: list[int]
    state_len: int  # state部分のトークン数（STATE_ENDを含む）
    branch_spans: list[BranchSpan]


def load_packer_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerBase:
    """packer用のtokenizerを返す。特殊トークンは追加しない（自然文のみでpackする）。"""
    return AutoTokenizer.from_pretrained(model_name)


def pack(request: PackedRequest, tokenizer: PreTrainedTokenizerBase) -> PackedSequence:
    """`PackedRequest` を1本のトークン列に直列化する。

    state → "\\n\\n" → (branchごとに、選択肢を地の文へコンマ区切りで埋め込んだ
    instructions → "\\nAnswer:") をそのままの順で繋げるだけ。tree mask・position id・
    readoutはここでは計算しない。
    """
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids: list[int] = encode(request.state) + encode("\n\n")
    state_len = len(input_ids)

    branch_spans: list[BranchSpan] = []
    for branch in request.branches:
        branch_start = len(input_ids)
        option_start_indices: list[int] = []
        option_end_indices: list[int] = []

        if branch.kind == "noul":
            choices = ", ".join(branch.options) if branch.options else "yes, no"
            input_ids.extend(encode(f"{branch.instructions} Answer with one word — {choices}.\nAnswer:"))
        else:
            input_ids.extend(encode(f"{branch.instructions} Answer with one word — "))
            for i, option in enumerate(branch.options):
                option_start_indices.append(len(input_ids))  # そのoptionの最初のトークン
                input_ids.extend(encode(option))
                option_end_indices.append(len(input_ids) - 1)  # そのoptionの最後のトークン
                is_last = i == len(branch.options) - 1
                input_ids.extend(encode("." if is_last else ", "))
            input_ids.extend(encode("\nAnswer:"))

        decision_index = len(input_ids) - 1

        branch_spans.append(
            BranchSpan(
                kind=branch.kind,
                start=branch_start,
                option_start_indices=option_start_indices,
                option_end_indices=option_end_indices,
                decision_index=decision_index,
            )
        )

    return PackedSequence(input_ids=input_ids, state_len=state_len, branch_spans=branch_spans)


def _smoke_check() -> None:
    """AG News風の1リクエスト（state＋Choice branch1つ＋Noul branch1つ）をpackして境界を検証する。"""
    tokenizer = load_packer_tokenizer()

    request = PackedRequest(
        state="Oil prices surged 5% today after OPEC announced production cuts.",
        branches=[
            Branch(
                kind="choice",
                instructions="What is the topic of this article?",
                options=["World", "Sports", "Business", "Sci-Tech"],
            ),
            Branch(
                kind="noul",
                instructions="Is this article about the economy?",
            ),
        ],
    )

    packed = pack(request, tokenizer)

    print(f"input_ids length: {len(packed.input_ids)}")
    print(f"state_len: {packed.state_len}")
    print(f"num branches: {len(packed.branch_spans)}")

    decoded_state = tokenizer.decode(packed.input_ids[: packed.state_len])
    print(f"state block decoded: {decoded_state!r}")
    assert decoded_state.endswith("\n\n"), "state blockの末尾が区切りの空行になっていない"

    for i, (branch, span) in enumerate(zip(request.branches, packed.branch_spans)):
        decoded_decision = tokenizer.decode(
            packed.input_ids[max(span.start, span.decision_index - 4): span.decision_index + 1]
        )
        assert decoded_decision.endswith("Answer:"), (
            f"branch {i}: decision_index直前が'Answer:'で終わっていない（{decoded_decision!r}）"
        )
        assert len(span.option_end_indices) == len(branch.options), (
            f"branch {i}: option_end_indicesの数がoptionsの数と一致しない"
        )
        assert len(span.option_start_indices) == len(branch.options), (
            f"branch {i}: option_start_indicesの数がoptionsの数と一致しない"
        )
        for opt_text, start_idx, end_idx in zip(branch.options, span.option_start_indices, span.option_end_indices):
            assert start_idx <= end_idx, f"branch {i} option {opt_text!r}: start_idx > end_idx"
            decoded_option = tokenizer.decode(packed.input_ids[start_idx: end_idx + 1])
            assert decoded_option.strip() == opt_text.strip(), (
                f"branch {i} option {opt_text!r}: option区間のデコード結果{decoded_option!r}が一致しない"
            )
            # そのoptionの最後のトークンをデコードしたら、option文字列の末尾と一致するはず
            opt_last_token_text = tokenizer.decode([packed.input_ids[end_idx]])
            assert opt_text.rstrip().endswith(opt_last_token_text.strip()) or opt_last_token_text.strip() in opt_text, (
                f"branch {i} option {opt_text!r}: option_end_indexが指すトークン{opt_last_token_text!r}がoption末尾と整合しない"
            )
        print(f"branch {i} ({branch.kind}): start={span.start}, decision_index={span.decision_index}, "
              f"option_end_indices={span.option_end_indices}")

    # branch同士がトークン列上で重ならず、順番通りに並んでいることを確認
    for prev, cur in zip(packed.branch_spans, packed.branch_spans[1:]):
        assert cur.start == prev.decision_index + 1, "branch同士がトークン列上で連続していない（隙間または重なりがある）"

    assert packed.branch_spans[-1].decision_index == len(packed.input_ids) - 1, (
        "最後のbranchの'Answer:'直後トークンがトークン列の末尾になっていない"
    )

    print("OK: state/branch/option/Answerの境界情報がすべて期待通り。")


if __name__ == "__main__":
    _smoke_check()
