"""共有stateと複数branchを1本のトークン列へ直列化する。

`<STATE_END>`・`<SEP>`・`<DECISION>`をトークナイザへ追加し、各branchのinstruction、
選択肢、decision tokenの境界位置を`PackedSequence`として返す。

ChoiceとScoreは選択肢または基準の説明文を`<SEP>`で区切る。Noulは選択肢を持たず、
instructionと`<DECISION>`だけで表す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .backbone import MODEL_NAME

STATE_END = "<STATE_END>"
SEP = "<SEP>"
DECISION = "<DECISION>"
PACKER_SPECIAL_TOKENS = [STATE_END, SEP, DECISION]

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
        if self.kind == "noul" and self.options:
            raise ValueError("noul branch は options を持てない（yes/noに選択肢は不要）")
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
    option_end_indices: list[int]  # 各選択肢の最後のトークンのindex（順序どおり）。noulなら空
    decision_index: int  # <DECISION>トークンのindex（このbranchの最後）


@dataclass
class PackedSequence:
    """sequence packerの出力。"""

    input_ids: list[int]
    state_len: int  # state部分のトークン数（STATE_ENDを含む）
    branch_spans: list[BranchSpan]


def load_packer_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerBase:
    """packer用の特殊トークン（STATE_END/SEP/DECISION）を追加したtokenizerを返す。

    既存のQwen2.5特殊トークン（`<|im_start|>`等）とは衝突しない3つを追加するだけで、
    通常のBPE語彙は一切変更しない。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": PACKER_SPECIAL_TOKENS})
    return tokenizer


def pack(request: PackedRequest, tokenizer: PreTrainedTokenizerBase) -> PackedSequence:
    """`PackedRequest` を1本のトークン列に直列化する。

    state → STATE_END → (branchごとに instructions → [options<SEP>]* → DECISION) を
    そのままの順で繋げるだけ。tree mask・position id・readoutはここでは計算しない。
    """
    state_end_id, sep_id, decision_id = tokenizer.convert_tokens_to_ids(PACKER_SPECIAL_TOKENS)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids: list[int] = encode(request.state) + [state_end_id]
    state_len = len(input_ids)

    branch_spans: list[BranchSpan] = []
    for branch in request.branches:
        branch_start = len(input_ids)
        input_ids.extend(encode(branch.instructions))

        option_end_indices: list[int] = []
        for option in branch.options:
            input_ids.extend(encode(option))
            option_end_indices.append(len(input_ids) - 1)  # そのoptionの最後のトークン
            input_ids.append(sep_id)

        input_ids.append(decision_id)
        decision_index = len(input_ids) - 1

        branch_spans.append(
            BranchSpan(
                kind=branch.kind,
                start=branch_start,
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
                instructions="この記事のトピックを選べ。",
                options=["World", "Sports", "Business", "Sci-Tech"],
            ),
            Branch(
                kind="noul",
                instructions="この記事は経済に関するものか？",
            ),
        ],
    )

    packed = pack(request, tokenizer)

    print(f"input_ids length: {len(packed.input_ids)}")
    print(f"state_len: {packed.state_len}")
    print(f"num branches: {len(packed.branch_spans)}")

    decoded_state = tokenizer.decode(packed.input_ids[: packed.state_len])
    print(f"state block decoded: {decoded_state!r}")
    assert decoded_state.endswith(STATE_END), "state blockの末尾がSTATE_ENDになっていない"

    for i, (branch, span) in enumerate(zip(request.branches, packed.branch_spans)):
        assert tokenizer.decode([packed.input_ids[span.decision_index]]) == DECISION, (
            f"branch {i}: decision_index位置のトークンがDECISIONでない"
        )
        assert len(span.option_end_indices) == len(branch.options), (
            f"branch {i}: option_end_indicesの数がoptionsの数と一致しない"
        )
        for opt_text, end_idx in zip(branch.options, span.option_end_indices):
            # そのoptionの最後のトークンをデコードしたら、option文字列の末尾と一致するはず
            opt_last_token_text = tokenizer.decode([packed.input_ids[end_idx]])
            assert opt_text.rstrip().endswith(opt_last_token_text.strip()) or opt_last_token_text.strip() in opt_text, (
                f"branch {i} option {opt_text!r}: option_end_indexが指すトークン{opt_last_token_text!r}がoption末尾と整合しない"
            )
            # end_indexの次はSEP（そのoptionが最後の選択肢でも、DECISIONの前にSEPが入る設計）
            assert tokenizer.decode([packed.input_ids[end_idx + 1]]) == SEP, (
                f"branch {i} option {opt_text!r}: option直後がSEPになっていない"
            )
        print(f"branch {i} ({branch.kind}): start={span.start}, decision_index={span.decision_index}, "
              f"option_end_indices={span.option_end_indices}")

    # branch同士がトークン列上で重ならず、順番通りに並んでいることを確認
    for prev, cur in zip(packed.branch_spans, packed.branch_spans[1:]):
        assert cur.start == prev.decision_index + 1, "branch同士がトークン列上で連続していない（隙間または重なりがある）"

    assert packed.branch_spans[-1].decision_index == len(packed.input_ids) - 1, (
        "最後のbranchのDECISIONがトークン列の末尾になっていない"
    )

    print("OK: state/branch/option/DECISIONの境界情報がすべて期待通り。")


if __name__ == "__main__":
    _smoke_check()
