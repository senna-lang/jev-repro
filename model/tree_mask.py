"""共有stateとbranch群のためのcausal tree attention maskとposition IDを構築する。

各トークンは過去のstateと同じbranchの過去だけを参照できる。stateのposition IDは
`0..S-1`、各branchのposition IDは`S`から独立して振るため、branchを並べ替えても
他branchの位置番号は変化しない。

    mask(i, j) = [j <= i] AND ([j in state] OR [branch(i) = branch(j)])
"""

from __future__ import annotations

import torch

from .packer import PackedSequence

# 全branchで共有されるstate部分を表すbranch id
STATE_BRANCH_ID = -1


def build_branch_ids(packed: PackedSequence) -> list[int]:
    """各トークンが所属するbranchのindexを返す（state部分は`STATE_BRANCH_ID`）。"""
    branch_ids = [STATE_BRANCH_ID] * len(packed.input_ids)
    for branch_idx, span in enumerate(packed.branch_spans):
        for i in range(span.start, span.decision_index + 1):
            branch_ids[i] = branch_idx
    return branch_ids


def build_tree_attention_mask(packed: PackedSequence) -> torch.Tensor:
    """`mask(i,j) = causal(i,j) AND (state(j) OR branch(i)==branch(j))` を (T, T) の bool テンソルで返す。

    `mask[i, j] == True` は「query位置iがkey位置jを見てよい」の意味（通常のattention maskの向き）。
    """
    branch_ids = torch.tensor(build_branch_ids(packed), dtype=torch.long)
    t = branch_ids.shape[0]

    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    state_mask_j = branch_ids == STATE_BRANCH_ID  # (T,) — jがstateかどうか
    same_branch = branch_ids.unsqueeze(1) == branch_ids.unsqueeze(0)  # (T,T) — branch(i)==branch(j)

    return causal & (state_mask_j.unsqueeze(0) | same_branch)


def build_position_ids(packed: PackedSequence) -> list[int]:
    """state: 0..S-1、各branch: S, S+1, S+2, ...（他のbranchの長さ・並び順を一切参照しない）。"""
    s = packed.state_len
    position_ids = list(range(s))
    for span in packed.branch_spans:
        length = span.decision_index - span.start + 1
        position_ids.extend(range(s, s + length))
    return position_ids


def _smoke_check() -> None:
    from .packer import Branch, PackedRequest, load_packer_tokenizer, pack

    tokenizer = load_packer_tokenizer()

    branch_a = Branch(
        kind="choice",
        instructions="この記事のトピックを選べ。",
        options=["World", "Sports", "Business", "Sci-Tech"],
    )
    branch_b = Branch(kind="noul", instructions="この記事は経済に関するものか？")

    state = "Oil prices surged 5% today after OPEC announced production cuts."

    packed_ab = pack(PackedRequest(state=state, branches=[branch_a, branch_b]), tokenizer)
    packed_ba = pack(PackedRequest(state=state, branches=[branch_b, branch_a]), tokenizer)

    mask_ab = build_tree_attention_mask(packed_ab)
    pos_ab = build_position_ids(packed_ab)

    t = len(packed_ab.input_ids)
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))

    print(f"T={t}, state_len={packed_ab.state_len}")
    print(f"mask shape: {tuple(mask_ab.shape)}, visible pairs: {int(mask_ab.sum())} / causal pairs: {int(causal.sum())}")
    print(f"position_ids: {pos_ab}")

    # 1. mask は causal の部分集合（何も新しく見せてはいけない）
    assert torch.equal(mask_ab & causal, mask_ab), "maskがcausalの範囲を超えて可視にしている"

    branch_ids = build_branch_ids(packed_ab)
    span_a, span_b = packed_ab.branch_spans  # 順序は [branch_a, branch_b]

    # 2. state トークンは、causalに許される範囲では全員から見える
    for j in range(packed_ab.state_len):
        for i in range(j, t):
            assert mask_ab[i, j], f"state token j={j} が i={i} から見えていない"

    # 3. branch同士は完全に不可視（同じbranch内以外はcausalでも見えない）
    for i in range(t):
        for j in range(t):
            if branch_ids[i] != STATE_BRANCH_ID and branch_ids[j] != STATE_BRANCH_ID:
                if branch_ids[i] != branch_ids[j]:
                    assert not mask_ab[i, j], f"別branch同士 i={i}(branch={branch_ids[i]}) j={j}(branch={branch_ids[j]}) が見えてしまっている"

    # 4. 同じbranch内は普通のcausalと一致する
    for i in range(t):
        for j in range(t):
            if branch_ids[i] != STATE_BRANCH_ID and branch_ids[i] == branch_ids[j]:
                assert mask_ab[i, j] == causal[i, j], f"同一branch内 i={i} j={j} でcausalと不一致"

    # 5. position_id: state部分は 0..S-1、各branchはS,S+1,...で始まる
    s = packed_ab.state_len
    assert pos_ab[:s] == list(range(s)), "state部分のposition_idが0..S-1になっていない"
    assert pos_ab[span_a.start] == s, "branch_aの先頭position_idがSになっていない"
    assert pos_ab[span_b.start] == s, "branch_bの先頭position_idがSになっていない（branch順序に依存してはいけない）"

    # 6. isolation: branchの並び順を入れ替えても、各branch自身のposition_id列とmask構造は不変
    pos_ba = build_position_ids(packed_ba)
    # packed_ba は [branch_b, branch_a] の順。branch_a の中身は packed_ab では branches[0]、packed_ba では branches[1]
    span_a_in_ba = packed_ba.branch_spans[1]
    len_a = span_a.decision_index - span_a.start + 1
    len_a_in_ba = span_a_in_ba.decision_index - span_a_in_ba.start + 1
    assert len_a == len_a_in_ba, "branch_aのトークン長が並び順によって変わっている（tokenizeがおかしい）"
    pos_a_in_ab = pos_ab[span_a.start : span_a.decision_index + 1]
    pos_a_in_ba = pos_ba[span_a_in_ba.start : span_a_in_ba.decision_index + 1]
    assert pos_a_in_ab == pos_a_in_ba, "branch_aのposition_id列が、並び順を変えただけで変化している（isolation違反）"

    print("OK: mask=causalの部分集合／state全可視／branch間不可視／同branch内はcausal一致／"
          "position_idはbranchごとにS始まり／並び順を変えてもbranch自身のposition_idは不変。")


if __name__ == "__main__":
    _smoke_check()
