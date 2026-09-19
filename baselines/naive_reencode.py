"""共有stateを再利用せず、branchごとに標準causal forwardを実行する比較用実装。

`JevModel`と同じbackboneおよびreadoutの重みを使い、各branchを`state + branch`として
独立に評価する。tree attention maskによる複数branchの同時評価とは実行方法だけが異なる。
"""

from __future__ import annotations

import torch

from model.forward import JevModel
from model.packer import PackedRequest, pack
from model.readout import score_expectation


def naive_reencode_forward(model: JevModel, request: PackedRequest) -> list[dict]:
    """questionごとに`[state] + [そのbranch]`を独立forwardする。tree maskは使わない（＝標準causal）。

    `model`は`forward.JevModel`のインスタンス。backbone・tokenizer・readout（pointer/noul）の
    重みをそのまま流用し、計算グラフだけをbranchごとに独立させる。
    """
    results: list[dict] = []
    for branch in request.branches:
        single_request = PackedRequest(state=request.state, branches=[branch])
        packed = pack(single_request, model.tokenizer)  # branchは1つだけなので、標準causalで十分

        input_ids = torch.tensor([packed.input_ids], dtype=torch.long)
        with torch.no_grad():
            out = model.backbone(input_ids=input_ids)  # attention_mask/position_ids省略＝標準causal, 0..T-1
        hidden = out.last_hidden_state[0]

        span = packed.branch_spans[0]
        h_dec = hidden[span.decision_index]

        if branch.kind == "noul":
            p_true = torch.sigmoid(model.noul(h_dec)).item()
            results.append({"kind": "noul", "p_true": p_true})
            continue

        h_opts = hidden[span.option_end_indices]
        z = model.pointer(h_dec, h_opts)
        p = torch.softmax(z, dim=-1)
        entry = {"kind": branch.kind, "options": branch.options, "probabilities": p.tolist()}
        if branch.kind == "score":
            entry["score"] = score_expectation(p)
        results.append(entry)

    return results


def _smoke_check() -> None:
    from model.packer import Branch

    model = JevModel()

    # branchが1つだけのリクエストなら、tree mask版と naive re-encode版は
    # 数式上まったく同じ計算になるはず（state+単一branchでmaskが標準causalに退化するため）。
    single_branch_request = PackedRequest(
        state="Oil prices surged 5% today after OPEC announced production cuts.",
        branches=[
            Branch(
                kind="choice",
                instructions="この記事のトピックを選べ。",
                options=["World", "Sports", "Business", "Sci-Tech"],
            )
        ],
    )

    tree_mask_result = model.forward(single_branch_request)[0]
    naive_result = naive_reencode_forward(model, single_branch_request)[0]

    print(f"tree mask版:      {tree_mask_result['probabilities']}")
    print(f"naive re-encode版: {naive_result['probabilities']}")

    assert tree_mask_result["probabilities"] == naive_result["probabilities"], (
        "branchが1つだけのときは、tree mask版とnaive re-encode版が一致するはず"
    )

    # 複数branchでも、独立forwardとして壊れずに動くことを確認
    multi_branch_request = PackedRequest(
        state=single_branch_request.state,
        branches=[
            single_branch_request.branches[0],
            Branch(kind="noul", instructions="この記事は経済に関するものか？"),
        ],
    )
    multi_results = naive_reencode_forward(model, multi_branch_request)
    assert len(multi_results) == 2
    assert abs(sum(multi_results[0]["probabilities"]) - 1.0) < 1e-4
    assert 0.0 <= multi_results[1]["p_true"] <= 1.0

    print("OK: branch1つならtree mask版と完全一致（重み共有・計算グラフだけが違う）／"
          "複数branchでも独立forwardが壊れず動く。")


if __name__ == "__main__":
    _smoke_check()
