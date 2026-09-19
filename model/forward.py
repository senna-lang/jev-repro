"""共有stateと複数branchを1回のバックボーンforwardで評価するモデルを組み立てる。

packer、tree attention mask、branch別position ID、readoutを接続し、Choice/Scoreには
選択肢確率、Noulには真である確率を返す。バックボーンは固定し、readoutだけを学習可能にする。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, Qwen2Model

from .backbone import MODEL_NAME
from .packer import Branch, PackedRequest, load_packer_tokenizer, pack
from .readout import NoulReadout, PointerReadout, score_expectation
from .tree_mask import build_position_ids, build_tree_attention_mask


def load_jev_backbone(model_name: str = MODEL_NAME) -> tuple[Qwen2Model, "PreTrainedTokenizerBase"]:  # noqa: F821
    """packer用の特殊トークンを含んだtokenizerと、それに合わせてembeddingを拡張したQwen2Modelを返す。

    `backbone.load_backbone`との違い：packerが追加した`<STATE_END>`/`<SEP>`/`<DECISION>`の3トークン分、
    埋め込み行列を`resize_token_embeddings`で拡張してから`Qwen2Model`を取り出す。これをしないと、
    packerが作るinput_idsのうち新トークンのidが埋め込み表の範囲外を指してしまう。

    `attn_implementation="eager"`を明示：tree attention maskをadditive float maskとして直接渡すため、
    加算方式（`attn_weights + mask`）が確定しているeager実装に固定する（sdpa等はmaskの型解釈が異なる）。
    """
    tokenizer = load_packer_tokenizer(model_name)
    causal_lm = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    causal_lm.resize_token_embeddings(len(tokenizer))

    backbone = causal_lm.model
    assert isinstance(backbone, Qwen2Model), f"想定外の型: {type(backbone)}"
    backbone.eval()

    return backbone, tokenizer


def to_additive_mask(bool_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """tree attention maskの`(T, T)` bool tensorを、eager attentionが期待する加算形式に変換する。

    可視: 0.0 / 不可視: `dtype`の最小値（softmax前に加算すると事実上 -inf になる）。
    形は`(1, 1, T, T)`——`Qwen2Model.forward`はこれを4D maskとしてそのまま使う
    （`transformers.masking_utils._preprocess_mask_arguments`: 4Dならそのまま通す）。
    """
    t = bool_mask.shape[0]
    mask = torch.zeros(t, t, dtype=dtype)
    mask.masked_fill_(~bool_mask, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


class JevModel:
    """packer、tree attention mask、position ID、readoutを結合するforwardラッパー。"""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.backbone, self.tokenizer = load_jev_backbone(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.pointer = PointerReadout(hidden_size)
        self.noul = NoulReadout(hidden_size)

    @torch.no_grad()
    def forward(self, request: PackedRequest) -> list[dict]:
        """1リクエストを1回のforwardで処理し、branchごとの結果を返す。"""
        packed = pack(request, self.tokenizer)
        bool_mask = build_tree_attention_mask(packed)
        position_ids = build_position_ids(packed)

        input_ids = torch.tensor([packed.input_ids], dtype=torch.long)
        attention_mask = to_additive_mask(bool_mask, dtype=self.backbone.dtype)
        pos_ids = torch.tensor([position_ids], dtype=torch.long)

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos_ids)
        hidden = out.last_hidden_state[0]  # (T, hidden_size) — batch=1

        results: list[dict] = []
        for branch, span in zip(request.branches, packed.branch_spans):
            h_dec = hidden[span.decision_index]
            if branch.kind == "noul":
                p_true = torch.sigmoid(self.noul(h_dec)).item()
                results.append({"kind": "noul", "p_true": p_true})
                continue

            h_opts = hidden[span.option_end_indices]  # (K, hidden_size)
            z = self.pointer(h_dec, h_opts)
            p = torch.softmax(z, dim=-1)
            entry = {"kind": branch.kind, "options": branch.options, "probabilities": p.tolist()}
            if branch.kind == "score":
                entry["score"] = score_expectation(p)
            results.append(entry)

        return results


def _smoke_check() -> None:
    model = JevModel()

    request = PackedRequest(
        state="Oil prices surged 5% today after OPEC announced production cuts.",
        branches=[
            Branch(
                kind="choice",
                instructions="この記事のトピックを選べ。",
                options=["World", "Sports", "Business", "Sci-Tech"],
            ),
            Branch(kind="noul", instructions="この記事は経済に関するものか？"),
            Branch(
                kind="score",
                instructions="この記事の緊急度を1〜3で評価せよ。",
                options=["低い", "中程度", "高い"],
            ),
        ],
    )

    results = model.forward(request)
    for branch, result in zip(request.branches, results):
        print(f"[{result['kind']}] {branch.instructions!r} -> {result}")

    # 形と確率の妥当性
    choice_result, noul_result, score_result = results
    assert len(choice_result["probabilities"]) == 4
    assert abs(sum(choice_result["probabilities"]) - 1.0) < 1e-4
    assert 0.0 <= noul_result["p_true"] <= 1.0
    assert len(score_result["probabilities"]) == 3
    assert abs(sum(score_result["probabilities"]) - 1.0) < 1e-4
    assert 1.0 <= score_result["score"] <= 3.0

    # 同じ入力を2回forwardして、完全に同じ結果になること（eval+no_grad+dropout=0による決定性）を確認
    results_again = model.forward(request)
    for a, b in zip(results, results_again):
        if a["kind"] == "noul":
            assert a["p_true"] == b["p_true"], "同一入力なのにNoulの出力が再現しない"
        else:
            assert a["probabilities"] == b["probabilities"], "同一入力なのにChoice/Scoreの出力が再現しない"

    print("OK: backbone+packer+tree mask+readoutが結合して1回のforwardで動き、出力の形・確率の和・決定性も確認済み。")


if __name__ == "__main__":
    _smoke_check()
