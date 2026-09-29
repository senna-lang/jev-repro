"""Qwen2.5-0.5Bをロードし、最終隠れ状態を返す`Qwen2Model`バックボーンだけを公開する。

語彙予測用のLM headは使わず、sequence packer・attention mask・readoutが利用する
トークン表現を提供する。バックボーンの重みは変更しない。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Model

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def load_backbone(model_name: str = MODEL_NAME) -> tuple[Qwen2Model, AutoTokenizer]:
    """`Qwen/Qwen2.5-0.5B` をロードし、LM head を捨てて `Qwen2Model` だけを返す。

    `AutoModelForCausalLM` としてロードしたあと `.model` 属性（`Qwen2Model` 本体、
    最終隠れ状態までを計算する部分）だけを取り出す。`lm_head`（語彙への射影）は
    Jev repro では使わない（readout は pointer 方式で別途実装するため）。

    重みは一切書き換えない。事前学習済みのまま returns する。
    """
    causal_lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    backbone = causal_lm.model
    assert isinstance(backbone, Qwen2Model), (
        f"想定外の型: {type(backbone)}。Qwen2Model が取り出せていない。"
    )
    backbone.eval()

    return backbone, tokenizer


def _smoke_check(model_name: str = MODEL_NAME) -> None:
    """ロード後、最終隠れ状態が`(batch, seq_len, hidden_size)`で出ることを確認する。"""
    backbone, tokenizer = load_backbone(model_name)

    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        out = backbone(**inputs)

    hidden = out.last_hidden_state
    n_params = sum(p.numel() for p in backbone.parameters())

    print(f"model: {model_name}")
    print(f"backbone class: {type(backbone).__name__}")
    print(f"hidden_size (config): {backbone.config.hidden_size}")
    print(f"num_hidden_layers (config): {backbone.config.num_hidden_layers}")
    print(f"backbone param count: {n_params:,}")
    print(f"input token ids: {inputs['input_ids'].tolist()}")
    print(f"last_hidden_state.shape: {tuple(hidden.shape)}")

    assert hidden.shape == (1, inputs["input_ids"].shape[1], backbone.config.hidden_size), (
        "last_hidden_state の形が (batch, seq_len, hidden_size) になっていない"
    )
    print("OK: Qwen2Model 部分だけを使って forward が通り、形も期待通り。")


if __name__ == "__main__":
    _smoke_check()
