"""Apple Silicon inference for the System One adapter; no project-local imports."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.activations import swiglu
from transformers import AutoTokenizer

from packing import Branch, LETTERS, PackedRequest, pack_request, position_ids

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class MLXModel:
    def __init__(self, adapter: str | Path = "."):
        adapter = Path(adapter)
        self.model, _ = mlx_load(BASE_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self._merge_lora(adapter)
        self.model.set_dtype(mx.float16)
        mx.eval(self.model.parameters())
        args = self.model.args
        self._tied = args.tie_word_embeddings
        self._inv_freq = 1.0 / (args.rope_theta ** (mx.arange(0, args.head_dim, 2, dtype=mx.float32) / args.head_dim))
        self._labels: dict[int, mx.array] = {}

    def _merge_lora(self, adapter: Path) -> None:
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config.get("use_dora") or config.get("use_rslora"):
            raise ValueError("only plain LoRA adapters are supported")
        scale = config["lora_alpha"] / config["r"]
        weights = mx.load(str(adapter / "adapter_model.safetensors"))
        a_keys = [key for key in weights if key.endswith(".lora_A.weight")]
        if len(a_keys) * 2 != len(weights):
            raise ValueError("adapter contains non-LoRA tensors; not supported")
        for key in a_keys:
            linear = self.model
            for part in key.removeprefix("base_model.model.").removesuffix(".lora_A.weight").split("."):
                linear = linear[int(part)] if part.isdigit() else getattr(linear, part)
            a = weights[key].astype(mx.float32)
            b = weights[key.replace("lora_A", "lora_B")].astype(mx.float32)
            linear.weight = (linear.weight.astype(mx.float32) + scale * (b @ a)).astype(mx.float16)
            mx.eval(linear.weight)

    @staticmethod
    def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        xf = x.astype(mx.float32)
        rotated = mx.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
        return (xf * cos + rotated * sin).astype(x.dtype)

    def _forward(self, ids: mx.array, positions: mx.array, mask: mx.array, decisions: mx.array) -> mx.array:
        inner = self.model.model
        length = ids.shape[0]
        freqs = positions.astype(mx.float32)[:, None] * self._inv_freq[None]
        angles = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = mx.cos(angles)[None, None], mx.sin(angles)[None, None]
        mask = mask[None, None]
        h = inner.embed_tokens(ids[None])
        for layer in inner.layers:
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).reshape(1, length, attn.n_heads, -1)).transpose(0, 2, 1, 3)
            k = attn.k_norm(attn.k_proj(x).reshape(1, length, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = attn.v_proj(x).reshape(1, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
            h = h + attn.o_proj(out.transpose(0, 2, 1, 3).reshape(1, length, -1))
            x = layer.post_attention_layernorm(h)
            h = h + layer.mlp.down_proj(swiglu(layer.mlp.gate_proj(x), layer.mlp.up_proj(x)))
        h = inner.norm(h[0, decisions])
        return inner.embed_tokens.as_linear(h) if self._tied else self.model.lm_head(h)

    def _label_ids(self, count: int) -> mx.array:
        if count not in self._labels:
            ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS[:count]]
            if any(len(token) != 1 for token in ids):
                raise ValueError("answer labels must each be one token")
            self._labels[count] = mx.array([token[0] for token in ids])
        return self._labels[count]


def load_model(adapter: str | Path = ".") -> tuple[MLXModel, object]:
    model = MLXModel(adapter)
    return model, model.tokenizer


def predict_request(model: MLXModel, tokenizer, request: PackedRequest) -> list[list[float]]:
    """Return canonical-order distributions in one tree-masked forward pass."""
    ids, state_len, spans = pack_request(request, tokenizer)
    branches = [-1] * len(ids)
    for index, (start, end) in enumerate(spans):
        branches[start:end + 1] = [index] * (end - start + 1)
    branch_ids = mx.array(branches)
    t = len(ids)
    mask = mx.tril(mx.ones((t, t), dtype=mx.bool_)) & (
        (branch_ids == -1)[None, :] | (branch_ids[:, None] == branch_ids[None, :])
    )
    logits = model._forward(
        mx.array(ids), mx.array(position_ids(state_len, spans)), mask, mx.array([end for _, end in spans])
    )
    probs = [
        mx.softmax(logits[index, model._label_ids(len(branch.options or ["No", "Yes"]))].astype(mx.float32), axis=-1)
        for index, branch in enumerate(request.branches)
    ]
    mx.eval(probs)
    return [p.tolist() for p in probs]


if __name__ == "__main__":
    model, tokenizer = load_model(Path(__file__).resolve().parent)
    request = PackedRequest(
        state="A customer received a replacement device after the first one stopped charging.",
        branches=[
            Branch("choice", "Which issue is described?", ["Delivery delay", "Charging failure", "Billing error"]),
            Branch("noul", "Does the record mention a replacement?"),
        ],
    )
    for branch, probabilities in zip(request.branches, predict_request(model, tokenizer, request)):
        print(branch.kind, probabilities)
