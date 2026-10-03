"""Mac (Apple Silicon) inference for the E2 LoRA on MLX.

Same contract as `e2.evaluate_lora.predict_request` (packing, tree mask, position ids, answer-letter
readout, canonical-order averaging). Only the execution backend differs:

- Weights: Qwen3-4B base + LoRA merged once at load (W + (alpha/r) * B @ A, accumulated in fp32, cast to fp16).
- Forward: mlx_lm Qwen3 modules with our own layer loop. mlx_lm's `Model.__call__` only supports causal
  masks and contiguous RoPE offsets, so the 4D tree mask and per-branch position ids are applied here.
  RoPE follows HF's rotate_half convention with cos/sin computed in fp32.
- Readout: final norm + tied lm_head only at decision positions.

e2/ is not modified; training and CUDA evaluation keep using PyTorch.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.models.activations import swiglu
from mlx_lm.tuner.lora import LoRALinear
from transformers import AutoTokenizer

from e2.backbone import get_backbone, read_base_record
from e2.bonsai_parity import TARGETS
from e2.chat_packer import pack_chat_request
from e2.evaluate_lora import _label_ids
from model.packer import Branch, PackedRequest
from model.tree_mask import build_position_ids, build_tree_attention_mask

MODEL = get_backbone("qwen").train_repo

DEFAULT_ADAPTER = Path("checkpoints/e2/lr-1e-4/epoch-3-quarter-3")


class JevMLX:
    def __init__(self, adapter: Path = DEFAULT_ADAPTER, dtype=mx.float16, backbone: str = "qwen"):
        spec = get_backbone(backbone)
        if spec.name == "qwen":
            self.model, _ = mlx_load(MODEL)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
            self._merge_lora(adapter, dtype)
            self.model.set_dtype(dtype)
        else:
            self._load_packed(spec, adapter, dtype)
        mx.eval(self.model.parameters())
        args = self.model.args
        self._tied = args.tie_word_embeddings
        self._inv_freq, self._rope_factor = self._rope_frequencies(spec)
        self._labels: dict[int, mx.array] = {}

    def _rope_frequencies(self, spec) -> tuple[mx.array, float]:
        args = self.model.args
        if not args.rope_scaling:
            return 1.0 / (args.rope_theta ** (mx.arange(0, args.head_dim, 2, dtype=mx.float32) / args.head_dim)), 1.0
        from transformers import AutoConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        config = AutoConfig.from_pretrained(spec.infer_repo, revision=spec.infer_revision)
        inv_freq, factor = ROPE_INIT_FUNCTIONS[args.rope_scaling["rope_type"]](config, "cpu")
        return mx.array(inv_freq.float().numpy()), float(factor)

    def _load_packed(self, spec, adapter: Path, dtype) -> None:
        """Frozen packed base + fp32 LoRA sidecars (no merge, no requantize, no dense weights)."""
        read_base_record(adapter, spec)
        config = json.loads((adapter / "adapter_config.json").read_text())
        if (config["r"], config["lora_alpha"], config.get("bias")) != (16, 32, "none") or config.get("use_dora") or config.get("use_rslora") \
                or config.get("rank_pattern") or config.get("alpha_pattern") or config.get("modules_to_save") \
                or set(config["target_modules"]) != {t.split(".")[1] for t in TARGETS}:
            raise ValueError(f"{adapter}: adapter_config.json is not the E2 LoRA (r=16, alpha=32, all seven projections)")
        self.model, _ = mlx_load(spec.infer_repo, revision=spec.infer_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.train_repo, revision=spec.train_revision)
        self.model.set_dtype(dtype)  # float parts only; packed uint32 weights are skipped. Must precede the sidecars.
        weights = mx.load(str(adapter / "adapter_model.safetensors"))
        layers = self.model.model.layers
        names = [(i, t) for i in range(len(layers)) for t in TARGETS]
        expected = {f"base_model.model.model.layers.{i}.{t}.lora_{ab}.weight" for i, t in names for ab in "AB"}
        if set(weights) != expected:
            raise ValueError(f"{adapter}: LoRA tensors do not match the 252 expected targets")
        scale = config["lora_alpha"] / config["r"]
        for i, t in names:
            parent_name, leaf = t.split(".")
            parent = getattr(layers[i], parent_name)
            base = getattr(parent, leaf)
            if not isinstance(base, nn.QuantizedLinear):
                raise ValueError(f"layers.{i}.{t} is {type(base).__name__}, expected QuantizedLinear")
            sidecar = LoRALinear.from_base(base, r=config["r"], dropout=0.0, scale=scale)
            prefix = f"base_model.model.model.layers.{i}.{t}"
            sidecar.lora_a = weights[f"{prefix}.lora_A.weight"].astype(mx.float32).T
            sidecar.lora_b = weights[f"{prefix}.lora_B.weight"].astype(mx.float32).T
            setattr(parent, leaf, sidecar)

    def _merge_lora(self, adapter: Path, dtype) -> None:
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
            linear.weight = (linear.weight.astype(mx.float32) + scale * (b @ a)).astype(dtype)
            mx.eval(linear.weight)

    def _rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        xf = x.astype(mx.float32)
        rotated = mx.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
        return (xf * cos + rotated * sin).astype(x.dtype)

    def forward(self, input_ids: mx.array, position_ids: mx.array, mask: mx.array, decisions: mx.array) -> mx.array:
        """input_ids/position_ids: (T,), mask: (T, T) bool with True = may attend, decisions: (N,). Returns (N, vocab)."""
        inner = self.model.model
        length = input_ids.shape[0]
        freqs = position_ids.astype(mx.float32)[:, None] * self._inv_freq[None]
        angles = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = mx.cos(angles)[None, None], mx.sin(angles)[None, None]
        if self._rope_factor != 1.0:  # YaRN attention factor, applied to cos/sin as in HF
            cos, sin = cos * self._rope_factor, sin * self._rope_factor
        mask = mask[None, None]
        h = inner.embed_tokens(input_ids[None])
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
            self._labels[count] = mx.array(_label_ids(self.tokenizer, count, "cpu").tolist())
        return self._labels[count]

    def predict_request(self, state: str, branches: list[Branch], orders: list[list[list[int]]] | None = None) -> list[list[float]]:
        """Canonical-order distributions per branch. `orders=None` is the single canonical order;
        pass `e2.evaluate_lora.option_orders(...)` per branch for order averaging.
        Noul branches carry options ["No", "Yes"] as in e2 data."""
        if orders is None:
            orders = [[list(range(len(branch.options)))] for branch in branches]
        expanded = [
            (index, order, Branch(branch.kind, branch.instructions, [branch.options[i] for i in order]))
            for index, (branch, branch_orders) in enumerate(zip(branches, orders))
            for order in branch_orders
        ]
        packed = pack_chat_request(PackedRequest(state, [branch for _, _, branch in expanded]), self.tokenizer)
        logits = self.forward(
            mx.array(packed.input_ids),
            mx.array(build_position_ids(packed)),
            mx.array(build_tree_attention_mask(packed).numpy()),
            mx.array([span.decision_index for span in packed.branch_spans]),
        )
        probs = [
            mx.softmax(logits[row, self._label_ids(len(order))].astype(mx.float32), axis=-1)
            for row, (_, order, _) in enumerate(expanded)
        ]
        mx.eval(probs)
        predictions: list[list[list[float]]] = [[] for _ in branches]
        for p, (index, order, _) in zip(probs, expanded):
            ordered = p.tolist()
            canonical = [0.0] * len(order)
            for position, original in enumerate(order):
                canonical[original] = ordered[position]
            predictions[index].append(canonical)
        return [[sum(values) / len(values) for values in zip(*per_order)] for per_order in predictions]
