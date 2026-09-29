"""packed request形式の中で、信号がどこで失われるのかを切り分ける（診断の記録）。

`experiments/backbone_probe.py`で、凍結Qwen2.5-0.5BはLM head zero-shotなら
sst5最遠ペア二値 0.900、公式severity argmax 4/5一致を出すことが分かった。つまり
**信号はbackboneの中にある**。一方で当時の特殊トークン形式（`<STATE_END>`/`<SEP>`/
`<DECISION>`）の訓練済みpointer readoutは同じタスクで0.525（バイアス校正しても0.795）、
severityは1/5しか当たらなかった。原因の切り分けの結果、地の文形式（選択肢をコンマ区切りで
instructions内に埋め込み、末尾を"Answer:"で締める）が0.910・severity tau+0.600まで回復し、
`model/packer.py::pack()`はこの形式に移行済み（詳細: `results/2026-09-22_score_diagnosis.md`）。
このスクリプトは移行前の中間案（自然文マーカー／ラベル+解答キュー）を現行`pack()`と比較する
ための記録として残す。

失われる場所の候補は3つあった:

  (P) packing。tree attention maskや位置IDの組み方で、決定トークンが選択肢や本文を
      うまく見られていない。
  (Q) パラメータ化。`z_i = (W_q LN(h_dec))·(W_k LN(h_opt_i))/√d`という学習済み双線形形式
      そのものが、必要な比較を表現できていない。
  (R) 訓練。表現力はあるが、5500例・凍結backboneという条件で悪い解に落ちている
      （選択肢固有バイアス支配）。

これらはpacked requestの同じ隠れ状態から、readoutだけを差し替えて比べれば分離できる:

  1. **logit lens**: `h_dec`をbackboneのLM headに通し、各選択肢の先頭トークンの確率を読む。
     pointer readoutを完全に迂回するので、これが効けば(P)は無罪。
  2. **恒等初期化pointer（未訓練）**: `W_q = W_k = I`、つまりLayerNorm後の素の内積
     `h_dec·h_opt_i/√d`。学習を一切経ないので、これが効けば(Q)も無罪で、(R)=訓練が犯人。
  3. **訓練済みpointer**: 参照（旧特殊トークン形式で訓練した重みをそのまま流用。
     現行`pack()`に対しては分布外になる点に注意）。

使い方:
    python3 -m experiments.readout_probe checkpoints/pointer_diverse5_e3.pt
"""

from __future__ import annotations

import math
import sys

import torch
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM

from data.sst5 import INSTRUCTION_VARIANTS, LEVEL_ORDER, load_sst5
from legacy.experiments.backbone_probe import SEVERITY_STATES
from legacy.experiments.official_compare import SEVERITY_LEVELS
from legacy.model_readout.backbone import MODEL_NAME
from legacy.model_readout.forward import JevModel, to_additive_mask
from model.packer import Branch, BranchSpan, PackedRequest, PackedSequence, pack
from model.tree_mask import build_position_ids, build_tree_attention_mask

SEVERITY_QUESTION = "How severe is the reported issue?"

# 移行前に検討した中間フォーマット2案。`decision`は当時"<DECISION>"という
# `resize_token_embeddings`で足した未学習の埋め込みトークンだったため、自然文
# "Answer:"に置き換えた案（詳細は上記モジュールdocstring）。
TEXT_MARKERS = {"state_end": "\n\n", "sep": "\n", "decision": "\nAnswer:"}


def pack_with_text_markers(request: PackedRequest, tokenizer) -> PackedSequence:
    """`pack`と同じ構造を、特殊トークンの代わりに自然文マーカーで組む。

    境界（state_len / option_end_indices / decision_index）の意味は`pack`と同一なので、
    tree maskもposition IDもそのまま使える。違いは区切りのトークンだけ。
    """
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids: list[int] = encode(request.state) + encode(TEXT_MARKERS["state_end"])
    state_len = len(input_ids)

    branch_spans: list[BranchSpan] = []
    for branch in request.branches:
        branch_start = len(input_ids)
        input_ids.extend(encode(branch.instructions))

        option_end_indices: list[int] = []
        for option in branch.options:
            input_ids.extend(encode(option))
            option_end_indices.append(len(input_ids) - 1)
            input_ids.extend(encode(TEXT_MARKERS["sep"]))

        input_ids.extend(encode(TEXT_MARKERS["decision"]))
        decision_index = len(input_ids) - 1

        branch_spans.append(
            BranchSpan(
                kind=branch.kind,
                start=branch_start,
                option_start_indices=[],  # 旧フォーマット: 参考実装、mean-poolは使わない
                option_end_indices=option_end_indices,
                decision_index=decision_index,
            )
        )

    return PackedSequence(input_ids=input_ids, state_len=state_len, branch_spans=branch_spans)


def pack_with_labeled_options(request: PackedRequest, tokenizer) -> PackedSequence:
    """`pack_with_text_markers`にさらに、選択肢へのラベル付けと明示的な解答キューを足す。

    自然文マーカーだけでは回復しなかった（logit lens 0.500）。残る疑いは選択肢が
    「本文の続き」のように裸で並んでいること（ラベル無し）と、decision位置の直前に
    「これから選ぶ」という明示的な合図が無いこと。両方を足して切り分ける。
    """
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids: list[int] = encode(request.state) + encode("\n\n")
    state_len = len(input_ids)

    branch_spans: list[BranchSpan] = []
    for branch in request.branches:
        branch_start = len(input_ids)
        input_ids.extend(encode(branch.instructions))
        input_ids.extend(encode("\n"))

        option_end_indices: list[int] = []
        for i, option in enumerate(branch.options):
            input_ids.extend(encode(f"Option {i + 1}: {option}"))
            option_end_indices.append(len(input_ids) - 1)
            input_ids.extend(encode("\n"))

        input_ids.extend(encode("Answer:"))
        decision_index = len(input_ids) - 1

        branch_spans.append(
            BranchSpan(
                kind=branch.kind,
                start=branch_start,
                option_start_indices=[],  # 旧フォーマット: 参考実装、mean-poolは使わない
                option_end_indices=option_end_indices,
                decision_index=decision_index,
            )
        )

    return PackedSequence(input_ids=input_ids, state_len=state_len, branch_spans=branch_spans)


@torch.no_grad()
def _hidden_states(model: JevModel, request: PackedRequest,
                   packer=pack) -> tuple[torch.Tensor, torch.Tensor]:
    """packed requestを1回forwardし、最初のbranchの`h_dec`と`h_opts`を返す。"""
    packed = packer(request, model.tokenizer)
    bool_mask = build_tree_attention_mask(packed)
    position_ids = build_position_ids(packed)

    out = model.backbone(
        input_ids=torch.tensor([packed.input_ids], dtype=torch.long),
        attention_mask=to_additive_mask(bool_mask, dtype=model.backbone.dtype),
        position_ids=torch.tensor([position_ids], dtype=torch.long),
    )
    hidden = out.last_hidden_state[0]
    span = packed.branch_spans[0]
    return hidden[span.decision_index], hidden[span.option_end_indices]


@torch.no_grad()
def _logit_lens(model: JevModel, lm_head: torch.nn.Module, norm: torch.nn.Module,
                h_dec: torch.Tensor, options: list[str]) -> list[float]:
    """`h_dec`をLM headに通し、各選択肢の先頭トークンのlog確率を読む。

    最終layer normを通してからlm_headに掛ける（`Qwen2ForCausalLM`と同じ経路。
    `JevModel`が使う`Qwen2Model`の`last_hidden_state`は既にnorm適用済みだが、
    ここでは経路を明示するために同じ関数を参照している）。

    選択肢の先頭トークンが衝突していると（例: `very negative`と`very positive`はどちらも
    ` very`）、比較が常に引き分けになりaccuracyがちょうど0.500に見える。実際に一度これで
    誤った結論を出したので、衝突は例外にする。
    """
    logits = lm_head(h_dec)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = [
        model.tokenizer(" " + option.strip(), add_special_tokens=False)["input_ids"][0]
        for option in options
    ]
    assert len(set(token_ids)) == len(token_ids), (
        f"選択肢の先頭トークンが衝突している: {options} -> {token_ids}。"
        "logit lensでは区別できないので、先頭が異なる選択肢を使うこと。"
    )
    return [float(log_probs[i]) for i in token_ids]


def _identity_pointer(h_dec: torch.Tensor, h_opts: torch.Tensor) -> list[float]:
    """W_q = W_k = I の未訓練pointer。LayerNorm後の素の内積。"""
    dim = h_dec.shape[-1]
    layer_norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
    q = layer_norm(h_dec)
    k = layer_norm(h_opts)
    return (k @ q / math.sqrt(dim)).tolist()


@torch.no_grad()
def _trained_pointer(model: JevModel, h_dec: torch.Tensor, h_opts: torch.Tensor) -> list[float]:
    return model.pointer(h_dec, h_opts).tolist()


def _softmax(scores: list[float]) -> list[float]:
    highest = max(scores)
    weights = [math.exp(s - highest) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


READOUTS = ("logit lens", "恒等初期化pointer(未訓練)", "訓練済みpointer")
PACKERS = (
    ("自然文マーカー(旧・中間案)", pack_with_text_markers),
    ("ラベル+解答キュー(旧・中間案)", pack_with_labeled_options),
    ("地の文（現行pack()）", pack),
)


def _all_readouts(model: JevModel, lm_head, norm, request: PackedRequest,
                  options: list[str], packer) -> dict[str, list[float]]:
    h_dec, h_opts = _hidden_states(model, request, packer)
    return {
        "logit lens": _softmax(_logit_lens(model, lm_head, norm, h_dec, options)),
        "恒等初期化pointer(未訓練)": _softmax(_identity_pointer(h_dec, h_opts)),
        "訓練済みpointer": _softmax(_trained_pointer(model, h_dec, h_opts)),
    }


def probe_extreme_pair(model: JevModel, lm_head, norm, n_per_class: int = 50) -> None:
    """sst5の最遠ペア（very negative / very positive）二択を、readoutを差し替えて解く。"""
    print(f"\n=== sst5最遠ペア二択（backbone LM head zero-shotでは0.900） ===")
    dataset = load_sst5("validation")
    examples: list[tuple[str, int]] = []
    negatives = positives = 0
    for row in dataset:
        label = int(row["label"])
        if label == 0 and negatives < n_per_class:
            examples.append((row["text"], 0))
            negatives += 1
        elif label == 4 and positives < n_per_class:
            examples.append((row["text"], 1))
            positives += 1
        if negatives >= n_per_class and positives >= n_per_class:
            break

    # `backbone_probe`の対照（自然文プロンプトで0.900）と同じ語を使う。`very negative`/
    # `very positive`は先頭トークンが両方` very`で、logit lensでは区別できない。
    options = ["negative", "positive"]
    branch = Branch(
        kind="score",
        instructions="Is the sentiment of this review negative or positive?",
        options=options,
    )
    print(f"  n={len(examples)} (negative={negatives}, positive={positives}, チャンス=0.500)")

    for packer_name, packer in PACKERS:
        correct = {name: 0 for name in READOUTS}
        picks = {name: [0, 0] for name in READOUTS}
        for text, gold in examples:
            probs_by_readout = _all_readouts(
                model, lm_head, norm, PackedRequest(state=text, branches=[branch]), options, packer
            )
            for name, probs in probs_by_readout.items():
                pick = int(probs[1] > probs[0])
                correct[name] += int(pick == gold)
                picks[name][pick] += 1

        print(f"  [{packer_name}]")
        for name in READOUTS:
            print(f"    {name:<24} accuracy={correct[name] / len(examples):.3f}  "
                  f"選択の偏り={picks[name]}")


def probe_severity(model: JevModel, lm_head, norm) -> None:
    """公式severity表（K=3・別ドメイン）を、readoutを差し替えて解く。"""
    print(f"\n=== 公式severity表 K=3（backbone LM head zero-shotではargmax 4/5一致） ===")
    official = [score for _, score in SEVERITY_STATES]
    branch = Branch(kind="score", instructions=SEVERITY_QUESTION, options=list(SEVERITY_LEVELS))

    for packer_name, packer in PACKERS:
        positions = {name: [] for name in READOUTS}
        argmaxes = {name: [] for name in READOUTS}
        for state, _ in SEVERITY_STATES:
            probs_by_readout = _all_readouts(
                model, lm_head, norm, PackedRequest(state=state, branches=[branch]),
                list(SEVERITY_LEVELS), packer,
            )
            for name, probs in probs_by_readout.items():
                positions[name].append(sum(i * p for i, p in enumerate(probs)))
                argmaxes[name].append(max(range(len(probs)), key=lambda i: probs[i]))

        print(f"  [{packer_name}]")
        for name in READOUTS:
            tau, _ = kendalltau(official, positions[name])
            matches = sum(int(a == b) for a, b in zip(argmaxes[name], [0, 1, 1, 1, 2]))
            print(f"    {name:<24} tau={tau:+.3f}  argmax={argmaxes[name]}  一致={matches}/5")
    print(f"    {'(公式)':<24}          argmax={[0, 1, 1, 1, 2]}")


def _main() -> None:
    torch.manual_seed(0)
    model = JevModel()
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    model.pointer.load_state_dict(ckpt["state_dict"])
    print(f"readout: {sys.argv[1]} (epochs={ckpt.get('epochs')})")

    # LM headだけを別途ロードする（JevModelはbackbone本体しか持たない）。
    causal_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    causal_lm.eval()
    lm_head, norm = causal_lm.lm_head, causal_lm.model.norm

    probe_extreme_pair(model, lm_head, norm)
    probe_severity(model, lm_head, norm)


if __name__ == "__main__":
    _main()
