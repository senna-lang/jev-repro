"""BoolQ（yes/no読解）でNoulReadoutを訓練する。

これまでこのプロジェクトのどの訓練スクリプトも`NoulReadout`を訓練していなかった
（`experiments/train.py`のdocstring「バックボーンとNoulReadoutは更新しない」の通り、
yes/noの訓練データが無かったため）。Gold V1評価でNoulのthreshold_accuracyが実行の
たびに変わっていたのは、この未訓練のランダム初期化パラメータを毎回測っていたのが原因
（`results/`未記載、セッション内調査で判明）。

訓練前にGate 0スタイルのzero-shot Q1チェック（`experiments/boolq_probe.py`）を実施済み:
accuracy=0.700, p<0.0001, n=120 — 凍結backboneに既にyes/no判別の手がかりがあることを確認
してから着手している。

`model.pointer`は一切触らない（Choice/Scoreの既存性能に影響しないことを保証するため、
optimizerには`model.noul.parameters()`のみを登録する）。

使い方:
    python3 -m experiments.train_noul
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from data.boolq import load_boolq_stratified, to_packed_example
from legacy.model_readout.forward import JevModel, to_additive_mask
from model.packer import pack
from model.tree_mask import build_position_ids, build_tree_attention_mask

EPOCHS = 3
SEED = 0
TRAIN_PER_LABEL = 300  # 2 labels x 300 = 600件
HELD_OUT_PER_LABEL = 100  # 2 labels x 100 = 200件（train部分と重複しないoffsetから取る）


@dataclass
class NoulExample:
    """1件のNoul branchリクエストと正解ラベル（yes=True / no=False）。"""

    request: object
    label: bool


def _build_examples(n_per_label: int, offset: int, seed: int) -> list[NoulExample]:
    rows = load_boolq_stratified(n_per_label, seed=seed, offset=offset)
    examples = []
    for r in rows:
        request, label = to_packed_example(r["question"], r["passage"], r["answer"])
        examples.append(NoulExample(request=request, label=label))
    return examples


def _noul_logit(model: JevModel, example: NoulExample, *, requires_grad: bool) -> torch.Tensor:
    """1件のNoulExampleから、NoulReadoutのlogit（スカラー、未sigmoid）を計算する。

    backboneは常に`torch.no_grad()`（Noul訓練でもbackboneは凍結のまま、
    `experiments/train.py::_choice_logits`のbackbone_grad=False相当）。
    """
    packed = pack(example.request, model.tokenizer)
    bool_mask = build_tree_attention_mask(packed)
    position_ids = build_position_ids(packed)

    input_ids = torch.tensor([packed.input_ids], dtype=torch.long)
    attention_mask = to_additive_mask(bool_mask, dtype=model.backbone.dtype)
    pos_ids = torch.tensor([position_ids], dtype=torch.long)

    with torch.no_grad():
        out = model.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos_ids)
    hidden = out.last_hidden_state[0]

    span = packed.branch_spans[0]
    h_dec = hidden[span.decision_index]

    with torch.set_grad_enabled(requires_grad):
        return model.noul(h_dec)


def _init_noul_from_lm_head(model: JevModel, model_name: str) -> None:
    """NoulReadoutの線形層を、事前学習済みLM headの" Yes"/" No"方向で初期化する。

    ランダム初期化からの訓練（600件・3epoch）はheld-out accuracy=0.5300で、readoutなしの
    zero-shot probe（LM head丸ごと、accuracy=0.700, n=120, `experiments/boolq_probe.py`）
    に遠く及ばなかった。NoulReadoutは`h_dec`から1方向を新規に学習する必要があるが、
    896次元に対し600件では学習不足になりやすい。LM headの出力層は
    `logit_yes - logit_no = (w_yes - w_no)·h_dec`という、既にzero-shotで機能する方向を
    embedding行列内に持っている。この方向でwarm-startし、そこから微調整する。
    """
    from transformers import AutoModelForCausalLM

    causal_lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    lm_head_weight = causal_lm.get_output_embeddings().weight.detach()

    yes_id = model.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id = model.tokenizer.encode(" No", add_special_tokens=False)[0]
    direction = lm_head_weight[yes_id] - lm_head_weight[no_id]  # (hidden_size,)

    with torch.no_grad():
        model.noul.linear.weight.copy_(direction.unsqueeze(0))
        model.noul.linear.bias.zero_()

    del causal_lm


def train_noul_readout(
    model: JevModel,
    train_examples: list[NoulExample],
    epochs: int = 3,
    lr: float = 1e-3,
    log_every: int = 100,
    seed: int = 0,
) -> None:
    """`NoulReadout`（線形層）をbinary cross-entropyで訓練する。backbone・pointerは凍結のまま。"""
    optimizer = torch.optim.Adam(model.noul.parameters(), lr=lr)
    examples = list(train_examples)

    for epoch in range(epochs):
        torch.manual_seed(seed + epoch)
        import random

        random.Random(seed + epoch).shuffle(examples)
        epoch_loss = 0.0
        epoch_correct = 0
        start = time.perf_counter()

        for i, example in enumerate(examples):
            optimizer.zero_grad()
            logit = _noul_logit(model, example, requires_grad=True)
            target = torch.tensor(1.0 if example.label else 0.0)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += int((logit.item() > 0.0) == example.label)

            if log_every and (i + 1) % log_every == 0:
                print(f"  epoch {epoch} [{i + 1}/{len(examples)}] running_loss={epoch_loss / (i + 1):.4f}")

        elapsed = time.perf_counter() - start
        print(f"epoch {epoch}: loss={epoch_loss / len(examples):.4f}, "
              f"train_acc={epoch_correct / len(examples):.4f}, {elapsed:.1f}s")


@torch.no_grad()
def evaluate_noul(model: JevModel, eval_examples: list[NoulExample]) -> dict:
    """threshold accuracy（p_true>0.5 vs label）とbrierを計算する。"""
    correct = 0
    brier_sum = 0.0
    for example in eval_examples:
        logit = _noul_logit(model, example, requires_grad=False)
        p_true = float(torch.sigmoid(logit))
        target = 1.0 if example.label else 0.0
        correct += int((p_true > 0.5) == example.label)
        brier_sum += (p_true - target) ** 2
    n = len(eval_examples)
    return {"accuracy": correct / n, "brier": brier_sum / n, "n": n}


def _main() -> None:
    print(f"Noul訓練: BoolQ, epochs={EPOCHS}, train={2 * TRAIN_PER_LABEL}件, held_out={2 * HELD_OUT_PER_LABEL}件")

    train_examples = _build_examples(TRAIN_PER_LABEL, offset=0, seed=SEED)
    held_out_examples = _build_examples(HELD_OUT_PER_LABEL, offset=TRAIN_PER_LABEL, seed=SEED)

    model = JevModel(option_pooling="mean")  # pointerは使わないのでpoolingは無関係、既存最良と揃えておく

    random_init = evaluate_noul(model, held_out_examples)
    print(f"訓練前（ランダム初期化）: accuracy={random_init['accuracy']:.4f}, brier={random_init['brier']:.4f}")

    _init_noul_from_lm_head(model, model.backbone.config._name_or_path)
    warm_start = evaluate_noul(model, held_out_examples)
    print(f"warm-start直後（微調整前）: accuracy={warm_start['accuracy']:.4f}, brier={warm_start['brier']:.4f}")

    train_noul_readout(model, train_examples, epochs=EPOCHS, lr=1e-4, seed=SEED)

    after = evaluate_noul(model, held_out_examples)
    print(f"訓練後: accuracy={after['accuracy']:.4f}, brier={after['brier']:.4f}")

    stopping_condition = after["accuracy"] > 0.6 and after["accuracy"] >= warm_start["accuracy"]
    print(f"停止条件（held-out accuracy > 0.6 かつ warm-startを下回らない）: {stopping_condition}")

    checkpoint_path = Path("checkpoints/noul_boolq_e3.pt")
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "noul_state_dict": model.noul.state_dict(),
            "datasets": ["boolq"],
            "epochs": EPOCHS,
            "held_out_accuracy": after["accuracy"],
            "held_out_brier": after["brier"],
            "warm_start_accuracy": warm_start["accuracy"],
            "warm_start_brier": warm_start["brier"],
        },
        checkpoint_path,
    )
    print(f"保存: {checkpoint_path}")


if __name__ == "__main__":
    _main()
