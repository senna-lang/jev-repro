"""固定したバックボーン上でAG News分類用のPointerReadoutを訓練する。

各例をtree attention mask付きでforwardし、Choiceの選択肢logitにcross-entropyを適用する。
訓練前後のheld-outデータに対する正解率とECEを測定する。バックボーンとNoulReadoutは更新しない。
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from data.ag_news import AGNewsExample, eval_split, to_packed_example, train_subset
from model.forward import JevModel, to_additive_mask
from model.packer import pack
from model.tree_mask import build_position_ids, build_tree_attention_mask


def _choice_logits(model: JevModel, example: AGNewsExample, *, requires_grad: bool) -> torch.Tensor:
    """1件のAGNewsExample（Choice branch1つ）から、pointer readoutのlogit z（未softmax）を計算する。

    backboneは常に`torch.no_grad()`（重みを更新しないため）。`requires_grad`は「pointer readout側の
    計算に勾配を通すか」を切り替えるだけ（訓練時True、評価時False＝`evaluate`側で`@torch.no_grad()`済み
    なのでここは実質常にFalseで呼ばれる）。
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
    h_opts = hidden[span.option_end_indices]

    with torch.set_grad_enabled(requires_grad):
        return model.pointer(h_dec, h_opts)


def train_readout(
    model: JevModel,
    train_examples: list[AGNewsExample],
    epochs: int = 3,
    lr: float = 1e-3,
    log_every: int = 100,
) -> None:
    """`PointerReadout`（W_q, W_k）だけをcross-entropyで訓練する。backboneは凍結。"""
    optimizer = torch.optim.Adam(model.pointer.parameters(), lr=lr)
    examples = list(train_examples)

    for epoch in range(epochs):
        random.shuffle(examples)
        epoch_loss = 0.0
        epoch_correct = 0
        start = time.perf_counter()

        for i, example in enumerate(examples):
            optimizer.zero_grad()
            z = _choice_logits(model, example, requires_grad=True)
            loss = F.cross_entropy(z.unsqueeze(0), torch.tensor([example.label]))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += int(z.argmax().item() == example.label)

            if log_every and (i + 1) % log_every == 0:
                print(f"  epoch {epoch} [{i + 1}/{len(examples)}] running_loss={epoch_loss / (i + 1):.4f}")

        elapsed = time.perf_counter() - start
        print(f"epoch {epoch}: loss={epoch_loss / len(examples):.4f}, "
              f"train_acc={epoch_correct / len(examples):.4f}, {elapsed:.1f}s")


def expected_calibration_error(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """10分割のECE（Expected Calibration Error）。

    各予測のtop-1確信度を10個のbinに割り振り、bin内の平均確信度とbin内の正解率の差の絶対値を、
    binのサンプル数で重み付けして平均する。
    """
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    n = len(confidences)
    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        bin_conf = sum(c for c, _ in bin_items) / len(bin_items)
        bin_acc = sum(1 for _, correct in bin_items if correct) / len(bin_items)
        ece += (len(bin_items) / n) * abs(bin_conf - bin_acc)
    return ece


@torch.no_grad()
def evaluate(model: JevModel, eval_examples: list[AGNewsExample]) -> dict:
    """正解率とECE（10分割）を計算する。"""
    confidences = []
    corrects = []
    for example in eval_examples:
        result = model.forward(example.request)[0]
        probabilities = result["probabilities"]
        pred = max(range(len(probabilities)), key=lambda k: probabilities[k])
        confidences.append(probabilities[pred])
        corrects.append(pred == example.label)

    accuracy = sum(corrects) / len(corrects)
    ece = expected_calibration_error(confidences, corrects)
    return {"accuracy": accuracy, "ece": ece, "n": len(eval_examples)}


def _build_examples(dataset) -> list[AGNewsExample]:
    label_names = dataset.features["label"].names
    return [to_packed_example(ex["text"], ex["label"], label_names) for ex in dataset]


def _main(train_size: int = 2000, eval_size: int = 500, epochs: int = 3) -> None:
    print(f"train_size={train_size}, eval_size={eval_size}, epochs={epochs}")
    model = JevModel()

    train_examples = _build_examples(train_subset(train_size))
    eval_examples = _build_examples(eval_split(eval_size))

    print("\n=== 訓練前 ===")
    before = evaluate(model, eval_examples)
    print(f"accuracy={before['accuracy']:.4f}, ECE={before['ece']:.4f} (n={before['n']})")

    print("\n=== 訓練 ===")
    train_readout(model, train_examples, epochs=epochs)

    print("\n=== 訓練後 ===")
    after = evaluate(model, eval_examples)
    print(f"accuracy={after['accuracy']:.4f}, ECE={after['ece']:.4f} (n={after['n']})")

    print(f"\n正解率: {before['accuracy']:.4f} -> {after['accuracy']:.4f} "
          f"({'改善' if after['accuracy'] > before['accuracy'] else '悪化/不変'})")
    print(f"ECE:    {before['ece']:.4f} -> {after['ece']:.4f} "
          f"({'改善(低下)' if after['ece'] < before['ece'] else '悪化/不変'})")

    if after["accuracy"] > before["accuracy"] and after["ece"] < before["ece"]:
        print("\nP3・P4: 予測通り。正解率もECEも両方改善した。")
    elif after["accuracy"] > before["accuracy"]:
        print("\nP3: 予測通り（正解率は改善）。P4: 正解率ほどECEは改善しなかった——"
              "Guo et al. 2017の指摘（cross-entropyで正解率を上げてもECEが改善するとは限らない）と整合。"
              "TypeSafeがRLCDという専用手法を必要とした理由の傍証として解釈できる。")
    else:
        print("\n正解率が改善しなかった。epoch数またはデータ量を増やして再測定する。")


if __name__ == "__main__":
    _main()
