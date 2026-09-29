"""固定したバックボーン上でAG News分類用のPointerReadoutを訓練する。

各例をtree attention mask付きでforwardし、Choiceの選択肢logitにcross-entropyを適用する。
訓練前後のheld-outデータに対する正解率とECEを測定する。バックボーンとNoulReadoutは更新しない。
"""

from __future__ import annotations

from pathlib import Path

import random
import time

import torch
import torch.nn.functional as F

from data.ag_news import eval_split, to_packed_example, train_subset
from data.common import ChoiceExample, reshuffled
from legacy.model_readout.forward import JevModel, pool_options, to_additive_mask
from model.packer import pack
from legacy.model_readout.readout import PointerReadout
from model.tree_mask import build_position_ids, build_tree_attention_mask


def _choice_logits(
    model: JevModel, example: ChoiceExample, *, requires_grad: bool, backbone_grad: bool = False
) -> torch.Tensor:
    """1件のChoiceExample（Choice branch1つ）から、pointer readoutのlogit z（未softmax）を計算する。

    `backbone_grad=False`（既定）ならbackboneは常に`torch.no_grad()`（重みを更新しない）。
    `backbone_grad=True`はgradual unfreezing ablation用で、`_set_backbone_trainable_layers`で
    `requires_grad=True`にした層だけがoptimizerに登録されている前提（他の層は
    `requires_grad=False`のままなのでgradは作られない）。`requires_grad`は「pointer readout側の
    計算に勾配を通すか」を切り替える（訓練時True、評価時False＝`evaluate`側で`@torch.no_grad()`済み
    なのでここは実質常にFalseで呼ばれる）。
    """
    packed = pack(example.request, model.tokenizer)
    bool_mask = build_tree_attention_mask(packed)
    position_ids = build_position_ids(packed)

    input_ids = torch.tensor([packed.input_ids], dtype=torch.long)
    attention_mask = to_additive_mask(bool_mask, dtype=model.backbone.dtype)
    pos_ids = torch.tensor([position_ids], dtype=torch.long)

    with torch.set_grad_enabled(backbone_grad and requires_grad):
        out = model.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos_ids)
    hidden = out.last_hidden_state[0]

    span = packed.branch_spans[0]
    h_dec = hidden[span.decision_index]
    h_opts = pool_options(hidden, span, model.option_pooling)

    with torch.set_grad_enabled(requires_grad):
        return model.pointer(h_dec, h_opts)


def _set_backbone_trainable_layers(model: JevModel, n: int) -> None:
    """backboneの末尾n層だけ`requires_grad=True`にする（gradual unfreezing）。n=0で全凍結。

    LP-FT/gradual unfreezing文献の手法: readout（linear probe相当）と一緒に、backboneの
    最終数層だけを小さいlearning rateで学習する。0.5Bクラスの小さいbackboneでは
    最終1〜2層に情動・順序の区別が線形には現れていない可能性を検証するためのもの。
    """
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    if n > 0:
        for layer in model.backbone.layers[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)


def train_readout(
    model: JevModel,
    train_examples: list[ChoiceExample],
    epochs: int = 3,
    lr: float = 1e-3,
    log_every: int = 100,
    shuffle_options: bool = False,
    seed: int = 0,
    score_mse_weight: float = 1.0,
    unfreeze_last_n_layers: int = 0,
    backbone_lr: float = 1e-5,
    extra_backbone_params: list | None = None,
) -> None:
    """`PointerReadout`（W_q, W_k）をcross-entropyで訓練する。既定ではbackboneは凍結。

    `unfreeze_last_n_layers > 0`ならbackbone末尾のn層も`backbone_lr`（既定でpointerの
    1/100）で同時に学習する（gradual unfreezing ablation。詳細は
    `_set_backbone_trainable_layers`）。この関数は`model.backbone`の重みを直接書き換える
    副作用があるので、呼び出し側はablationのarmごとに新しい`JevModel()`を使うこと
    （同一backboneを使い回すと前armの学習が後armに漏れる）。

    `extra_backbone_params`を渡すと、層単位のfreeze/unfreeze制御を一切せず、渡された
    パラメータ集合（既に`requires_grad=True`になっている前提）をそのまま`backbone_lr`で
    optimizerに登録する。LoRAアダプタなど、層単位の`unfreeze_last_n_layers`では表現でき
    ない訓練対象を渡すためのもの。指定時は`unfreeze_last_n_layers`は無視される。

    `shuffle_options=True`の場合、エポックごとに各exampleの選択肢の順序をランダムに並べ替える
    （正解ラベルは追跡して付け替える）。目的は、選択肢の表面順序に対するreadoutの過学習を防ぐこと
    （固定順序で訓練すると、選択肢テキストの意味ではなく位置に依存した判断を学習してしまう）。

    Score branch（`kind == "score"`）には、cross-entropyに加えて正規化MSE項を足す:
    `((score_expectation - gold_level) / (K-1))^2`。cross-entropyは選択肢を名義尺度
    （順序無視のクラス）として扱うため、レベルを1つ外すのも全部外すのも同じ罰則になる。
    MSE項は「隣のレベルに外れる」方を「正反対のレベルに外れる」より軽く罰する、順序を
    意識した補助信号を与える（公式比較のseverity/frustrationで隣接レベルへの誤りが
    多かったことに対応）。
    """
    if extra_backbone_params is not None:
        backbone_grad = True
        param_groups = [
            {"params": list(model.pointer.parameters()), "lr": lr},
            {"params": list(extra_backbone_params), "lr": backbone_lr},
        ]
    else:
        _set_backbone_trainable_layers(model, unfreeze_last_n_layers)
        backbone_grad = unfreeze_last_n_layers > 0
        param_groups = [{"params": list(model.pointer.parameters()), "lr": lr}]
        if backbone_grad:
            backbone_params = [
                p for layer in model.backbone.layers[-unfreeze_last_n_layers:] for p in layer.parameters()
            ]
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
    optimizer = torch.optim.Adam(param_groups)

    examples = list(train_examples)
    shuffle_rng = random.Random(seed)

    for epoch in range(epochs):
        random.shuffle(examples)
        epoch_examples = [reshuffled(ex, shuffle_rng) for ex in examples] if shuffle_options else examples
        epoch_loss = 0.0
        epoch_correct = 0
        start = time.perf_counter()

        for i, example in enumerate(epoch_examples):
            optimizer.zero_grad()
            z = _choice_logits(model, example, requires_grad=True, backbone_grad=backbone_grad)
            loss = F.cross_entropy(z.unsqueeze(0), torch.tensor([example.label]))

            k = z.shape[0]
            if example.request.branches[0].kind == "score" and k > 1:
                probs = torch.softmax(z, dim=-1)
                levels = torch.arange(1, k + 1, dtype=probs.dtype)
                pred_score = (levels * probs).sum()
                gold_score = float(example.label + 1)
                loss = loss + score_mse_weight * ((pred_score - gold_score) / (k - 1)) ** 2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += int(z.argmax().item() == example.label)

            if log_every and (i + 1) % log_every == 0:
                print(f"  epoch {epoch} [{i + 1}/{len(examples)}] running_loss={epoch_loss / (i + 1):.4f}")

        elapsed = time.perf_counter() - start
        print(f"epoch {epoch}: loss={epoch_loss / len(examples):.4f}, "
              f"train_acc={epoch_correct / len(examples):.4f}, {elapsed:.1f}s")

    if extra_backbone_params is None and backbone_grad:
        _set_backbone_trainable_layers(model, 0)


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
def evaluate(model: JevModel, eval_examples: list[ChoiceExample]) -> dict:
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


def _build_examples(dataset) -> list[ChoiceExample]:
    label_names = dataset.features["label"].names
    return [to_packed_example(ex["text"], ex["label"], label_names) for ex in dataset]

def _main(
    train_size: int = 2000,
    eval_size: int = 500,
    epochs: int = 3,
    seed: int | None = None,
) -> None:
    print(f"train_size={train_size}, eval_size={eval_size}, epochs={epochs}, seed={seed}")
    model = JevModel()

    if seed is not None:
        # scaling系実験と同じ手順: seedで初期化し直したPointerReadoutから訓練を始める
        # （データ量以外の要因を揃え、以降のシャッフルも再現可能にする）。
        torch.manual_seed(seed)
        random.seed(seed)
        model.pointer = PointerReadout(model.backbone.config.hidden_size)

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

    # 訓練済みreadoutの重みをcheckpointに保存する（backboneは固定なので保存対象はpointerのみ）。
    # 読み込み側は `model.pointer.load_state_dict(torch.load(path)["state_dict"])`。
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"pointer_agnews_s{train_size}_e{epochs}.pt"
    torch.save(
        {
            "readout": "PointerReadout",
            "hidden_size": model.backbone.config.hidden_size,
            "train_size": train_size,
            "epochs": epochs,
            "state_dict": model.pointer.state_dict(),
        },
        ckpt_path,
    )
    print(f"checkpoint: {ckpt_path}")

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
