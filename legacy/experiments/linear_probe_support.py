"""凍結backboneの隠れ状態に、サポート系タスクの情報が線形に読めるかを測る（zero-shotとは別の問い）。

`experiments/support_task_probe.py`のzero-shot（LM head）では、ticket queue（部署）・anger強度とも
有意な信号が出なかった。それが「backboneの限界」か「ラベルのノイズ」か「プロンプトで引き出せない
だけ」かを切り分ける:

  - backbone特徴のlinear probe ≫ chance: 隠れ状態に信号はある（readout/転移の問題）
  - TF-IDF高・backbone低: backboneの表現の限界
  - 両方chance並み: ラベルがテキストから学べない（データの問題）

特徴は1回の凍結forward（`pack()`と同じ地の文形式、1 branch）から2種類取る:
  (i) state部分トークンの平均（causalなのでbranchの影響を受けない）
  (ii) h_dec（"Answer:"位置、本番readoutが読む位置）
Gold V1は読まない。readout重みは作らない（sklearnの分類器のみ、保存しない）。
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch
from scipy.stats import binomtest, kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.anger_intensity import INSTRUCTION_VARIANTS as ANGER_INSTRUCTIONS
from data.anger_intensity import LEVEL_ORDER, _INTENSITY_TO_LABEL, load_anger_intensity_stratified
from data.ticket_queue import INSTRUCTION_VARIANTS as QUEUE_INSTRUCTIONS
from data.ticket_queue import QUEUE_OPTIONS, QUEUES, load_ticket_queue_stratified, state_text
from legacy.model_readout.forward import load_jev_backbone, to_additive_mask
from model.packer import Branch, PackedRequest, pack
from model.tree_mask import build_position_ids, build_tree_attention_mask

SEED = 0
QUEUE_TRAIN, QUEUE_HELD = 100, 40  # per queue
ANGER_TRAIN, ANGER_HELD = 150, 50  # per level


@torch.no_grad()
def _features(backbone, tokenizer, state: str, branch: Branch) -> tuple[np.ndarray, np.ndarray]:
    packed = pack(PackedRequest(state=state, branches=[branch]), tokenizer)
    span = packed.branch_spans[0]
    assert span.decision_index == len(packed.input_ids) - 1
    out = backbone(
        input_ids=torch.tensor([packed.input_ids]),
        attention_mask=to_additive_mask(build_tree_attention_mask(packed), dtype=backbone.dtype),
        position_ids=torch.tensor([build_position_ids(packed)]),
    )
    hidden = out.last_hidden_state[0]
    return hidden[: packed.state_len].mean(dim=0).numpy(), hidden[span.decision_index].numpy()


def _extract(backbone, tokenizer, states: list[str], branch: Branch) -> tuple[np.ndarray, np.ndarray]:
    pairs = [_features(backbone, tokenizer, s, branch) for s in states]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def _classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def _report(name: str, y_train, y_held, train_pred, held_pred, chance: float, ordinal: bool) -> None:
    train_acc = float(np.mean(train_pred == y_train))
    correct = int(np.sum(held_pred == y_held))
    n = len(y_held)
    p = binomtest(correct, n, p=chance, alternative="greater").pvalue
    line = f"  {name:22s} train={train_acc:.3f} held-out={correct / n:.3f} (chance={chance:.3f}, p={p:.4f})"
    if ordinal:
        tau, tau_p = kendalltau(y_held, held_pred)
        line += f" tau={tau:+.3f} (p={tau_p:.4f})"
    print(line + f" pred={dict(sorted(Counter(held_pred.tolist()).items()))}")


def _run(task: str, backbone, tokenizer, train_states, y_train, held_states, y_held, branch, chance, ordinal) -> None:
    print(f"\n=== {task}: train n={len(y_train)}, held-out n={len(y_held)} ===")
    tfidf = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)), LogisticRegression(C=10.0, max_iter=5000))
    tfidf.fit(train_states, y_train)
    _report("TF-IDF+LR（基準）", y_train, y_held, tfidf.predict(train_states), tfidf.predict(held_states), chance, ordinal)

    train_mean, train_dec = _extract(backbone, tokenizer, train_states, branch)
    held_mean, held_dec = _extract(backbone, tokenizer, held_states, branch)
    for name, x_train, x_held in (("backbone state-mean", train_mean, held_mean), ("backbone h_dec", train_dec, held_dec)):
        clf = _classifier().fit(x_train, y_train)
        _report(name, y_train, y_held, clf.predict(x_train), clf.predict(x_held), chance, ordinal)


def _main() -> None:
    backbone, tokenizer = load_jev_backbone()

    queue_train = load_ticket_queue_stratified(QUEUE_TRAIN, seed=SEED)
    queue_held = load_ticket_queue_stratified(QUEUE_HELD, seed=SEED, offset=QUEUE_TRAIN)
    queue_branch = Branch(kind="choice", instructions=QUEUE_INSTRUCTIONS[0], options=[QUEUE_OPTIONS[q] for q in QUEUES])
    label = {q: i for i, q in enumerate(QUEUES)}
    _run(
        "ticket queue（5部署）", backbone, tokenizer,
        [state_text(r["subject"], r["body"]) for r in queue_train], np.array([label[q] for q in queue_train["queue"]]),
        [state_text(r["subject"], r["body"]) for r in queue_held], np.array([label[q] for q in queue_held["queue"]]),
        queue_branch, chance=1 / len(QUEUES), ordinal=False,
    )

    anger_train = load_anger_intensity_stratified(ANGER_TRAIN, seed=SEED)
    anger_held = load_anger_intensity_stratified(ANGER_HELD, seed=SEED, offset=ANGER_TRAIN)
    assert not set(anger_train["id"]) & set(anger_held["id"])
    anger_branch = Branch(kind="score", instructions=ANGER_INSTRUCTIONS[0], options=list(LEVEL_ORDER))
    _run(
        "anger強度（3レベル）", backbone, tokenizer,
        list(anger_train["tweet"]), np.array([_INTENSITY_TO_LABEL[v] for v in anger_train["sentiment_intensity"]]),
        list(anger_held["tweet"]), np.array([_INTENSITY_TO_LABEL[v] for v in anger_held["sentiment_intensity"]]),
        anger_branch, chance=1 / len(LEVEL_ORDER), ordinal=True,
    )


if __name__ == "__main__":
    random.seed(SEED)
    _main()
