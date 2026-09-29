"""合成サポートデータ（`data/synthetic_support.py`）が学習に使えるかをタスクごとに確かめる（学習前のQ1）。

`experiments/linear_probe_support.py`と同じ手法: 凍結backboneの隠れ状態（state平均、h_dec）に
ロジスティック回帰を当て、held-out正解率をchanceと比べる。TF-IDF+LRも併記し、
  - TF-IDFが1.0近く: ラベルが表層語で漏れていて簡単すぎる（学習信号として弱い）
  - どちらもchance並み: 生成されたラベルがテキストと結びついていない
を検出する。h_decはタスクのラベルプール全体を選択肢に並べた1 branchから取る。
Gold V1は読まない。

使い方:
    python3 -m experiments.synthetic_probe [--version 0]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import binomtest, kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data.synthetic_support import TASKS, load_rows, split_rows
from legacy.experiments.linear_probe_support import _classifier, _features
from legacy.model_readout.forward import load_jev_backbone
from model.packer import Branch


def _line(name: str, y_held, pred, chance: float, ordinal: bool) -> str:
    correct = int(np.sum(pred == y_held))
    n = len(y_held)
    p = binomtest(correct, n, p=chance, alternative="greater").pvalue
    text = f"    {name:20s} held-out={correct / n:.3f} (chance={chance:.3f}, p={p:.4f})"
    if ordinal:
        tau, tau_p = kendalltau(y_held, pred)
        text += f" tau={tau:+.3f} (p={tau_p:.4f})"
    return text


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.version)
    train_rows, held_rows = split_rows(rows)
    backbone, tokenizer = load_jev_backbone()

    for task, spec in TASKS.items():
        train = [r for r in train_rows if r["task"] == task]
        held = [r for r in held_rows if r["task"] == task]
        y_train = np.array([spec.labels.index(r["label"]) for r in train])
        y_held = np.array([spec.labels.index(r["label"]) for r in held])
        chance = 1 / len(spec.labels)
        ordinal = spec.kind == "score"
        print(f"\n=== {task}（{spec.kind}, K={len(spec.labels)}, train={len(train)}, held-out={len(held)}） ===")

        tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(C=10.0, max_iter=5000))
        tfidf.fit([r["text"] for r in train], y_train)
        print(_line("TF-IDF+LR", y_held, tfidf.predict([r["text"] for r in held]), chance, ordinal))

        branch = Branch(kind=spec.kind, instructions=spec.question, options=list(spec.labels))
        feats = {name: [_features(backbone, tokenizer, r["text"], branch) for r in part]
                 for name, part in (("train", train), ("held", held))}
        for label, idx in (("backbone state-mean", 0), ("backbone h_dec", 1)):
            x_train = np.stack([f[idx] for f in feats["train"]])
            x_held = np.stack([f[idx] for f in feats["held"]])
            clf = _classifier().fit(x_train, y_train)
            print(_line(label, y_held, clf.predict(x_held), chance, ordinal))


if __name__ == "__main__":
    _main()
