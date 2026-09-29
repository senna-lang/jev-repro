"""複数データセット（AG News, banking77など）で共有するChoice/Score例の型とヘルパー。

順序過敏性への対処として、訓練時に選択肢の順序をエポックごとにシャッフルするための
`reshuffled`もここに置く（既存の`PackedRequest`は不変オブジェクトとして扱い、
シャッフル済みの新しいコピーを都度作る）。`pick_instruction`は、readoutが特定の
instructions文面に癒着するのを防ぐための、instructions文面のランダム選択ヘルパー。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from model.packer import Branch, PackedRequest


@dataclass
class ChoiceExample:
    """1件のChoice branchリクエストと正解ラベルのindex。

    データセット非依存（AG News・banking77など、どのソースの例もこの形に変換する）。
    """

    request: PackedRequest
    label: int


def reshuffled(example: ChoiceExample, rng: random.Random) -> ChoiceExample:
    """選択肢の順序をランダムに並べ替えた新しい`ChoiceExample`を返す（元のexampleは変更しない）。

    正解ラベルの文字列を追跡し、並べ替え後のindexに付け替える。branchは1つのみ対応
    （このリポジトリのChoice訓練データは常に1リクエスト1 branch）。

    Score branch（`kind == "score"`）は並べ替えない：選択肢の順序自体がレベルの大小
    （`score_expectation`が前提とする1始まりの順序）を表すため、シャッフルすると
    ラベルの意味が壊れる。素通しで返す。
    """
    branch = example.request.branches[0]
    if branch.kind == "score":
        return example

    correct_text = branch.options[example.label]

    permuted = list(range(len(branch.options)))
    rng.shuffle(permuted)
    new_options = [branch.options[i] for i in permuted]
    new_label = new_options.index(correct_text)

    new_branch = Branch(kind=branch.kind, instructions=branch.instructions, options=new_options)
    new_request = PackedRequest(state=example.request.state, branches=[new_branch])
    return ChoiceExample(request=new_request, label=new_label)


def pick_instruction(variants: list[str], rng: random.Random) -> str:
    """instructions文面の候補からランダムに1つ選ぶ（readoutの特定文面への過学習を防ぐ）。"""
    return rng.choice(variants)
