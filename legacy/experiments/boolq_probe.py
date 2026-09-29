"""凍結backboneがNoul（yes/no）判定に必要な手がかりを、readout訓練前から持っているか確認する
zero-shot probe（Gate 0スタイル、Q1チェック）。

`model/packer.py::pack()`のnoul branchテンプレートと同一の地の文プロンプト
（"{passage}\\n\\n{question} Answer yes or no.\\nAnswer:"）に対し、LM headで
" Yes"/" No" の次トークンlog確率を比較する。readout（NoulReadout）は一切使わない
（訓練前の判定なので存在する必要がない）。

判定基準: accuracy が chance（50%）を統計的に上回るか（二項検定）。
"""

from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.boolq import load_boolq_stratified
from legacy.model_readout.backbone import MODEL_NAME


def _prose_cue_prompt(passage: str, instructions: str) -> str:
    """`model/packer.py::pack()`のnoul branch用テンプレートと同一の地の文プロンプト。"""
    return f"{passage}\n\n{instructions} Answer yes or no.\nAnswer:"


def _first_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids = []
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        ids.append(token_ids[0])
    assert len(set(ids)) == len(ids), f"候補語の先頭トークンが重複: {words} -> {ids}"
    return ids


@torch.no_grad()
def _log_probs(causal_lm, tokenizer, prompt: str, token_ids: list[int]) -> list[float]:
    inputs = tokenizer(prompt, return_tensors="pt")
    logits = causal_lm(**inputs).logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)
    return [float(log_probs[i]) for i in token_ids]


def _question_to_instructions(question: str) -> str:
    q = question.strip()
    q = q[0].upper() + q[1:] if q else q
    if not q.endswith("?"):
        q += "?"
    return q


def _binomial_p_value(k: int, n: int, p: float = 0.5) -> float:
    """両側二項検定のp値（正規近似、n>=30想定）。"""
    mean = n * p
    std = math.sqrt(n * p * (1 - p))
    z = (k - mean) / std
    # 標準正規分布の両側p値
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def probe_boolq(causal_lm, tokenizer, n_per_label: int = 60, seed: int = 0) -> None:
    """BoolQのheld-out相当サンプルに対し、readoutなしのzero-shot正解率を測る（Q1チェック）。"""
    words = [" Yes", " No"]
    token_ids = _first_token_ids(tokenizer, words)

    ds = load_boolq_stratified(n_per_label=n_per_label, seed=seed, offset=0)
    correct = 0
    n = 0
    yes_picks = 0
    for ex in ds:
        instructions = _question_to_instructions(ex["question"])
        prompt = _prose_cue_prompt(ex["passage"], instructions)
        lp_yes, lp_no = _log_probs(causal_lm, tokenizer, prompt, token_ids)
        predicted_yes = lp_yes > lp_no
        yes_picks += int(predicted_yes)
        if predicted_yes == bool(ex["answer"]):
            correct += 1
        n += 1

    accuracy = correct / n
    p_value = _binomial_p_value(correct, n, p=0.5)
    print(f"BoolQ zero-shot probe: n={n}, accuracy={accuracy:.4f}, p={p_value:.4f}, yes_picks={yes_picks}/{n}")
    print(f"  判定: accuracy>chance(0.5)かつp<0.05 -> {accuracy > 0.5 and p_value < 0.05}")


def _main() -> None:
    print(f"model: {MODEL_NAME}（凍結・LM head込み、readoutは通さない）")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    causal_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, attn_implementation="eager")
    causal_lm.eval()

    probe_boolq(causal_lm, tokenizer)


if __name__ == "__main__":
    _main()
