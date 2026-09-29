"""凍結backboneがそもそも順序尺度（severity/frustration/report_quality/感情強度）の情報を
持っているかを、readoutを通さずLM head zero-shotで測る（`EXP_HEAD_VS_BACKBONE.md` Gate 0）。

`results/2026-09-22_score_diagnosis.md`で、Score readoutの病理を
「選択肢固有バイアスの支配＋テキスト依存信号の弱さ」まで絞り込んだ。推論時のバイアス除去で
前者は消えるが、公式severityは救われない（校正後も全stateをcosmeticに置く）。
残る問いは、**信号が無いのはreadoutの学習の失敗なのか、凍結backboneにそもそも情報が
無いのか**である。これはreadoutを一切通さずLM head zero-shotで測れば切り分けられる。

Phase 9で同じ手法を使い、同一の凍結Qwen2.5-0.5BがAG News accuracy 0.675を出して
「backboneサイズは主因ではない」と判断した経緯がある。今回はその順序尺度版。

プロンプトは`model/packer.py::pack()`の地の文形式（"{instructions} Answer with one word —
{opt1}, {opt2}, ....\nAnswer:"）と、選択肢テキストは`official_compare.py`の公式option文字列
をそのまま使う（本番の訓練済みモデルが受け取る入力と同一形式で測るため）。

LM headはトークンごとの事前確率が強いので（例えば`positive`は`negative`より出やすい）、
readout側と同じnull state校正を併記する。内容のないstateでの次トークン分布を引く。

severity_tableのみ、公式ドキュメントに掲載された既知の正しい順序（`SEVERITY_STATES`）を
持つので、Kendall tauで定量判定できる。frustration/report_qualityには独立した既知順序の
state集合が存在しない（`official_compare.py`には単一state 1件ずつしかない）ため、
同じ5 severity statesに別の質問を当てて分布・argmaxのみ定性的に観察する
（`EXP_HEAD_VS_BACKBONE.md` §3.1: 未検証のground truthを捏造しない）。

使い方:
    python3 -m experiments.backbone_probe
"""

from __future__ import annotations

import math

import torch
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.sst5 import load_sst5
from legacy.experiments.official_compare import FRUSTRATION_LEVELS, SEVERITY_LEVELS
from legacy.model_readout.backbone import MODEL_NAME

# severityは公式ドキュメントに正解順序ありの5-state表がある。frustration/report_qualityは
# 単一例（spinner_ticket）しかなく、独立の既知順序state集合が無いので定量判定は行わない。
SEVERITY_STATES: list[tuple[str, float]] = [
    ("The export button is misaligned by a few pixels on the settings page.", 0.0),
    (
        "The PDF export button does nothing when clicked. I can still export to CSV and "
        "convert it myself, but that takes ages.",
        1.0,
    ),
    (
        "Export to PDF fails with a spinner that never finishes. Some of our team say CSV "
        "export still works for them, others say it fails too.",
        1.12,
    ),
    (
        "The export button crashes the settings page in Safari. It works in Chrome, but a "
        "few of our customers only use Safari.",
        1.3,
    ),
    ("Nobody on our team can log in since this morning. We get a 500 error on every attempt.", 2.0),
]

SEVERITY_QUESTION = "How severe is the reported issue?"
FRUSTRATION_QUESTION = "How frustrated is the customer?"
REPORT_QUALITY_QUESTION = "How much does the report give an engineer to work with?"
REPORT_QUALITY_LEVELS = [
    "No detail; just says something is broken",
    "Names the feature but no steps or environment",
    "Steps to reproduce or environment, but not both",
    "Steps to reproduce and environment",
]


def _prose_cue_prompt(state: str, instructions: str, options: list[str]) -> str:
    """`model/packer.py::pack()`のScore/Choice branch用テンプレートと同一の地の文プロンプト。"""
    joined = ", ".join(options[:-1]) + (", " if len(options) > 1 else "") + options[-1] + "."
    return f"{state}\n\n{instructions} Answer with one word — {joined}\nAnswer:"


SENTIMENT_PROMPT = (
    "Movie review: {state}\n"
    "Is the sentiment of this review negative or positive? Answer with one word.\n"
    "Answer:"
)
SENTIMENT_WORDS = [" negative", " positive"]

NULL_STATE = "N/A"


def _first_token_ids(tokenizer, words: list[str]) -> list[int]:
    """各候補語の最初のトークンID。候補同士が同じ先頭トークンだと比較にならないので検査する。"""
    ids = [tokenizer(word, add_special_tokens=False)["input_ids"][0] for word in words]
    assert len(set(ids)) == len(ids), f"候補語の先頭トークンが衝突している: {words} -> {ids}"
    return ids


@torch.no_grad()
def _log_probs(causal_lm, tokenizer, prompt: str, token_ids: list[int]) -> list[float]:
    """プロンプト直後の次トークン分布から、候補語の先頭トークンのlog確率を取り出す。"""
    inputs = tokenizer(prompt, return_tensors="pt")
    logits = causal_lm(**inputs).logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)
    return [float(log_probs[i]) for i in token_ids]


@torch.no_grad()
def _option_avg_log_probs(causal_lm, tokenizer, prompt: str, options: list[str]) -> list[float]:
    """各optionをteacher-forcingでスコアする: prompt直後にoptionトークン列を続けた場合の
    平均トークンlog確率（合計だと長い選択肢が不利になるので平均を取る）。

    `_first_token_ids`の先頭トークンのみ比較する方式は、report_qualityの選択肢が
    "Steps to reproduce or environment..." / "Steps to reproduce and environment"のように
    先頭数トークンを共有する場合に区別できない。全トークンをteacher-forcingでスコアすれば
    衝突しない。
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    scores: list[float] = []
    for option in options:
        option_ids = tokenizer(option, add_special_tokens=False, return_tensors="pt")["input_ids"]
        full_ids = torch.cat([prompt_ids, option_ids], dim=1)
        logits = causal_lm(full_ids).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        start = prompt_ids.shape[1] - 1  # このindexのlogitがoptionの最初のトークンを予測する
        n = option_ids.shape[1]
        total = sum(float(log_probs[start + i, int(option_ids[0, i])]) for i in range(n))
        scores.append(total / n)
    return scores


def _calibrated(log_probs: list[float], bias: list[float]) -> list[float]:
    adjusted = [lp - b for lp, b in zip(log_probs, bias)]
    highest = max(adjusted)
    weights = [math.exp(a - highest) for a in adjusted]
    total = sum(weights)
    return [w / total for w in weights]


def _expectation(probs: list[float]) -> float:
    return sum(i * p for i, p in enumerate(probs))


def probe_severity(causal_lm, tokenizer) -> None:
    """公式severity表の5stateを、readoutを通さずLM headで判定する（Gate 0の主判定）。

    `argmax`一致率だけだと定数予測器（常に同じレベルを返す）が公式の多数派レベルと偶然
    一致して高得点に見える（`results/2026-09-22_score_diagnosis.md`の誤読を参照）。
    Kendall tau・p値・定数予測器フラグ・予測幅を必ず併記する
    （`official_compare.py::run_severity_table`と同じ指標定義）。
    """
    print("\n=== Gate 0: backboneはseverityを知っているか（LM head zero-shot, K=3, 地の文プロンプト） ===")
    null_prompt = _prose_cue_prompt(NULL_STATE, SEVERITY_QUESTION, SEVERITY_LEVELS)
    bias = _option_avg_log_probs(causal_lm, tokenizer, null_prompt, SEVERITY_LEVELS)

    official = [score for _, score in SEVERITY_STATES]
    raw_positions: list[float] = []
    calibrated_positions: list[float] = []
    raw_argmax: list[int] = []
    calibrated_argmax: list[int] = []
    for state, _ in SEVERITY_STATES:
        prompt = _prose_cue_prompt(state, SEVERITY_QUESTION, SEVERITY_LEVELS)
        log_probs = _option_avg_log_probs(causal_lm, tokenizer, prompt, SEVERITY_LEVELS)
        probs = [math.exp(lp) for lp in log_probs]
        total = sum(probs)
        probs = [p / total for p in probs]
        calibrated_probs = _calibrated(log_probs, bias)

        raw_positions.append(_expectation(probs))
        calibrated_positions.append(_expectation(calibrated_probs))
        raw_argmax.append(max(range(len(probs)), key=lambda i: probs[i]))
        calibrated_argmax.append(max(range(len(calibrated_probs)), key=lambda i: calibrated_probs[i]))
        print(f"  {state[:52]:<52} 校正後 " +
              " ".join(f"L{i}={p:.3f}" for i, p in enumerate(calibrated_probs)))

    raw_tau, raw_p = kendalltau(official, raw_positions)
    calibrated_tau, calibrated_p = kendalltau(official, calibrated_positions)
    raw_constant = len(set(raw_argmax)) == 1
    calibrated_constant = len(set(calibrated_argmax)) == 1
    raw_spread = max(raw_positions) - min(raw_positions)
    calibrated_spread = max(calibrated_positions) - min(calibrated_positions)
    print(f"  tau: 校正前 {raw_tau:+.3f} (p={raw_p:.3f}) / 校正後 {calibrated_tau:+.3f} (p={calibrated_p:.3f})")
    print(f"  定数予測器: 校正前 {'YES' if raw_constant else 'no'} / 校正後 {'YES' if calibrated_constant else 'no'}")
    print(f"  予測幅(max-min): 校正前 {raw_spread:.3f} / 校正後 {calibrated_spread:.3f}")
    print(f"  argmax: 校正前 {raw_argmax} / 校正後 {calibrated_argmax}  (公式={[0, 1, 1, 1, 2]})")


def probe_ordinal_no_ground_truth(causal_lm, tokenizer, name: str, question: str, options: list[str]) -> None:
    """frustration/report_qualityには公式ドキュメント掲載の独立した既知順序state集合が
    存在しない（`official_compare.py`は各1件のみ）。severityと同じ5 statesに別の質問を
    当てて分布・argmaxのみ定性的に観察する。tauは計算しない
    （`EXP_HEAD_VS_BACKBONE.md` §3.1: 未検証のground truthを捏造しない）。
    """
    print(f"\n=== Gate 0（定性のみ・既知順序なし）: {name}, K={len(options)} ===")
    null_prompt = _prose_cue_prompt(NULL_STATE, question, options)
    bias = _option_avg_log_probs(causal_lm, tokenizer, null_prompt, options)

    for state, _ in SEVERITY_STATES:
        prompt = _prose_cue_prompt(state, question, options)
        log_probs = _option_avg_log_probs(causal_lm, tokenizer, prompt, options)
        calibrated_probs = _calibrated(log_probs, bias)
        print(f"  {state[:52]:<52} 校正後 " +
              " ".join(f"L{i}={p:.3f}" for i, p in enumerate(calibrated_probs)))


def probe_sentiment(causal_lm, tokenizer, n_per_class: int = 50) -> None:
    """対照: 最も易しい情動タスク（sst5の最遠2レベルの二値判定）をLM headで解く。

    ここでも精度が出ないなら、凍結backboneには情動軸の情報が乏しいということになり、
    Score readoutが弱いのはデータではなく表現の問題になる。
    """
    print(f"\n=== 対照: backboneは感情極性を知っているか（LM head zero-shot, 二値） ===")
    token_ids = _first_token_ids(tokenizer, SENTIMENT_WORDS)
    bias = _log_probs(causal_lm, tokenizer, SENTIMENT_PROMPT.format(state=NULL_STATE), token_ids)

    dataset = load_sst5("validation")
    negatives: list[str] = []
    positives: list[str] = []
    for row in dataset:
        if int(row["label"]) == 0 and len(negatives) < n_per_class:
            negatives.append(row["text"])
        elif int(row["label"]) == 4 and len(positives) < n_per_class:
            positives.append(row["text"])
        if len(negatives) >= n_per_class and len(positives) >= n_per_class:
            break

    raw_correct = calibrated_correct = 0
    raw_picks = [0, 0]
    calibrated_picks = [0, 0]
    for text, gold in [(t, 0) for t in negatives] + [(t, 1) for t in positives]:
        log_probs = _log_probs(causal_lm, tokenizer, SENTIMENT_PROMPT.format(state=text), token_ids)
        raw_pick = int(log_probs[1] > log_probs[0])
        calibrated_probs = _calibrated(log_probs, bias)
        calibrated_pick = int(calibrated_probs[1] > calibrated_probs[0])
        raw_correct += int(raw_pick == gold)
        calibrated_correct += int(calibrated_pick == gold)
        raw_picks[raw_pick] += 1
        calibrated_picks[calibrated_pick] += 1

    total = len(negatives) + len(positives)
    print(f"  n={total} (negative={len(negatives)}, positive={len(positives)}, チャンス=0.500)")
    print(f"  accuracy: 校正前 {raw_correct / total:.3f} / 校正後 {calibrated_correct / total:.3f}")
    print(f"  選択の偏り: 校正前 {raw_picks} / 校正後 {calibrated_picks}")


def _main() -> None:
    print(f"model: {MODEL_NAME}（凍結・LM head込み、readoutは通さない）")
    causal_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    causal_lm.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    probe_sentiment(causal_lm, tokenizer)
    probe_severity(causal_lm, tokenizer)
    probe_ordinal_no_ground_truth(causal_lm, tokenizer, "frustration", FRUSTRATION_QUESTION, FRUSTRATION_LEVELS)
    probe_ordinal_no_ground_truth(
        causal_lm, tokenizer, "report_quality", REPORT_QUALITY_QUESTION, REPORT_QUALITY_LEVELS
    )


if __name__ == "__main__":
    _main()
