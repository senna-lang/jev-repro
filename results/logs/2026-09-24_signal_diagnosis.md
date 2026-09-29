# 2026-09-24: Gold V1品質の信号診断

Gold V1（Choice top1=0.475、Score top1=0.233＜ランダム基準0.311）の原因を、
言語・加法バイアス・readout取りこぼし・backbone天井・タスク構造の合う学習データ、
の順に切り分けた。Gold V1は評価専用（GOAL.md §6分離規則）。学習はしていない。

## 1. 言語ミスマッチ（棄却）

機械翻訳の日英対（`evaluation/private/gold_v1_adjudicated_en_mt.jsonl`、診断専用）。
詳細: `results/2026-09-24_language_mismatch_hypothesis_test.md`。

- Choice: JA=EN=0.475。40件中17件で予測が反転するが正誤は4対4で打ち消し。
- Score: 0.233→0.367（弱い改善、ランダム基準並み）。

## 2. null state校正（Scoreの主因ではない）

`evaluation/run.py --null-calibration`（既定オフ）。`NULL_STATES=("N/A","...")` の平均log確率を
choice/scoreのバイアスとして引く。sst5ではS1として0.564→0.795だった手法。

| | JA なし | JA あり | EN なし | EN あり |
|---|---|---|---|---|
| Choice top1 | 0.475 | 0.475 | 0.475 | 0.375 |
| Score top1 | 0.233 | 0.300 | 0.367 | 0.200 |
| Score E[s] sd | 0.17 | 0.19 | 0.36 | 0.23 |

JA 3択の最後の選択肢は 0/22→5/22 と位置バイアスは減るが、Scoreの予測幅はほぼ動かない。
英語版は悪化。Gold V1のScoreがランダム以下な主因は、null stateで引ける加法バイアスではない。

## 3. Gold V1 zero-shot（LM head、readoutなし）

`experiments/gold_v1_zeroshot_probe.py`。Qwen2.5-0.5B。

| | JA Choice | JA Score | EN Choice | EN Score |
|---|---|---|---|---|
| zero-shot raw | 0.450 | 0.367 | 0.400 | 0.300 |
| 学習済みpointer | 0.475 | 0.233 | 0.475 | 0.367 |

当初「(b) backboneに信号がない」と書いたが訂正する。Scoreではzero-shotがpointerより良い
（0.367 vs 0.233、MAE 0.692 vs 0.743、sd 0.41 vs 0.17）ので取りこぼし（a）もある。
ただしzero-shot自体がランダム基準（~0.31）並みで、天井は低い。Choiceはほぼ同点。

zero-shot rawでもJA 3択の最後の選択肢は0/22。位置バイアスはpointer固有ではない。

## 4. 同構造の英語データ（LM head zero-shot）

`experiments/support_task_probe.py`。

| 候補 | n | 指標 | 判定 |
|---|---|---|---|
| ticket `queue` 5部署（説明付きK=3〜4） | 100 | top1 0.320 / chance 0.297 / p=0.34 | LM headでは不採用 |
| tweet anger強度 | 90 | τ +0.124 / p=0.13、rawは90/90が「furious」 | 不採用 |
| report_quality相当 | — | HFで見つからず | 対象外 |

## 5. linear probe（隠れ状態に信号があるか）

`experiments/linear_probe_support.py`。凍結backbone 1回forwardから state平均と `h_dec`、
比較対象はTF-IDF+LR。

| タスク | 手法 | held-out | chance | p |
|---|---|---|---|---|
| queue 5部署 | TF-IDF+LR | 0.540 | 0.200 | <0.0001 |
| queue 5部署 | backbone state平均 | 0.495 | 0.200 | <0.0001 |
| queue 5部署 | backbone `h_dec` | 0.480 | 0.200 | <0.0001 |
| anger 3段階 | TF-IDF+LR | 0.373 | 0.333 | 0.17 |
| anger 3段階 | backbone `h_dec` | 0.340 | 0.333 | 0.46 |

**queue:** テキストから部署は学べ、凍結0.5Bの隠れ状態にも同じ信号がある。
LM head zero-shot 0.32は「プロンプトで引き出せない」だけ。本番pointerは学習readoutなので、
採否はlinear probeを基準にし、`ticket_queue` を訓練候補とする。

**anger:** TF-IDFもbackboneもchance。ラベルがテキストから安定して読めない。frustrationの代理には使わない。

`data/ticket_queue.py` は同一(subject, body)の重複行を除く（offset分割でも22件重なっていた）。
Human Resourcesのユニーク件数は461で、300+40のtrain/held-outは足りる。

## 6. より大きいモデル（Qwen2.5-1.5B zero-shot）

ディスク不足のため不完全な `Qwen2.5-7B-Instruct` キャッシュ（重みなし1.6GB）と
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`（276MB）を削除して確保。`sennaLLMLearner` は未削除。

| | JA Choice | JA Score | EN Choice | EN Score |
|---|---|---|---|---|
| 0.5B raw / cal | 0.450 / 0.450 | 0.367 / 0.300 | 0.400 / 0.375 | 0.300 / 0.267 |
| 1.5B raw / cal | 0.375 / **0.575** | 0.367 / 0.200 | 0.425 / **0.550** | 0.300 / 0.367 |

Choiceは校正後だけ上がる（n=40で幅は広い）。Scoreはサイズでは改善しない。
backbone載せ替えは絶対遅延が増えるため、この時点では採用しない。

## 次

Gate 1-d: 既存6データセットに `ticket_queue`（300件/部署×5）を足してpointerを再訓練し、
Gold V1 Choiceが0.475を上回るか、in-domainとbugzilla OOD τが退行しないかを見る。
