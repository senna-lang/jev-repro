# 2026-09-24: Gate 1-d ticket_queue 追加訓練 — Gold V1 Choiceは未達

`experiments/queue_gate1d.py`。既存6データセット + ticket_queue 300件/部署×5 = 1500件、
mean-pool、3 epoch、選択肢順シャッフル、seed=0。学習時間 6559s。
checkpoint: `checkpoints/pointer_gate1d_mean_e3.pt`。Gold V1は学習に未使用。

採用根拠はlinear probe（queue held-out 0.48、chance 0.20）。LM head zero-shot 0.32は
プロンプトで引き出せないだけ、と解釈した。

## 訓練ドメイン

| 指標 | Gate 1-c | Gate 1-d |
|---|---|---|
| AG News | 0.8650 | 0.8300 |
| banking77 | 0.7750 | 0.7800 |
| emotion | 0.5475 | 0.6175 |
| dbpedia14 | 0.9050 | 0.9225 |
| sst5 | 0.4550 / MAE 0.6435 | 0.4550 / MAE 0.6207 |
| 順序反転一致率 | 0.9500 | 0.9375 |
| bugzilla OOD τ | +0.423 (p=0.000) | +0.416 (p=0.000) |
| queue held-out | —（訓練前 0.285） | **0.470**（chance 0.298） |

queue自体は学習できている。bugzilla τは実質維持。AG Newsは0.035低下。

## Gold V1（停止条件の主判定）

| | Gate 1-c | Gate 1-d |
|---|---|---|
| JA Choice | **0.475** | 0.425（校正後 0.450） |
| JA Score | 0.233 | 0.167（校正後 0.333） |
| EN Choice | 0.475 | 0.350（校正後 0.525） |
| EN Score | 0.367 | 0.200（校正後 0.200） |
| Noul | 0.600 / 0.650 | 変化なし（noul checkpoint同一） |

**停止条件（JA Choice > 0.475 かつ in-domain / bugzilla τの退行なし）: 満たさない。**
英語Gold V1の校正後Choice 0.525は1-c校正後0.375より良いが、日本語の主判定は下がった。

## 解釈

隠れ状態にqueue信号はある（linear probe）。pointerもqueue held-outでは0.47まで上がる。
それがGold V1の日本語部署振り分けには載らない。英語校正後だけ上がるので、
残差はタスク構造の一致よりも言語／選択肢文面の差が大きい。

現行のGold V1評価用checkpointは `checkpoints/pointer_gate1c_mean_e3.pt` のまま。
