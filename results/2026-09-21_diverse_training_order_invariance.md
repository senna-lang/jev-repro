# 訓練データ多様化 — 順序反転耐性とドメイン外汎化（2026-09-21）

## 背景

`2026-09-19_official_compare_trained.md` / `2026-09-19_verifications.md` で診断した問題:

1. **飽和/calibrationバグ**: `PointerReadout`の`z = (W_q h_dec)·(W_k h_opt)/√d`が、生のbackbone
   隠れ状態のノルム(massive activation由来で`||h_dec||~250`)にそのまま比例し、訓練前後とも
   確率がほぼ0/1に飽和していた（訓練損失100〜270台）。
2. **順序過敏性**: AG News単一ドメイン・固定4択・固定順のみで訓練したcheckpoint
   (`pointer_agnews_s10000_e1.pt`)は、選択肢の順序を反転しただけでaccuracyが0.808→0.183に崩壊、
   base/reversed予測一致率9.2%。
3. **ドメイン外汎化の欠如**: 公式Jevドキュメント掲載のサポートチケット例に対し、Choice top-1一致
   2/8、確率はほぼone-hotのまま。

## 変更1: readoutのスケーリング修正

`model/readout.py`の`PointerReadout`に、projection前の`LayerNorm`（`h_dec`用・`h_opts`用それぞれ）
を追加。AG News 2,000件×1epochで再訓練した結果:

| | 訓練前 | 訓練後（旧/未修正） | 訓練後（LayerNorm修正後） |
|---|---:|---:|---:|
| accuracy | 0.282 | 0.818（旧run） | 0.806 |
| ECE | 0.027 | 0.180（旧run） | **0.109** |
| 訓練損失 | – | ~262(桁違いに大) | **1.78**（理論最小ln4≈1.386に近い） |

飽和は解消（held-out mean max-prob 0.9985→0.912）。ただし順序反転耐性はほぼ無変化
（acc_reversed=0.050、base/reversed一致率10%）— これはcausal attentionの構造的性質
（`<DECISION>`はbranch内で後続の選択肢しか見えない）であり、スケーリングバグとは無関係と
当初は判断した。

## 変更2: 訓練データの多様化（本命）

`data/common.py`に`ChoiceExample`（データセット非依存の型）と`reshuffled()`
（選択肢順をランダムに並べ替え、正解ラベルを追跡して付け替える）を追加。
`data/banking77.py`を新設（banking77: 銀行サポートチケットの意図分類、77クラスから
正解1つ＋ランダムdistractorをサンプルしてK=3〜6の可変選択肢問題に変換）。
`experiments/train.py`の`train_readout`に`shuffle_options`引数を追加し、有効時は
エポックごとに各exampleの選択肢順をシャッフルする。

`experiments/train_diverse.py`: AG News 2,000件＋banking77 1,500件（計3,500件）を混合し、
選択肢順シャッフル有効で1epoch訓練（LayerNorm修正済みreadout使用、backbone凍結は従来通り）。

### 結果

| 指標 | 訓練前 | 訓練後 | 参考: 単一ドメイン固定順(旧) |
|---|---:|---:|---:|
| AG News accuracy | 0.282 | 0.716 | 0.806（固定順のみ・タスクが単純） |
| AG News ECE | 0.027 | 0.156 | 0.109 |
| banking77 accuracy (K可変3-6, chance≈0.22) | 0.202 | 0.508 | 未測定（新規タスク） |
| banking77 ECE | 0.074 | 0.345 | 未測定 |
| **順序反転 base/reversed一致率** | 0.175 | **0.800** | **0.100** |

訓練: 3,500件×1epoch、CPU、1066秒（約17.8分）。train_acc=0.5197（loss 2.46、まだ下降中で
1epochでは未収束）。checkpoint: `checkpoints/pointer_diverse_agnews2000_banking1500_e1.pt`。

## 結論

**順序過敏性はアーキテクチャの構造的制約ではなく、訓練データの偏り（単一ドメイン・固定順）に
起因していた**。選択肢順をシャッフルし複数ドメイン・可変Kで訓練しただけで、base/reversed一致率
が10%→80%に改善した。「causal attentionの構造上直せない」という前回の仮説は誤りだったと訂正する。

トレードオフ: AG News単独精度は0.806→0.716に低下。ただし旧0.806は「4文字列パターンの丸暗記」に
よる脆い精度（順序を変えると0.183に崩壊）であり、今回はより難しい複合タスクを1epochのみで
打ち切った暫定結果（収束途中）。

## 次のステップ（未実施）

- epoch数を増やして収束させる（train_accが0.52で下降中に打ち切っているため、伸びしろが見込める）
- 収束後、公式Jevサポートチケット例（`experiments/official_compare.py`）で再評価し、
  billing/technical/salesのone-hot崩壊が改善するか確認する

## 再現コマンド

```bash
python3 -m experiments.train_diverse
```
