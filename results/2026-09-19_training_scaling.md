# 訓練結果 — データ量スケーリング 0〜20,000件（2026-09-19）

- **設定**: 訓練対象は`PointerReadout`のみ（backbone凍結）。損失はcross-entropy、Adam lr=1e-3、batch_size=1（paddingなし）。各サイズでseed=0でreadoutを作り直し、**1 epoch**で統一（compute-per-exampleを揃え、データ量の効果だけを見る）。評価はAG News test split held-out 500件、ECEは10分割。タスクはstate=記事本文・選択肢=4クラス（criteria説明付き）のChoice分類。

| 訓練データ量 | accuracy | ECE |
|---:|---:|---:|
| 0（訓練前・ランダム初期化） | 0.2260 | 0.5664 |
| 500 | 0.6200 | 0.3779 |
| 2,000 | 0.8180 | 0.1800 |
| 10,000 | **0.8300** | **0.1706** |
| 20,000 | 0.7720 | 0.2280 |

- **0→10,000件までは正解率・ECEとも単調に改善**。20,000件では両方悪化——1 epoch・batch_size=1・学習率固定・シャッフルseed未固定・1シードのみの実行のため、「データ量の効果」と「たぶんの勾配ノイズ」が分離できていない。非単調性はそのまま正直に記録する。
- 別設定（full post-train）: 1,000件×3 epochで accuracy 0.2920→0.7420、ECE 0.6363→0.2571（両方改善、約18.4分）。
- 実行時間: スケーリング全サイズ合計 約3時間5分（CPU）。
- 観察: 訓練損失は終始100〜270台と大きい——未学習のpointer readoutが0/1近傍に飽和した確率を出すため`-log(p_label)`が跳ねる。バグではなく、飽和したdot-product readoutの初期状態の特性（order sensitivityの検証でも同一の飽和が観測されている）。
- `NoulReadout`はAG Newsにyes/noの訓練データがなく未訓練のまま（意図的な制約）。
- 再現: `python3 -m experiments.scaling_curve`（旧スケーリング）/ `python3 -m experiments.train`（単一設定。訓練後のreadoutは`checkpoints/`に保存）

## 再現可能な10,000件checkpoint再訓練

旧スケーリングのベスト設定（10,000件×1 epoch）を、readout初期化とエポック内シャッフルの両方をseed=0に固定して再訓練した。

| | accuracy | ECE |
|---|---:|---:|
| 訓練前 | 0.2260 | 0.5664 |
| 訓練後 | **0.8040** | **0.1980** |

- train accuracy: 0.7448
- epoch loss: 262.9216
- 訓練時間: 2,848.4秒（約47.5分）、訓練前後評価込みwall time: 3,129.43秒（約52.2分）
- checkpoint: `checkpoints/pointer_agnews_s10000_e1.pt`（約6.1MB、gitignore対象）

```text
train_size=10000, eval_size=500, epochs=1, seed=0

=== 訓練前 ===
accuracy=0.2260, ECE=0.5664 (n=500)

=== 訓練 ===
epoch 0: loss=262.9216, train_acc=0.7448, 2848.4s

=== 訓練後 ===
accuracy=0.8040, ECE=0.1980 (n=500)

正解率: 0.2260 -> 0.8040 (改善)
ECE:    0.5664 -> 0.1980 (改善(低下))
checkpoint: checkpoints/pointer_agnews_s10000_e1.pt
```

## 再現性についての注記

上段の旧scaling実行はエポック内シャッフル（`random.shuffle`）がseed未固定だった。今回のcheckpointはreadout初期化に加えてシャッフルもseed=0で固定したため、同一設定・同一データでも旧ベスト値0.8300/0.1706と完全一致する重み・数値にはならない。同条件の再現可能な再訓練であり、旧実行のcheckpoint復元ではない。
