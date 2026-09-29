# 2026-09-24: Noul（yes/no）readout訓練 + Gold V1評価パイプライン構築

## 背景

`evaluation/evaluate_checkpoint.py`（GOAL.md §2-4準拠の評価オーケストレータ）をGold V1に対して実行したところ、Noulのthreshold_accuracyが実行のたびに異なる値（0.6→0.3→別値）を示した。

## 原因

`model/forward.py::JevModel.__init__`が`self.noul = NoulReadout(hidden_size)`を毎回ランダム初期化するが、**このプロジェクトのどの訓練スクリプトもNoulReadoutを訓練・保存していなかった**（`experiments/train.py`のdocstring「バックボーンとNoulReadoutは更新しない」の通り、yes/no訓練データが無かったため）。全checkpointは`model.pointer.state_dict()`のみを保存しており、Noul結果は訓練済みモデルの性能ではなく未訓練の乱数を測っていた。

## 対処

1. **データ**: `data/boolq.py` — `google/boolq`（読解yes/no QA、9427+3270件、62.3%がTrue）を、state=passage/instructions=question（先頭大文字化+"?"補完）/label=answerとしてNoul branchに変換。`load_boolq_stratified(n_per_label, seed, offset)`はラベルごとに均等にサンプルし、`offset`で train/held-out の非重複を保証（`bugzilla_severity`の前半/後半プール分割方式は別実装だが確認の結果こちらも重複なし、後述の訂正参照）。

2. **Q1チェック（Gate 0スタイル、訓練前）**: `experiments/boolq_probe.py` — 凍結backbone＋LM headで" Yes"/" No"の次トークンlog確率を比較。結果: accuracy=0.700, p<0.0001, n=120（yes_picks=44/120、偏りなし）。信号確認済みで訓練に着手。

3. **訓練**（`experiments/train_noul.py`）:
   - 素朴なランダム初期化からの訓練（600件×3epoch、lr=1e-3）: held-out accuracy=0.5300（ほぼchance）、停止条件未達。
   - 原因調査: LM headの出力層は既に`logit_yes - logit_no = (w_yes - w_no)·h_dec`というzero-shotで機能する方向を持っているが、`NoulReadout`（新規896→1の線形層）はこれをゼロから学習する必要があり、600件では学習不足。
   - **修正**: `_init_noul_from_lm_head` — `NoulReadout.linear.weight`を`lm_head.weight[" Yes"] - lm_head.weight[" No"]`で初期化（warm-start）。warm-start直後held-out accuracy=0.6050, brier=0.2359（訓練なしでこの時点で目標到達）。
   - その後lr=1e-4で3epoch微調整: train_acc 0.62→0.66→0.705、**held-out accuracy=0.6300, brier=0.2274**。停止条件（held-out accuracy>0.6 かつ warm-startを下回らない）達成。
   - 保存: `checkpoints/noul_boolq_e3.pt`（キー`noul_state_dict`、`model.pointer`とは独立）。

4. **評価パイプライン統合**: `evaluation/run.py`・`evaluation/evaluate_checkpoint.py`に`--noul-checkpoint`オプションを追加。未指定時はWARNINGを出しランダム初期化のまま動作（後方互換）。

## Gold V1再評価結果（`checkpoints/pointer_gate1c_mean_e3.pt` + `checkpoints/noul_boolq_e3.pt`）

`evaluation/private/runs/pointer_gate1c_mean_e3_with_noul/summary.json`。同一コマンドを複数回再実行し、Noul含む全指標が完全に決定論的（bit-for-bit一致）であることを確認。

| kind | n | 主指標 |
|---|---|---|
| Choice | 40 | top1_accuracy=0.475, brier=0.689, nll=1.262 |
| Score | 30 | top1_accuracy=0.233, brier=0.771, nll=1.338, MAE=0.743, EMD=0.409 |
| **Noul** | 20 | **threshold_accuracy=0.600**, brier=0.2095, nll=0.681（訓練前は乱数依存で無意味だった） |

頑健性・レイテンシは前回（noul未訓練時）と同一（Noul訓練はpointerに影響しない設計のため予想通り）:
- `single_branch_packing`/`branch_reorder`/`branch_add_unrelated`/`branch_remove_sibling`: top1_consistency=1.0, drift≈0
- `choice_option_reversal`: top1_consistency=0.725（位置バイアス残存、未調査）
- packed vs naive re-encode: branch数16でnaive/packed比 5.01×

## 残課題

- Choice/Score品質（top1 0.475/0.233）はGold V1（日本語）と訓練データ（全て英語）の言語・ドメインミスマッチが主因という仮説、未検証。
- ~~`bugzilla_severity`の train/held-out重複バグの修正とGate 1c再検証~~ — 2026-09-24 追記: 実装を再確認した結果、`load_bugzilla_severity_stratified`はプール全体を`n_per_level`に依存しない前半/後半インデックスで先に二分割してからストラティファイド抽出するため、重複バグは存在しなかった（実測: train 1800件・held-out 300件、重複2件=0.11%は定型タイトルの自然な偶然一致で split方式の欠陥ではない）。前回の指摘は誤りで、Gate 1cのOOD tau=+0.423は有効。
- `choice_option_reversal`の位置バイアス原因調査。
