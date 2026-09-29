# 選定 LoRA のレイテンシ内訳（2026-09-28）

対象: E2 選定 LoRA（`checkpoints/e2/lr-1e-4/epoch-3-quarter-3/`、Qwen3-4B-Instruct-2507）。
入力: ユースケース5問 × 日本語・英語（`.scratch/usecase_demo/run.py` の `CASES`）。学習・選定には使わない。

## 結論

- 順番平均が遅く見えた原因は **CPU スレッドの過剰確保**。Runpod の Pod（割り当て 32 vCPU）で PyTorch がホストの 255 コアを見てスレッドを 255 本立て、CPU 上の 4D tree mask 生成（`build_tree_attention_mask`）が約 400ms かかっていた。`torch.set_num_threads(1)` で 0.2ms。
- スレッドを 8 本に固定すると、順番平均は 1 件 **73.7ms**（forward 71.0ms）。1回推論（67ms）とほぼ同じ。
- attention の実装（bool mask で遅いカーネルに落ちる）という事前の仮説は外れ。mask の有無の差は約 4ms。
- 本番で使うときは推論前に `torch.set_num_threads(...)` を vCPU 数以下に固定する。`e2/` の本体コードは未変更。

## 測定値（A100 SXM 80GB、torch 2.8.0+cu128、transformers 5.17.0、peft 0.21.0、スレッド 8、中央値）

| 条件 | 合計 | forward |
|---|---:|---:|
| 順番平均（現行） | 73.7ms | 71.0ms |
| mask なし（同じトークン列） | 70.0ms | 67.2ms |
| 答えの位置を1つだけ | 73.1ms | 70.8ms |
| 後処理なし | 73.0ms | 71.3ms |

前処理 1〜2ms、GPU 転送 0.3ms 以下。

並びの数（Choice/Score 8件）: 2→70.5ms（195 tok）、3→70.6ms、4→71.6ms（296 tok）、6→74.2ms、8→76.2ms（498 tok）。
4 つの並びを別々に 4 回 forward: 253.8ms（まとめると 70.3ms、約 3.6 倍差）。

数百トークンでは forward は約 70ms の固定時間でほぼ決まり、並び（branch）を増やしてもほぼ横ばい。Archer Hume 氏が Jev で観測した「約 100 問までほぼ横ばい」と同じ形。

## Jev との比較（同じ10入力、1問ずつ送信）

Jev のサーバー内時間は応答ヘッダ `x-envoy-upstream-service-time`（Hume 氏と同じ指標。Cloudflare の先の上流サービスの時間で、共有サービスの待ちを含む）で測った。

| | 中央値 | 範囲 |
|---|---:|---|
| 選定 LoRA（A100、Pod 内、スレッド固定、順番平均） | 73.7ms | – |
| Jev サーバー内時間（22:3x JST 1回目、成功分の先頭4件） | – | 52〜249ms |
| Jev サーバー内時間（2回目、成功24件） | 300ms | 88〜2,661ms（p90 901ms） |
| Jev 手元からの合計（12:xx JST、健全時） | 259ms | 210〜333ms |

- 2回目の測定時、Jev は 30 件中 6 件が 503（upstream connect error）で、遅延も大きかった。Jev 側が不安定な時間帯だった。
- 手元（東京）から Jev の入口（Cloudflare、NRT）までの TCP 接続は約 11ms。Jev の遅さは通信ではなくサーバー側。
- LoRA を手元から呼ぶ構成は未測定。US-MD-1 の Pod までの TCP 接続は約 181ms なので、米国の Pod に置くと合計は約 250ms 前後になる見込み [INFERENCE]。日本リージョン（AP-JP-1 など）に置けば通信は数十 ms で済む見込み [INFERENCE]。

## 1回推論（元の並びのみ）と順番平均の精度差（教師テスト eval 1,200 branch）

A100、同じ環境・同じ checkpoint。順番平均は E2 報告値と完全一致（top-1 0.8792 / Brier 0.03685 / NLL 0.3264 / ECE 0.0295）。固定候補の評価で、選定には使わない。

| | top-1 | Brier | NLL | ECE |
|---|---:|---:|---:|---:|
| 順番平均 | 0.8792 | 0.0369 | 0.326 | 0.030 |
| 1回推論 | 0.8767 | 0.0391 | 0.344 | 0.032 |

kind 別 top-1（順番平均 → 1回推論）: Choice 0.945 → 0.947、Score 0.777 → 0.777、Noul 0.935 → 0.924。
argmax が両者で一致した branch は 95.7%。結果は `.scratch/usecase_demo/single_vs_avg.json`、スクリプトは `single_vs_avg.py`。

日英5問（1回推論 / 順番平均、%）: 1 JA 0/6/94/0 と 0/27/72/0、1 EN 0/62/38/1 と 0/87/13/0。2〜4 は同じ答え。5 JA 60/4/36/1 と 52/4/44/1（どちらも誤答）、5 EN 1/0/98/1 と 1/1/98/0。

## 参考: 以前の測定（修正前）

| | 中央値 |
|---|---:|
| Mac M2（MPS）1回推論 / 順番平均 | 1.72s / 2.86s |
| A100 1回推論 / 順番平均（スレッド未固定） | 67ms / 424ms |
| Jev API（手元から、通信込み） | 259ms |

Jev の値は通信込み、A100 の値は Pod 内の時間なので同じ条件の比較ではない。

## 未確認

- forward の固定 70ms の内訳（カーネル起動の負担、LoRA 未統合など）。
- 1回推論と順番平均の精度差。

## 成果物

- `.scratch/usecase_demo/latency_breakdown.py`（計測）、`.scratch/usecase_demo/profile_results.json`（結果）
- `.scratch/usecase_demo/latency.py`（最初の測定）、`.scratch/usecase_demo/results.json`（Jev と LoRA の出力比較）
- 使用 Pod `atvfw2u9y5jzkh`・`b1mg41nqc017i3` は削除済み。
