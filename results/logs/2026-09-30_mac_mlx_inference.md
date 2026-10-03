# Mac（M2）での推論高速化: LoRA 統合 + fp16 + MLX（2026-09-30）

対象: E2 選定 LoRA（`checkpoints/e2/lr-1e-4/epoch-3-quarter-3/`、Qwen3-4B-Instruct-2507）。
環境: Apple M2、メモリ 24GB、torch 2.10.0（MPS）、transformers 4.57.6、peft 0.18.1、mlx 0.31.1、mlx-lm 0.31.2。
`e2/` の本体コードは変更していない。Mac 版は `mac/` に分けた。

## 結論

- Mac 版の推論は `mac/jev_mlx.py`（`JevMLX`）で行う。LoRA を本体に統合して fp16 で持ち、MLX 上で tree mask 付きの forward を1回行う。
- 順番平均の1件あたりの時間は、いまの構成（PyTorch MPS、bf16、LoRA 未統合）で約 3.0s、**MLX 版で 0.87s（約 3.5 倍速）**。
- 精度は下がっていない。教師テスト eval の先頭 240 branch で、A100 の予測と argmax が 99.6% 一致した（1件の食い違いは 0.492 対 0.493 のほぼ同点）。確率差は最大 0.025。
- 4bit/8bit 量子化と、1回推論（順番平均なし）への切り替えは、精度が変わるため採らなかった。

## レイテンシ（順番平均、ユースケース日英10問、平均 276 tok、各5回の中央値）

| 設定 | 1件あたり |
|---|---:|
| PyTorch MPS、bf16、LoRA 未統合（いまの構成） | 3.22s / 2.79s（測定順 1回目 / 6回目） |
| PyTorch MPS、LoRA 統合、bf16 | 2.63s / 2.57s |
| PyTorch MPS、LoRA 統合、fp16 | 2.05s / 1.86s |
| **MLX、LoRA 統合、fp16** | **0.87s** |

- PyTorch の3設定は、ドリフトを打ち消すため baseline → bf16 → fp16 → fp16 → bf16 → baseline の順で測った。
- forward 以外（前処理、転送、後処理）は合計 10ms 以下。
- MLX 版はトークン数にほぼ比例する（128 tok で 357ms、394 tok で 1,216ms）。残りはほぼ計算そのもの。
- 測定中は swap が効いていた（最大 17GB 使用）。時間の絶対値は環境によって変わる。
- 教師 eval のリクエストは長い（state 約 1,500 tok、12〜14 branch、合計 2,200〜2,800 tok）。`python -m mac.bench latency --limit 3` では 1件あたり中央値 25.4s だった（swap 使用 13.8GB の状態）。同じ 60 リクエストの eval にかかった時間は、MLX で 919s、PyTorch fp16 で 4,876s。
- 1件あたりの時間はトークン数より速く増える（276 tok → 約 2,500 tok でトークンは約 9 倍、時間は約 29 倍）。T×T の注意マスクと swap の影響と見ているが、内訳は測っていない [INFERENCE]。

## 精度（教師テスト eval の先頭 60 リクエスト = 240 branch、順番平均）

固定候補の評価で、選定には使わない。Gold V1 には触れていない。全 1,200 branch は、MPS で1設定あたり約 4 時間かかるため流していない。

| 設定 | top-1 | Brier | NLL | ECE | A100 との argmax 一致 | A100 との確率差（最大 / 平均） |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch MPS、いまの構成 | 0.8875 | 0.0355 | 0.319 | 0.027 | 100% | 0.044 / 0.0019 |
| PyTorch MPS、LoRA 統合、bf16 | 0.8875 | 0.0354 | 0.320 | 0.026 | 100% | 0.044 / 0.0028 |
| PyTorch MPS、LoRA 統合、fp16 | 0.8875 | 0.0352 | 0.316 | 0.023 | 100% | 0.026 / 0.0017 |
| **MLX、LoRA 統合、fp16** | 0.8917 | 0.0351 | 0.317 | 0.026 | 99.6% | 0.025 / 0.0021 |

- A100 の値は `.scratch/usecase_demo/single_vs_avg.json` の順番平均の予測（全体の値は `2026-09-28_latency_breakdown.md`）。
- MLX と PyTorch fp16 の差は、最大 0.019、平均 0.001。
- MLX の top-1 がわずかに高いのは、ほぼ同点の1件が教師の正解側に倒れたため。精度が上がったとは言えない。

## 実装（`mac/`）

- `mac/jev_mlx.py` の `JevMLX`
  - 読み込み時に `W + (alpha/r)·B·A` を fp32 で計算し、1回だけ fp16 に変換する。
  - 層を回す部分は自前で書いている。`mlx_lm` の Qwen3 の `__call__` は、因果マスクと連続した RoPE の位置にしか対応していないため。
  - tree mask は bool のまま `mx.fast.scaled_dot_product_attention` に渡す。RoPE は `position_ids` から fp32 で cos/sin を計算する（HF と同じ rotate_half の規則）。
  - `lm_head` は答えの位置にだけかける。
  - `predict_request(state, branches, orders=None)` は `e2.evaluate_lora.predict_request` と同じ入出力。`orders=None` なら1回推論、`option_orders(...)` を渡せば順番平均になる。パッキング、mask、答えのトークン ID は `e2` と `model` の関数をそのまま使っている。
- `mac/bench.py`: 教師 eval split でレイテンシと精度を測る。
  - `python -m mac.bench latency --limit 10`
  - `python -m mac.bench eval --limit 60 --out results/artifacts/mac/eval_mlx_fp16.json`
  - `--single` を付けると1回推論になる。

## 成果物

`results/artifacts/mac/`
- `latency_*_r{1..6}.json`: PyTorch の3設定のレイテンシ（測定順）
- `latency_mlx_fp16_r1.json`: MLX 版のレイテンシ
- `eval_{baseline,merged_bf16,merged_fp16,mlx_fp16}.json`: 240 branch の予測と指標
