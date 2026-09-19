# scaling_curve 実行結果（2026-09-19、smoke）

- コマンド: `python3 -c "from experiments.scaling_curve import run_scaling_curve; run_scaling_curve(sizes=[8], eval_size=8)"`
- 目的: experiments/への移動後の入口確認用smoke。サイズ8件は本来の設定ではなく、数値自体には意味がない。

```text
size=0 (訓練前): accuracy=0.2500, ECE=0.5638

=== size=8 ===
  epoch 0 [1/8] running_loss=0.8958
  epoch 0 [2/8] running_loss=170.5886
  epoch 0 [3/8] running_loss=113.7257
  epoch 0 [4/8] running_loss=85.2943
  epoch 0 [5/8] running_loss=68.2355
  epoch 0 [6/8] running_loss=56.8629
  epoch 0 [7/8] running_loss=87.2131
  epoch 0 [8/8] running_loss=158.6349
epoch 0: loss=158.6349, train_acc=0.5000, 3.6s
size=8: accuracy=0.2500, ECE=0.7500, train_time=3.6s
```

所感: `experiments.train`の再利用（`_build_examples`/`evaluate`/`train_readout`）が移動後も正しく動作。
full設定（500/2,000/10,000/20,000）の結果は未取得。
