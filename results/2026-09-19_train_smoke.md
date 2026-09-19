# train 実行結果（2026-09-19、smoke）

- コマンド: `python3 -c "from experiments.train import _main; _main(train_size=8, eval_size=8, epochs=1)"`
- 目的: experiments/への移動後の入口確認用smoke。サイズ8件は本来の設定ではなく、数値自体には意味がない。

```text
train_size=8, eval_size=8, epochs=1

=== 訓練前 ===
accuracy=0.1250, ECE=0.7339 (n=8)

=== 訓練 ===
epoch 0: loss=268.4870, train_acc=0.2500, 3.5s

=== 訓練後 ===
accuracy=0.1250, ECE=0.8731 (n=8)

正解率: 0.1250 -> 0.1250 (悪化/不変)
ECE:    0.7339 -> 0.8731 (悪化/不変)

正解率が改善しなかった。epoch数またはデータ量を増やして再測定する。
```

所感: 移動後も訓練ループ・評価が壊れず動作。full設定での結果は未取得。
