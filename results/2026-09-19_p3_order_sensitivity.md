# P3 order sensitivity 実行結果（2026-09-19）

- コマンド: `python3 -m experiments.sanity_check`（リポジトリルートから、seed固定で再現可能）
- 環境: macOS / CPU、Qwen2.5-0.5B float32
- 設計: 8ケース（state 4種 × 選択肢セット2種）× 全順列24通り。stateは合成テキスト1つ＋AG News test split（held-out）3記事。選択肢セットは短ラベル4択とcriteria説明付き4択。学習前後で同じ8ケースを測定（訓練: AG News train split 200件×1 epoch、readoutのみ・backbone凍結）。

```text
--- 学習前（ランダム初期化のreadout） ---
  synthetic×short4         range mean=0.7915 max=1.0000  top1安定率=0.25  平均確信度=0.9780
  synthetic×criteria4      range mean=1.0000 max=1.0000  top1安定率=0.96  平均確信度=0.9793
  AGNews#0×short4          range mean=1.0000 max=1.0000  top1安定率=0.12  平均確信度=0.9719
  AGNews#0×criteria4       range mean=1.0000 max=1.0000  top1安定率=1.00  平均確信度=1.0000
  AGNews#1×short4          range mean=1.0000 max=1.0000  top1安定率=0.92  平均確信度=0.9698
  AGNews#1×criteria4       range mean=1.0000 max=1.0000  top1安定率=1.00  平均確信度=0.9946
  AGNews#2×short4          range mean=0.9974 max=1.0000  top1安定率=0.17  平均確信度=0.9985
  AGNews#2×criteria4       range mean=1.0000 max=1.0000  top1安定率=1.00  平均確信度=0.9999
  --> 統計（選択肢32個）: range mean=0.9736 median=1.0000 max=1.0000 atol(1e-03)超え率=1.00

--- 軽い訓練（AG News train split 200件×1 epoch、readoutのみ・backbone凍結） ---
  epoch 0 [100/200] running_loss=236.8692
  epoch 0 [200/200] running_loss=192.5544
epoch 0: loss=192.5544, train_acc=0.4300, 47.9s

--- 学習後 ---
  synthetic×short4         range mean=1.0000 max=1.0000  top1安定率=0.83  平均確信度=0.9943
  synthetic×criteria4      range mean=1.0000 max=1.0000  top1安定率=0.42  平均確信度=0.9869
  AGNews#0×short4          range mean=0.7500 max=1.0000  top1安定率=0.42  平均確信度=0.9894
  AGNews#0×criteria4       range mean=1.0000 max=1.0000  top1安定率=0.62  平均確信度=0.9945
  AGNews#1×short4          range mean=1.0000 max=1.0000  top1安定率=0.46  平均確信度=0.9938
  AGNews#1×criteria4       range mean=1.0000 max=1.0000  top1安定率=0.42  平均確信度=0.9778
  AGNews#2×short4          range mean=0.7500 max=1.0000  top1安定率=0.58  平均確信度=0.9995
  AGNews#2×criteria4       range mean=1.0000 max=1.0000  top1安定率=0.62  平均確信度=0.9964
  --> 統計（選択肢32個）: range mean=0.9375 median=1.0000 max=1.0000 atol(1e-03)超え率=0.94

確信度（全ケース平均の最大確率）: 学習前=0.9865 -> 学習後=0.9916
並び順による確率変化幅: 学習前 mean=0.9736/max=1.0000 -> 学習後 mean=0.9375/max=1.0000（ノイズ床 atol=1e-03）
P3: OK
```

## 解釈

1. **並び順敏感性は学習後も消えない** — 変化幅 mean=0.94 / max=1.0、atol(1e-3)超え率0.94。ノイズ床を3桁以上上回る。
2. **飽和アーティファクトとの切り分け** — 軽い訓練（200件×1 epoch）では確信度はむしろ上がり（0.9865→0.9916）、0/1飽和自体は解消されなかった。しかし**softmaxは単調なので飽和はargmaxを変えられない**。top1安定率が学習後に0.42〜0.83（=順列の17〜58%で最高位選択肢が入れ替わる）であることは、logitの順位そのものが並び順で動いていることを意味し、飽和では説明できない構造的な効果。機構はbranch内causal attentionの非対称性（後ろの選択肢だけ前の選択肢を参照できる＋`<DECISION>`が見る文脈が並びで変わる）。
3. **確率の変化幅（≈1.0）自体は飽和に増幅されている** — 飽和が解けた状態での「適度な」変化幅を見るには、より強い訓練（full設定: 2,000件×3 epoch）での再測定が今後の課題。top1入れ替わりの有無はすでに本実験で確定している。
4. 同一run内でisolation / P1 / P2も再確認（すべてOK。isolation最大差分1.943e-05、P1: tree 4.30x vs naive 19.57x、P2: scorer diff=0・JevModel diff=-5.618）。

archerhumeのoption_order観測（並び順反転で確率が0.84–0.89→0.93–0.96に動く）と同じ方向性:
並び順は出力確率を実質的に動かす。
