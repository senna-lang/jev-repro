# 検証結果 — isolation / 並び順敏感性 / listwise相互作用（2026-09-19）

3つの構造検証の実測結果。いずれもbackbone（Qwen2.5-0.5B, 凍結）＋readoutのend-to-end実出力による。
学習前（ランダム初期化readout）でも成立するはずの、設計の構造的な性質の検証（並び順敏感性のみ学習前後比較）。

## 1. questionを追加・挿入しても、他のquestionの答えは変わらない（isolation）

- **測定方法**: 既存のchoice branch 1つに対し、他2question（noul・score）を先頭・中央・末尾に追加・挿入し、既存branchの確率を比較。
- **結果**: 複数回の実行で最大差分は 4.5e-16〜6e-4 の範囲、seed固定実行では 1.94e-05。許容誤差（atol=1e-3）の範囲内で不変。
- **誤差の正体**: 層ごとに調べると、embedding層（token id・position id適用直後）は常にビット単位で完全一致する一方、attention層を1つ通るごとに1e-6オーダーの丸め誤差が生まれ、24層で1e-4オーダーまで蓄積する。eager attentionはmaskで可視性を絞ってもQK^Tの計算自体を系列全体`(T, T)`に対して密に行うため、questionを追加してテンソル形状が変わると行列積の内部縮約順序が変わり、丸め誤差の経路がわずかに変わる（密attention実装特有の数値的非結合性）。mask・position idのロジックバグではない。
- **基準**: 以降の検証はすべて「この程度の誤差（〜1e-3）を除けば一致」という粒度で判定する。
- **実行出力**（seed固定実行の`python3 -m experiments.sanity_check`より）:

```text
=== isolation: questionを追加・挿入しても既存questionは不変か ===
branch_aのみ:              [0.028855524957180023, 0.9711413383483887, 1.1860998938573175e-06, 1.924616981341387e-06]
branch_aが先頭(他2つ付加): [0.028855524957180023, 0.9711413383483887, 1.1860998938573175e-06, 1.924616981341387e-06]
branch_aが中央:            [0.028855524957180023, 0.9711413383483887, 1.1860998938573175e-06, 1.924616981341387e-06]
branch_aが末尾:            [0.02883613109588623, 0.9711607694625854, 1.1863160125358263e-06, 1.925076048792107e-06]
最大差分: 1.943e-05（許容誤差 atol=1e-03）
isolation: 予測通り。他のquestionを何個・どこに追加・挿入しても、branch_aの確率は誤差1e-03の範囲内で不変（浮動小数点の丸め誤差を除き一致）。
```

- 再現: `python3 -m experiments.sanity_check`（isolationセクション）

## 2. 選択肢の並び順を変えると（内容は同じでも）出力確率が変わる（order sensitivity）

- **測定方法**: state 4種（合成テキスト1つ＋AG News held-out記事3つ）× 選択肢セット2種（短ラベル4択・criteria説明付き4択）の8ケースで、選択肢の中身・個数を変えず並び順だけを全24通り（4!）に変えて測定（集計対象は選択肢32個）。
- **学習前**（ランダム初期化readout）: 並び順だけで確率が大きく動いた——同一選択肢の確率の変化幅（max−min）は平均 0.974、最大 1.000、ノイズ床（1e-3）超え率 1.00。
- **軽い学習後**（AG News 200件×1 epoch、train_acc 0.43）: 平均 0.938、最大 1.000、超え率 0.94 と傾向は消えず。さらに学習後は**最も確率の高い選択肢自体が順列の17〜58%で入れ替わった**（top-1安定率 0.42〜0.83）。
- **解釈**: softmaxは単調なので、0/1への飽和だけではargmax（順位）を変えられない。top-1が入れ替わっているのはlogitの順位そのものが並び順で動いていることを意味し、未学習の飽和によるアーティファクトではなく構造的な性質。機構はbranch内causal attentionの非対称性（後ろの選択肢だけ前の選択肢を参照できる＋`<DECISION>`が見る文脈が並び順で変わる）。
- archerhumeのoption_order観測（並び順反転で0.84–0.89→0.93–0.96に変化）と同じ向きの現象。
- **実行出力**（seed固定実行の`python3 -m experiments.sanity_check`より）:

```text
=== P3: 選択肢の並び順だけを変えたときの出力確率（学習前後比較） ===
ケース: 8（state×選択肢セット）、各ケース全順列（4!=24通り）を測定

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
P3: 並び順で確率は動いた。学習後も平均・最大ともにatolを明確に上回る——未学習の飽和に依らない、branch内のcausal attention（後ろの選択肢だけ前の選択肢を参照できる非対称性）に由来する構造的な性質として扱える。
```

- 再現: `python3 -m experiments.sanity_check`（P3セクション、seed固定）

## 3. 無関係な選択肢を1つ追加しただけで、既存の選択肢どうしの優劣も変わる（listwise相互作用）

- **測定方法**: 2択のchoice questionに無関係な第3選択肢を追加し、既存2択のlog-oddsの変化を測定。
- **JevModel（pointer readout）**: log-oddsが必ず非ゼロだけ動く（符号・大きさは実行毎＝readout初期化毎に変わる。実測例: −18.55、−5.62）。機構: `<DECISION>`の隠れ状態 h_dec はbranch内の全選択肢を読んだ後に置かれるため選択肢追加で再計算される一方、既存選択肢の h_opt_i はcausal maskで未来を見ないため不変。よって log(p1/p2) = (W_q h_dec)·W_k(h_opt1 − h_opt2) が h_dec の変化分だけ動く。
- **independent scorer（対照）**: 優劣はまったく動かなかった（log-odds差 = 厳密に0、ビット単位で不変）。各 h_opt_i が他の選択肢を一切見ないためsoftmaxの分母がキャンセルし、数式上保証されたとおりの結果。
- 即ち「候補をまとめて見ている」ことを、独立採点では原理的に再現できない形で実証。
- **実行出力**（seed固定実行の`python3 -m experiments.sanity_check`より）:

```text
=== P2: 無関係な選択肢を追加したときのlog-odds ===
JevModel (pointer方式):     log-odds before=-1.361901  after=-6.979677  diff=-5.617776
IndependentScorer (対照): log-odds before=-8341.929688  after=-8341.929688  diff=+0.000000

JevModel diff = -5.617776 (0から離れているほど、listwise相互作用が効いている)
IndependentScorer diff = +0.000000 (理論上ちょうど0のはず)
P2: 予測通り。IndependentScorerのlog-oddsは不変（0）、JevModel（pointer方式）は動いた。
```

- 再現: `python3 -m experiments.sanity_check`（P2セクション）
