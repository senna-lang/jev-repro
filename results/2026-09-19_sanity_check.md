# sanity_check 実行結果（2026-09-19）

- コマンド: `python3 -m experiments.sanity_check`（リポジトリルートから）
- 環境: macOS / CPU、Qwen2.5-0.5B float32

```text
=== isolation: questionを追加・挿入しても既存questionは不変か ===
branch_aのみ:              [0.9998998641967773, 0.00010004567593568936, 4.230483074962876e-09, 3.834410833292168e-08]
branch_aが先頭(他2つ付加): [0.9998998641967773, 0.00010004567593568936, 4.230483074962876e-09, 3.834410833292168e-08]
branch_aが中央:            [0.9998998641967773, 0.00010004567593568936, 4.230483074962876e-09, 3.834410833292168e-08]
branch_aが末尾:            [0.9998998641967773, 0.00010009825928136706, 4.232299399831163e-09, 3.8358958676099064e-08]
最大差分: 5.258e-08（許容誤差 atol=1e-03）
isolation: 予測通り。他のquestionを何個・どこに追加・挿入しても、branch_aの確率は誤差1e-03の範囲内で不変（浮動小数点の丸め誤差を除き一致）。

=== P1: questionを増やしたときのforward時間 ===
N=  1  tree_mask=   117.5ms  naive_reencode=   130.7ms  naive/tree=1.11x
N=  2  tree_mask=   145.9ms  naive_reencode=   245.3ms  naive/tree=1.68x
N=  4  tree_mask=   184.8ms  naive_reencode=   502.8ms  naive/tree=2.72x
N=  8  tree_mask=   262.4ms  naive_reencode=  1017.8ms  naive/tree=3.88x
N= 16  tree_mask=   509.7ms  naive_reencode=  2117.3ms  naive/tree=4.15x

N=1->N=16: tree_mask time growth = 4.34x, naive_reencode time growth = 16.20x
P1: 予測通り、tree mask版の方が緩やかに増えている。

=== P2: 無関係な選択肢を追加したときのlog-odds ===
JevModel (pointer方式):     log-odds before=-6.811636  after=-25.359226  diff=-18.547590
IndependentScorer (対照): log-odds before=-8341.929688  after=-8341.929688  diff=+0.000000

JevModel diff = -18.547590 (0から離れているほど、listwise相互作用が効いている)
IndependentScorer diff = +0.000000 (理論上ちょうど0のはず)
P2: 予測通り。IndependentScorerのlog-oddsは不変（0）、JevModel（pointer方式）は動いた。

=== まとめ === isolation: OK / P1: OK / P2: OK
```

所感: 3項目すべてOK。isolationの最大差分5.258e-08はfloat32の丸め誤差の桁。
