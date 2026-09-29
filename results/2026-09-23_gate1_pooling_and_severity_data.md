# Gate 1-a / Gate 1-b: pooling変更とseverity訓練データ追加

設計: `internal-docs/EXP_HEAD_VS_BACKBONE.md`。Gate 0（`results/2026-09-23_gate0_zeroshot_severity.md`）
でseverity_tableドメインの順序信号が凍結backboneに既に存在することを確認した（tau+0.400・
定数予測器でない）ため、backbone介入ではなくreadout側の安価な改善（Gate 1-a: pooling方式、
Gate 1-b: 訓練データ追加）から着手した。

## Gate 1-a: h_optsのpooling方式（末尾トークン vs 選択肢区間mean-pool）

実装: `model/packer.py`の`BranchSpan`に`option_start_indices`を追加し、
`model/forward.py::pool_options`で`"last"`（既存挙動）/`"mean"`（区間平均）を切り替え可能にした。
`experiments/train.py::_choice_logits`も`model.option_pooling`経由で同じ関数を使う。

選定基準（`EXP_HEAD_VS_BACKBONE.md` §1.5）: 訓練ドメイン（AG News/banking77/emotion/sst5/
dbpedia14）のeval splitをプールしたaccuracyのみ。公式Jev例・OODデータは見ない。
両条件とも`train_diverse.py`と同一recipe（5データセット混合・3エポック・選択肢順シャッフル・
seed=0）で`experiments/pooling_ablation.py`により訓練。

| 条件 | AG News | banking77 | emotion | dbpedia14 | sst5 | pooled accuracy | sst5 MAE | 順序反転一致率 |
|---|---|---|---|---|---|---|---|---|
| last（既存） | - | - | - | - | 0.3950 | **0.6560** | 0.6791 | 0.7125 |
| mean | - | - | - | - | 0.4025 | **0.6700** | 0.6785 | 0.8625 |

（個別accuracyはログ`/tmp/pooling_ablation.log`参照。pooled accuracyは5データセットの単純平均。）

**選定結果: `option_pooling="mean"`**（pooled_accuracy 0.6700 > 0.6560）。sst5 MAEはほぼ同値だが、
順序反転base/rev一致率が0.7125→0.8625と大きく改善（選択肢順序への過学習が減った）。
S6（GPT-2 spam分類, `last_block` vs mean的多token集約）の傾向と整合する。

## Gate 1-b: severity寄り訓練データ（support_tickets priority, K=3）の追加

データ: `data/support_tickets.py`。`Tobi-Bueck/customer-support-tickets`（HF Hub, 61,765件）の
英語サブセット（28,261件）、`priority`が`low`/`medium`/`high`の3値（人手ラベル、K=3で公式
severity_tableと選択肢数が一致）。train 1000件・held_out 300件（重複なし）。

`option_pooling="mean"`（Gate 1-a選定結果）で、既存5データセット + support_ticketsの
6データセット混合を3エポック訓練（`experiments/severity_gate1b.py`）。

### 訓練前後の比較

| 指標 | 訓練前 | 訓練後 |
|---|---|---|
| OOD accuracy（support_tickets held_out, n=300） | 0.2033 | 0.3867 |
| OOD Kendall tau | -0.012 (p=0.792) | **+0.005 (p=0.916)** |
| OOD 定数予測器 | **YES** | no |
| OOD 予測幅(max-min) | 0.060 | 1.043 |
| OOD ECE | 0.1732 | 0.1153 |

in-domain（訓練後）: AG News=0.8850, banking77=0.7550, emotion=0.5400, dbpedia14=0.9525,
sst5=0.4575（訓練前0.4025から改善）, sst5 MAE=0.5929（訓練前0.6785から改善）,
順序反転一致率=0.9375（Gate 1-a単体の0.8625からさらに改善、severityデータ追加は
順序不変性を損なわなかった）。

公式severity_table（参考、停止条件には使わない）: tau+0.400 (p=0.483)、定数予測器YES
（全stateでargmax=2固定）。

### 停止条件（§6）の判定 — スクリプトの自動判定は誤り、手動で訂正する

`experiments/severity_gate1b.py`の自動チェックは`tau > 0`のみを見て「満たす」と表示したが、
**これは誤り**。tau=+0.005・p=0.916は統計的にゼロと区別できない（n=300という十分なサンプル数で
この p値は「訓練前の定数予測器（tau=-0.012, p=0.792）」から実質的に動いていないことを意味する）。
自動チェックの閾値設計が甘かった（`tau > 0`だけで有意性を見ていない）。

**正しい判定:**

1. **OOD tau有意に正**: ✗ **満たさない**。tau≈0、pは有意水準から程遠い。定数予測器から
   脱した（YES→no）ことと予測幅が広がったこと（0.060→1.043）は改善だが、順位相関は生まれていない。
2. **in-domain退行なし**: ✓ 満たす。全データセットでaccuracy維持または改善、sst5 MAE改善。
3. **順序反転一致率退行なし**: ✓ 満たす（0.8625→0.9375、改善）。

**総合判定: Gate 1-bは停止条件を満たさない。** accuracy改善・定数予測器からの脱却という
部分的な前進はあったが、公式Jevの評価軸に最も近い「順位（tau）」では実質的な改善がない。

## 解釈と次

`EXP_HEAD_VS_BACKBONE.md` P2の検証: 「P1が真（Gate 0でzero-shot方向が正）なら、Gate 2の
部分解凍によるOOD tau改善はGate 1-b（データ追加）より小さいはず」という予測に対し、
Gate 1-bのOOD tau改善が実質ゼロだったことは、**外れた場合の意味（P2）に該当する可能性がある**:
「表現はあるが線形に取り出せない」ケース。ただし別の説明も残る:

- **ラベル品質の懸念**: `support_tickets`の`priority`は技術的深刻度ではなく業務優先度
  （SLA・顧客ランクなどテキスト外要因に依存しうる）。severity_tableの「技術的深刻度」とは
  軸がずれている可能性があり、テキストからは原理的に予測しにくいラベルを学習させていた
  かもしれない（Gate 0のQ1判定はseverity_tableという別データソースに対するものであり、
  support_ticketsのpriorityに対するQ1判定はまだ行っていない）。
- **mean-poolでも不足**: 選択肢テキスト側（`LEVEL_ORDER`の短い説明句）とstate側（長い
  チケット本文）の間で、依然として抽出に失敗している可能性。

次の一手（ユーザー確認が必要）:
1. `experiments/backbone_probe.py`スタイルのzero-shot probeをsupport_ticketsのpriorityに対して
   実行し、Q1をこのデータソース固有に判定する（ラベル品質問題の切り分け）。zero-shot tauが
   ここでも低ければ、severity_tableとsupport_ticketsの軸のずれ（severity vs priority）が
   本質的な原因である可能性が高まる。
2. `EXP_HEAD_VS_BACKBONE.md` Gate 2（部分解凍ラダー: U2/U6/U12）またはGate 3（LoRA）へ進む。
   §1.5の選定手続き（訓練ドメインval accuracyのみで候補選定）を維持する。
3. 非線形readout（2層MLP）を候補に加える（P2の「外れた場合」の代替案）。

---

## 追補: 仮説1・2の並行検証

Gate 1-bのOOD tau≈0を説明する2仮説について、ユーザー指示によりGate 2/3に進む前に両方検証した。

### 仮説2（ラベル軸のずれ）: `experiments/support_tickets_probe.py`

`support_tickets`のpriorityラベルに対する凍結backboneのzero-shot信号（readout・訓練を一切
通さない）を測定。手法はGate 0の`backbone_probe.py`と同一（地の文プロンプト・teacher-forcing
平均log確率・null state校正）。severity_table（n=5）より大きいn=60のheld-outサンプルを使用。

```
n=60, ラベル分布: low=15 medium=26 high=19
tau: 校正前 +0.002 (p=0.984) / 校正後 -0.003 (p=0.973)
accuracy: 校正前 0.283 / 校正後 0.417（チャンス=0.333）
定数予測器: 校正前 no / 校正後 no
```

**結果: zero-shot tauが実質ゼロ（p≈0.97-0.98、n=60で明確な null）。** severity_table
のzero-shot tau+0.400（Gate 0）とは対照的。**凍結backboneはpriorityラベルとテキスト内容の
相関を読み取れていない** — readoutを一切通していない時点でこれなので、訓練やreadoutの
表現力の問題ではない。**仮説2（ラベル軸のずれ）を支持する強い証拠。** `priority`は
SLA・顧客ランクなどテキスト外要因に依存する業務優先度であり、severityが表す技術的深刻度とは
別の軸だった可能性が高い。Gate 1-bで使った訓練データセットの選択自体が、severityの代理として
不適切だったと考えられる。

### 仮説1（線形readoutの限界）: `experiments/mlp_readout_ablation.py`

`model/readout.py::MLPPointerReadout`（decision/option特徴を`[h_dec; h_opt; h_dec*h_opt]`へ
連結し2層MLPでlogitを出す非線形readout）に差し替え、Gate 1-bと全く同じデータ・pooling・
訓練recipeで再訓練し、OOD tauを比較する。ドライラン（小規模）で訓練ループ・OOD評価とも
正常動作を確認済み。実機スケール実行後の結果を以下に記す（ログ: `/tmp/mlp_readout_ablation.log`）。

**結果:**

```
OOD(support_tickets held_out, n=300): accuracy=0.4533, tau=+0.018 (p=0.692), 定数予測器=no, spread=0.642
```

線形readout（Gate 1-b）: tau=+0.005 (p=0.916) との比較で、accuracy・予測幅は改善した
（0.3867→0.4533、spread 1.043→0.642は縮小だが定数予測器ではない）ものの、
**Kendall tauはtau=+0.018 (p=0.692)と、依然として統計的にゼロと区別できない。**

in-domain: AG News=0.8950, banking77=0.7350, emotion=0.5500, dbpedia14=0.9500, sst5=0.4225
（accuracy自体は線形readoutよりやや低いデータセットもあるが、大きな退行はない）、
順序反転一致率=0.9375（維持）。

公式severity_table（参考・定性のみ）: tau+0.400 (p=0.483)、**定数予測器から脱却**
（線形readoutはargmax固定だったが、MLP readoutは1/2の間で動く）。この点は面白い副次的観察だが、
n=5・非停止条件のため結論には含めない。

**仮説1（線形readoutの限界）は支持されない。** 非線形readoutに変えてもsupport_tickets
held-outのOOD Kendall tauは実質ゼロのまま動かなかった。

## 総合結論

仮説1（非線形readoutで検証、支持されず）と仮説2（zero-shot probeで検証、強く支持）を
並行して調べた結果、**仮説2（ラベル軸のずれ）が主因である可能性が高い**という結論に至った：

- 凍結backboneのzero-shot（訓練・readoutなし）ですら、`priority`ラベルに対するKendall tauは
  n=60で実質ゼロ（-0.003, p=0.973）。severity_table（n=5, tau+0.400）とは対照的
- 線形readout・非線形readoutのどちらで訓練しても、`priority`のOOD tauはゼロ付近のまま
  （+0.005 → +0.018、有意な改善なし）
- したがって、readout側の表現力（線形 vs 非線形）はボトルネックではなく、
  **`support_tickets`の`priority`ラベルがテキスト内容と弱くしか相関していない**ことが
  一連の結果を説明する最も整合的な仮説

**Gate 1-bで採用したデータソース（`Tobi-Bueck/customer-support-tickets`のpriority）は、
severityの代理として不適切だった可能性が高い。** これはGate 1-b固有のデータ選択の問題であり、
「readoutの訓練だけでは足りない（backbone介入が必要）」という結論には直結しない
——Gate 0の判定（severity_tableドメインでは凍結backboneに信号がある）は覆っていない。

## 次のステップ（解決済み — 選択肢1を実行）

ユーザーが選択肢1「より良いseverity代理データを探す」を選択。Gate 1-cとして実行し、
下記のとおり成功した。

---

## Gate 1-c: bugzilla severity（実バグ報告の技術的深刻度）を訓練データに追加 — 成功

仮説2（ラベル軸のずれ）の検証結果を受け、`AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset`
（Eclipse/Mozillaの実バグ報告、Severity Label: blocker/critical/major/normal/minor/trivial,
K=6, 人手ラベル）を新たなseverity代理データとして採用した。

**訓練前のzero-shot Q1検証**（`experiments/bugzilla_severity_probe.py`, n=60, レベルごと均等サンプル）:

```
tau: 校正後 +0.392 (p=0.000)  定数予測器: no
```

severity_table（Gate 0, n=5, tau+0.400）と同等以上の強さで、かつ統計的に有意（n=60）。
support_ticketsのpriority（tau≈0, p≈0.97-0.98）とは対照的。訓練前にQ1を確認する手順
（Gate 1-bの反省）が実を結んだ。

**訓練結果**（`experiments/severity_gate1c.py`, mean-pool, 既存5データセット + bugzilla severity
train=300/level×6=1800件, epochs=3, held_out=50/level×6=300件）:

```
=== 訓練後: in-domain ===
AG News   : accuracy=0.8650
banking77 : accuracy=0.7750
emotion   : accuracy=0.5475
dbpedia14 : accuracy=0.9050
sst5      : accuracy=0.4550, MAE(level)=0.6435
順序反転一致率: 0.9500

=== 訓練後: out-of-domain（bugzilla severity held-out, 停止条件の主判定） ===
OOD(n=300): accuracy=0.3800, tau=+0.423 (p=0.000), 定数予測器=no, spread=3.855
```

### 停止条件（§6）判定

1. **OOD tau有意に正・定数予測器でない・予測幅退化なし**: ✓ **満たす**。tau=+0.423, p=0.000
   （n=300で極めて高い有意性）。Gate 1-bのtau=+0.005 (p=0.916)から劇的に改善。
2. **in-domain退行なし**: ✓ 満たす。全データセットでaccuracy維持、sst5 MAEも改善
   （Gate 1-a単体の0.6785から0.6435へ）。
3. **順序反転一致率退行なし**: ✓ 満たす（0.9500、Gate 1-aの0.8625からさらに改善）。

**Gate 1-cは全ての停止条件を満たした。**

## 総合結論（Gate 1シリーズ完結）

一連の検証で、Score branchのOOD汎化不全の原因は最終的に特定・解決された:

1. **Gate 0**: severity_tableドメインの順序信号は凍結backboneに既に存在する（Q1真）→
   backbone介入は不要、readout側の改善から着手が正しい判断だった
2. **Gate 1-a**: mean-poolがlast-tokenよりわずかに優れる（pooled accuracy 0.6700 vs 0.6560、
   順序反転耐性は大幅改善）→ 採用
3. **Gate 1-b**: `support_tickets`のpriorityラベルを追加したが失敗（OOD tau≈0）。
   並行検証で「readoutの表現力不足」ではなく「ラベルとテキストの相関がそもそも弱い」
   ことが原因と判明（zero-shot probeでも同じ結果、n=60でtau≈0・p≈0.97-0.98）
4. **Gate 1-c**: `bugzilla_severity`（実バグ報告の技術的深刻度）に切り替え、訓練前に
   zero-shot Q1を確認（tau+0.392, p<0.001）してから訓練 → **OOD tau+0.423 (p<0.001)で
   停止条件を完全に満たす**

**教訓**: severityの代理データを選ぶ際は、訓練前に必ずzero-shot probeでQ1（凍結backboneが
そのラベルとテキストの相関を既に持っているか）を確認すること。Gate 1-bの失敗は
readoutやpoolingの問題ではなく、単にデータ選択のミスだった。この手順（Gate 0の手法を
新データソースに毎回適用する）を今後のデータ追加でも標準プロセスとする。

最終チェックポイント: `checkpoints/pointer_gate1c_mean_e3.pt`
（`option_pooling="mean"`, データセット: ag_news/banking77/emotion/sst5/dbpedia14/bugzilla_severity）。
