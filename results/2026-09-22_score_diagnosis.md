# Score readoutの診断 — 3つの仮説の棄却と、選択肢バイアス支配の確定（2026-09-22）

## 背景

`2026-09-21_diverse_training_order_invariance.md`のAG News+banking77の2データセット訓練に、
`emotion`（態度Choice）・`sst5`（5段階順序Score）・`dbpedia14`（語彙多様化Choice）を加えた
5データセット混合を3エポック訓練した（`checkpoints/pointer_diverse5_e3.pt`、CPU 4828s）。

| データセット | chance水準 | 訓練前 | 訓練後 |
|---|---:|---:|---:|
| AG News (K=4) | 0.25 | 0.288 | 0.850 |
| banking77 (K=3-6) | ~0.22 | 0.205 | 0.703 |
| emotion (K=3-6) | ~0.22 | 0.228 | 0.415 |
| dbpedia14 (K=4-8) | ~0.17 | 0.150 | 0.913 |
| sst5 (5段階Score) | 0.20 | 0.138 | 0.270 |
| 順序反転 base/rev一致率 | — | 0.175 | 0.700 |

topic系（AG News, dbpedia14）は強く、情動系（emotion, sst5）が弱いという非対称が出た。
公式Jev比較ではChoice top-1一致が4/8→5/8に改善した一方、Score最高レベル一致は5/8→1/8に
「後退」したように見えた。本ドキュメントはこの後退の原因調査の記録である。

## 結論の先出し

1. **「Scoreの後退」は指標のアーティファクトだった**。2データセット版の5/8は、
   stateを一切見ない定数予測器がたまたま当たっていただけ。
2. 実際の病理は**選択肢固有バイアスの支配**。テキストに依存しない加法的オフセットが
   支配的で、テキスト依存の信号はその上の微小な揺らぎにすぎない。
3. 途中で立てた3つの仮説（MSE損失による平均回帰／選択肢テキストの形式／レベル数Kの
   一般化失敗）はいずれも**実験で棄却**された。
4. 推論時のバイアス除去（null state校正）でバイアス病理は消える（最遠ペア二択の
   accuracy 0.564→0.795）が、**公式severityは救われない**。そちらは信号自体が存在しない。

## 1. 「Score 5/8 → 1/8の後退」は偽りだった

`experiments/score_diagnose.py`で、公式Scoreドキュメントのseverity表（同じ質問を深刻度の
異なる5つのstateに当てる。正解の順序が既知）を両checkpointに当てた。

| 条件 | 2データセット版 | 5データセット版 |
|---|---|---|
| severity表5stateの予測位置 | 0.976, 0.978, 0.979, 0.979, 0.995（幅**0.018**） | 0.57, 0.17, 0.23, 0.32, 0.33（幅0.40） |
| argmax | **[1,1,1,1,1]** — 全state同じ | [1,0,0,0,0] |
| sst5 held-out 200件のKendall tau | **−0.039 (p=0.45)** | **+0.271 (p=2e−07)** |

2データセット版にはScoreの教師データが無く、**stateを無視して常にL1を返していた**。
公式の正解が[L0,L1,L1,L1,L2]でL1が3つあるため、そのまま3問「一致」していた。
一方5データセット版は訓練ドメインで統計的に有意な順序信号を獲得している。

**教訓**: argmax一致率は定数予測器に報酬を与える。順序尺度の評価には順位相関を併記しないと
判断を誤る。この誤読を一度実際にやってしまった。

## 2. 棄却された仮説

### 仮説A: ordinal MSE補助損失による平均回帰（棄却）

`train_readout`のScore補助損失`((E[level] − gold)/(K−1))^2`が、確率を散らして期待値だけ
合わせる戦略を誘発しているのではないか、という仮説。sst5単独1000件×1エポック、
`score_mse_weight`だけを変えたablation（`experiments/score_ablation.py`）:

| weight | sst5 tau | sst5 MAE | sst5幅 | severity tau | severity幅 | accuracy | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | +0.187 | 1.099 | 2.500 | +0.400 | 0.459 | 0.307 | 0.173 |
| 0.0 | +0.137 | 1.322 | 2.221 | **−0.600** | 0.142 | 0.163 | 0.399 |

MSEを外すと全指標が悪化する。仮説と逆。**MSE項はむしろ効いている**。

### 仮説B: 選択肢テキストの形式（ほぼ棄却）

sst5の選択肢は`very negative`のような短いレベル名、公式は`Cosmetic; no impact to
functionality`のような説明節。2x2（短いレベル名/説明節 × K=3/K=5）でseverity表を測ると、
K=3で短いレベル名 tau=0.000、説明節 tau=+0.200。n=5では差を判定できず、主要因ではない。

### 仮説C: レベル数Kの一般化失敗（症状ではあるが原因ではない）

訓練は常にK=5固定、公式はK=3（severity/frustration）とK=4（report_quality）。
同一テキスト（sst5）でKだけ変えると:

| K | tau | p |
|---:|---:|---:|
| 2 | +0.084 | 0.15 |
| 3 | −0.013 | 0.81 |
| 4 | +0.033 | 0.53 |
| **5**（訓練時の設定） | **+0.107** | **0.04** |

訓練したKでしか信号が無い、と読めた。そこで`data/sst5.py`を**K可変**（正解を含む長さK∈[2,5]
の連続部分区間をサンプルし、窓内の相対位置をラベルに付け替える）に改修して再ablation:

| arm | K=2 | K=3 | K=4 | K=5 | severity tau |
|---|---:|---:|---:|---:|---:|
| K=5固定 | +0.076 (ns) | −0.056 (ns) | +0.231 | +0.351 | −0.200 |
| K可変[2,5] | −0.027 (ns) | −0.088 (ns) | +0.173 | +0.203 | −0.200 |

**K可変化は効かなかった**。K=2/3は依然ゼロ、K=4/5はむしろ低下。

さらにこの測定には交絡があった。連続窓でKを狭めると選択肢間のレベル距離も同時に縮むため、
K=2の窓は必ず「隣接レベルの判別」になる。このモデルのMAEは約1.0レベルなので、隣接判別は
原理的に不可能。「K=2でtau≈0」はK一般化の失敗でも解像度不足でも説明できてしまう。

## 3. 確定した病理: 選択肢固有バイアスの支配

交絡を外すため、**同一テキスト・同一の意味距離でレイアウトだけを変える**測定を行った。
最も離れた2レベル（very negative / very positive）の例のみを使う:

| 条件 | 指標 | 結果 |
|---|---|---:|
| (a) K=5の全選択肢 → 予測scoreで順位付け | AUC | **0.708** |
| (b) 同じテキスト、`very negative` vs `very positive`の二択 | accuracy | **0.525** |

信号は存在する（AUC 0.708）のに、二択のargmaxにすると消える。K=5での予測score平均は
very negative=2.770 / very positive=2.951——4レベルの幅に対して差はわずか**0.18**。

AUCは定数バイアスに不変、argmaxは不変でない。この非対称が病理を名指しする。直接確認:

```
最遠ペア二択 n=39: モデルの選択 'very negative'=1 / 'very positive'=38
                  正解の内訳 'very negative'=18 / 'very positive'=21
sst5 多数派ベースライン: accuracy=0.307
訓練済みreadoutのsst5 accuracy: 0.270
```

**39件中38件が同じ選択肢**。ほぼ完全な定数予測器であり、正解率は「常に最頻レベルを答える」
戦略にすら負けている。

### なぜChoiceでは起きずScoreで起きたか

Choiceは例ごとに選択肢集合が変わる（ラベル名・K・distractor・順序をすべて振っている）ため、
選択肢固有の好みを持つ戦略は罰せられる。Scoreは**常に同じ5つの文字列**なので、定数バイアスが
正当な局所最適になる。そこから抜け出すには微細な感情強度を凍結特徴から取り出す必要があり、
それができていない。K可変化がScoreを救わなかったのは、Kだけ振っても**文字列が固定のまま**
だったからである。

## 4. S1: 推論時のバイアス除去（再訓練なし）

内容を持たないstate（`"N/A"`, `"..."`）で同じ選択肢集合をforwardし、その出力を
「テキストに依存しない選択肢固有の好み」として引く: `z'_i = z(state)_i − z(null)_i`。
softmaxは定数シフトに不変なので、log確率の差で厳密に同じ分布が得られる（forward内部に
触れずに実装できる）。既存checkpointのまま測定:

| 測定 | 校正前 | 校正後 |
|---|---:|---:|
| (a) 最遠ペア二択 accuracy | 0.564 | **0.795** |
| (a) 選択の偏り（正解=[18,21]） | [1, 38] | **[24, 15]** |
| (b) sst5 K=5 tau | +0.095 (p=0.12) | **+0.163 (p=0.007)** |
| (b) 予測位置の幅（gold幅4.0） | 1.517 | **2.049** |
| (b) sst5 accuracy（多数派=0.307） | 0.287 | 0.267 |
| (c) 公式severity tau | +0.200 | −0.200 |
| (c) 校正後argmax（公式=[0,1,1,1,2]） | — | [0,0,0,0,0] |

バイアス病理そのものは解消する（定数予測器が消え、順位相関が有意になり、圧縮も緩む）。
しかし**公式severityは救われない**。校正後も全stateを「cosmetic」に置く。

問題は2層に分かれていた:

1. 選択肢固有バイアス → S1で除去できる（実証済み）
2. テキスト依存信号そのものの弱さ → 校正では作り出せない。sst5でさえ校正後tau +0.163・
   正解率は多数派以下。バグ深刻度のドメインには信号がほぼ存在しない。

## 5. backboneは信号を持っているのか（LM head zero-shot）

「信号が無いのはreadoutの学習の失敗か、凍結backboneにそもそも情報が無いのか」を、
readoutを一切通さずLM headで測った（`experiments/backbone_probe.py`）。Phase 9で
AG News 0.675を出したのと同じ手法の順序尺度版。プロンプトは自然文:

```
Movie review: {text}
Is the sentiment of this review negative or positive? Answer with one word.
Answer:
```

| probe | 凍結backbone LM head zero-shot | 訓練済みpointer readout |
|---|---:|---:|
| sst5最遠ペア二値（感情極性） | **0.900** | 0.525（校正後0.795） |
| 公式severity argmax | **[0,1,1,1,1]** vs 公式[0,1,1,1,2] = 4/5一致 | [0,0,0,0,0] = 1/5 |

**backboneは両方知っていた**。severityもsentimentも凍結Qwen2.5-0.5Bの中に十分な信号がある。
S3（backbone差し替え・LoRA）はこの証拠により根拠を失う。取り出せていないのが問題。

## 6. どこで信号が失われるか（readout差し替え）

packed requestの同じ隠れ状態から、readoutだけを差し替えて比較した
（`experiments/readout_probe.py`）。`logit lens`は`h_dec`をbackboneのLM headに通して
選択肢の先頭トークン確率を読む方式で、pointer readoutを完全に迂回する。
さらにpacker側も、特殊トークン版と自然文マーカー版（`<STATE_END>`→`\n\n`、`<SEP>`→`\n`、
`<DECISION>`→`\nAnswer:`）の2条件で測った。

| packer | logit lens | 恒等初期化pointer(未訓練) | 訓練済みpointer |
|---|---:|---:|---:|
| 特殊トークン（現状） | 0.520 [86,14] | 0.500 [0,100] | 0.500 [0,100] |
| 自然文マーカー | 0.500 [100,0] | 0.510 [1,99] | 0.500 [100,0] |
| （参考）自然文プロンプト直接 | **0.900 [40,60]** | — | — |

同じ重みで、**自然文プロンプトなら0.900、packed request経由だとどのreadoutでも0.50前後**。
readoutの種類を問わず落ちるので、失われているのはreadoutより手前——リクエストの形式そのもの。
特殊トークン（`<DECISION>`は`resize_token_embeddings`で足した未学習の埋め込み）を自然文
マーカーに替えるだけでは回復しなかった。

現行`pack()`が作る列は、区切りテキストを一切挟まず連結する:

```
{state}<STATE_END>{instructions}{option_1}<SEP>{option_2}<SEP><DECISION>
```

自然文プロンプトとの差は、(a) 未学習の特殊トークン、(b) 選択肢が質問文と決定位置の間に
裸で並ぶ、(c) 区切り文字が無く単語が連結される、(d) 明示的な解答キューが無い、の4点。
(a)だけを潰しても効かなかったので、残りが効いている可能性が高い。

なお単一branchのリクエストではtree maskは通常のcausal maskと一致するため、
tree mask自体はこの劣化の原因ではない。

### 測定バグの記録

最初の測定で`very negative`/`very positive`を選択肢にしたため、**両方の先頭トークンが
` very`で衝突**し、logit lensが常に引き分け→index 0固定となり、accuracyがちょうど0.500・
選択の偏り[100,0]という「完全な定数予測器」に見えた。これはモデルの性質ではなく測定バグ。
`_logit_lens`に衝突検査のassertを入れ、選択肢を`negative`/`positive`に直して測り直した
（上表はすべて修正後の数値）。

## 追加した道具

- `experiments/score_diagnose.py` — 診断集。`python3 -m experiments.score_diagnose <ckpt> [診断名]`
  で個別実行できる（`format` / `monotonicity` / `in_domain` / `k` / `pairs` / `bias` /
  `calibration` / `extremes`）。
- `experiments/score_ablation.py` — sst5単独の安価なablationドライバ。アームを
  `(k_range, score_mse_weight)`で定義し、他の条件（データ・seed・初期化・epoch数）を固定する。
- `experiments/backbone_probe.py` — readoutを通さないLM head zero-shot。backboneが信号を
  持っているかの上限測定に使う。
- `experiments/readout_probe.py` — packed requestの同じ隠れ状態に対してreadoutとpackerを
  差し替え、信号が失われる場所を特定する。
- `data/sst5.py` — K可変化（`k_range`引数）。効果は出なかったが、公式のK=3/K=4と条件を
  揃えられるようになったため残す。

## 7. 形式の絞り込み: ラベル付けでは不十分、地の文だけが天井に届いた

`readout_probe`に3つ目の中間案を追加した: `Option N: text` を改行区切りで並べ、末尾に
`Answer:` を置く（「ラベル+解答キュー」）。

| packer | 最遠ペア logit lens | severity tau (logit lens) |
|---|---:|---:|
| ラベル+解答キュー | 0.600 | +0.000 |
| 地の文（backbone_probe再現） | **0.910** | **+0.600** |

構造化リストへのラベル付けだけでは天井に届かない。届くのは選択肢を1つの自然文にコンマ
区切りで埋め込み「次の1語で答えよ」と明示する形式だけだった。

## 8. `model/packer.py`を地の文形式へ移行

`pack()`を書き換え、`<STATE_END>`/`<SEP>`/`<DECISION>`の3特殊トークンを廃止した:

```
{state}\n\n{instructions} Answer with one word — {opt_1}, {opt_2}, ..., {opt_K}.\nAnswer:
```

Noulは `{instructions} Answer yes or no.\nAnswer:`。`model/forward.py`の
`resize_token_embeddings`呼び出しも削除した（readout重みはhidden_size依存で語彙数に
依存しないため、旧checkpointのpointer/noul重みはそのままロードできる）。
`model/packer.py`・`model/forward.py`・`model/readout.py`・`README.md`のsmoke check
すべて再実行して通過を確認した。移行後の`readout_probe`再実行で、地の文パッカーは
logit lens 0.910・severity tau+0.600（旧特殊トークン形式で訓練したpointer重みのまま
でもtau+0.800・4/5一致）を再現し、移行が正しいことを確認した。

## 9. 新形式での再訓練結果

`pointer_diverse5`を新形式で再訓練（3 epoch、同じ5データセット混合）。

| 指標 | 旧形式（特殊トークン） | 新形式（地の文） |
|---|---:|---:|
| sst5 accuracy（Score, in-domain） | 0.270 | **0.395** |
| sst5 MAE(level) | 0.845 | **0.679** |
| 公式severity_table Kendall tau（サポートチケット文, out-of-domain） | — | **+0.000**（定数予測器ではない） |

sst5（訓練データそのもののドメイン）は明確に改善した。一方、`official_compare.py`の
severity_table（サポートチケット文の重大度、訓練データに含まれないドメイン）は順位相関
ゼロのまま——フォーマットの天井は外れたが、severity_tableドメインへの汎化はまだ課題として
残る。`experiments/official_compare.py`にKendall tauと定数予測器フラグを追加し、
argmax一致率だけでの誤読（本診断の発端）を今後防ぐようにした。

## 次の判断材料

上限（自然文プロンプト0.900）と現状（packed経由0.52）の40ポイントの差はpackerの
書き換えで解消し、in-domainのsst5では実際に改善が確認できた。残る課題は
severity_tableドメイン（サポートチケット文の重大度）への汎化——sst5（映画レビューの
感情極性）とジャンルが離れすぎている可能性がある。次はseverity寄りの訓練データ
（バグ報告・サポートチケット風の重大度ラベル付きデータ）の追加、またはS2（Score
選択肢の語彙を振って訓練時の過学習を防ぐ）を評価する。
