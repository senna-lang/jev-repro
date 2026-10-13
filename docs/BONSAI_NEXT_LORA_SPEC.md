# Bonsai 4B 次の LoRA 実装仕様

地図: `.scratch/bonsai-next-lora/map.md`。各決定の根拠は括弧内のチケット（`.scratch/bonsai-next-lora/issues/NN-*.md`）にある。既存の Bonsai 仕様は `internal-docs/BONSAI_JEV_SPEC.md`（以下「既存仕様」）で、ここに書いていないことは既存仕様のまま。

## 0. ゴールと範囲

- **作るもの**: 凍結した Ternary-Bonsai-4B（既存仕様 §1 の commit）に付ける fp32 LoRA sidecar を 1 本。汎用性能を改善し、エージェント判断（speculation gate など）も改善する（01、06）。
- **守る契約**: 既存仕様 §0 と同じ。branch ごとのプロンプト、共有 prefix の packing、tree mask、native LM head での回答文字の読み出し、MLX で融合しない sidecar（04、10 の決定でヘッド方式には切り替えない）。
- **範囲外**: base の変更、full fine-tuning、判定ヘッド方式（比較は 10 で別に行う）、公開リポジトリでの配布手順。

## 1. 守るルール

- **ログ由来データを学習に使わない**（04）。セッションログの state、それを LLM に言い換えたもの、生成時にログを見本として見せたものは学習に入れない。ログは手元の評価（calib・test）にだけ使い、公開するのは数値と手順だけ。用語は `CONTEXT.md` の「ログ由来データ」。
- **蒸留を禁じる API の出力を学習に使わない**（マップ）。Claude・GPT・grok などが書いた文章やラベルは入れない。
- **Gold V1 を使わない**。
- **評価 6 セットを学習に使わない**（05）。AG News test、MASSIVE en-US scenario test、MNLI validation_matched、BoolQ validation、SST-5 test、`LocalLLaMA/typed-decisions` test。各データセットの train split も含めて丸ごと除く。
- **選定は学習ドメインの val だけ**（05）。汎用側は §3.4 の val、エージェント判断側は `calib_human`。test・公開ベンチ・typed-decisions は選定に使わない。

## 2. 正解ラベルの作り方（07）

| 対象 | 正解 |
|---|---|
| 人手ラベルのある公開データ（RAGTruth・AgentHallu） | 人手の二値ラベルに label smoothing をかけた分布。one-hot では学習しない |
| 規則・人の票で正解が付いた公開データ（tasksource 2 種、Open-Jev） | 出典の正解（分布があれば分布） |
| それ以外（E2 の state、auto thinking・router、選択肢数を変えた設問） | DeepSeek Flash の順番平均を温度 T でなだめた分布 |

- **順番平均**: noul は No/Yes を入れ替えた 2 順、choice は 4 順、score は低→高と高→低の 2 順。各順の確率を元の順に戻して平均してから温度をかける。
- **T と smoothing の値（kind 別）**: 学習に使わない dev で、人手ラベル（または人の票の分布）に対する NLL が最小になる値を kind ごとに選び、学習前に固定する。noul の T と smoothing は §3.3 の speculation dev（約 300 件）で決める。choice と score の T は、tasksource のうち人の票の分布を持つ行から元文書単位で取り分けた dev（kind ごとに約 300 件、学習に使わない）で、Flash に同じ設問を付けさせて決める。noul で決めた T を choice・score に流用しない（2026-10-13 外部レビュー）。
- **理由**: E2 では Flash の Noul の 89% が確率 0.99 以上で、LoRA がこの尖りを引き継いだ（Gold V1 の Noul NLL 0.885、Jev 0.254）。
- **speculation の設問文の定義**: 教師に付けさせる場合は `data/speculation_devset.py` の `LABEL_DEFINITION_V2` を使う。ただし評価・選定の正解は人手（§5.2）。

## 3. 学習データ（11、12、08、09）

### 3.1 全体

train 約 2 万 branch（汎用 75%、エージェント判断 25%）、val 1,000 branch。件数はすべて §3.5 の除去後の値。

### 3.2 汎用側（約 1.5 万 branch）

| 出典 | 件数 | 備考 |
|---|---:|---|
| E2 の train の state（`data/e2/`） | 4,500 | §2 の Flash 方式で付け直す。choice は §3.4 の K 分布、score は 3〜7 段に作り直す |
| `tasksource/tasksource-jev-typed-decisions` | 6,000 | 行ごとの `license`／`license_use` で絞る。ChaosNLI 由来の行は除く |
| `ZefanCai/Open-Jev` | 2,500 | CC0。`metadata_json` の教師ラベルは入力に使わない |
| `tasksource/procedural-typed-decisions` | 2,000 | Apache-2.0 |

- **ライセンス**: Apache・MIT・BSD・CC0・CC BY だけ。NC・ND・SA・不明は除き、除いた件数を記録する（12）。
- **kind の比率**: choice 35%、score 35%、noul 30%。
- 除去で足りなくなった出典の分は tasksource で埋める。

### 3.3 エージェント判断側（約 5 千 branch）

| 設問 | 件数 | 出典 | 正解 |
|---|---:|---|---|
| speculation（noul） | 1,700 | RAGTruth（**Mistral-7B-Instruct が書いた回答だけ**）1,200、AgentHallu 500 | 人手ラベル＋smoothing |
| auto thinking（choice 4 段階） | 1,700 | 公開のコーディング依頼文（SWE-bench の issue など） | Flash 順番平均＋T |
| router（choice 3 段階） | 1,600 | 同上。auto thinking と依頼文を重ねない | Flash 順番平均＋T |

- RAGTruth・AgentHallu から元資料単位で約 300 件を T と smoothing を決める dev として取り分け、学習には使わない。
- **RAGTruth の生成モデル**: GPT-3.5・GPT-4（OpenAI の規約）と Llama-2-7B/13B/70B-Chat（Llama 2 Community License §1(b)(v) が出力を他の LLM の改善に使うことを禁止）の回答は除く。残るのは Mistral-7B-Instruct だけで、RAGTruth 内で幻覚率が最も高いモデルなので、正負比とタスク比を記録する。件数は §9 で決め直す。AgentHallu も行ごとの `model_id` で同じ規則を当てる。
- **設問文**（08）: 固定文面 50%、言い換え 50%。言い換えは設問ごとに 5〜10 通りを Flash で作り、選択肢の意味は変えない。評価用に学習に出さない言い換えを別に 2 通り作る。固定文面の出典: speculation・HN は jeb-scratch／life のコード、router は `omp-bonsai-system-one`（Apache-2.0）、auto thinking は `@oh-my-pi/pi-coding-agent`（MIT、作者 Can Boluk。配布時に出典を明記）。

### 3.4 選択肢数と val

- **choice の K**（09）: K=2〜3 を 20%、4〜5 を 50%、6〜8 を 15%、9〜12 を 10%、13〜20 を 5%。回答文字は A〜T（1 token）。選択肢を増減した設問は新しい選択肢一式を教師に見せて付け直す。旧分布の割り直しや、足した選択肢への 0 埋めはしない。
- **val**: 1,000 branch。train と同じ出典・同じ比率から、元文書（state の出どころ）を train と分けて取る。

### 3.5 評価データとの重なりの除去（05）

1. **出典で除く**: 元データが評価 6 セットと同じもの（ChaosNLI、bekko の validation、kotoba-lang の変換データ、各セットの翻訳・ミラーなど）。
2. **文字列で除く**: 学習データの state を正規化（小文字化、空白の統一）し、評価 6 セットの入力と完全一致するもの、または 8 語の連続一致があるものを除く。
3. 出典ごとに除いた件数を記録する。

## 4. 学習（13、09、06）

- **固定（既存仕様 §3 のまま）**: FP16 base を凍結、LoRA と optimizer 状態は fp32、r=16・α=32・dropout 0.05・all-linear、非有限値で即停止。
- **レシピ**: lr 1e-4、warmup 3% のあと cosine で 0 付近まで、最大 2 epoch、1/4 epoch ごとに val。AdamW は betas (0.9, 0.999)、eps 1e-8、weight_decay 0。1 step は 16 state（既存の `ROWS_PER_STEP`、約 64 branch）なので、train 2 万 branch・2 epoch で全体は約 625 step。重みの EMA は減衰率 **0.99**（平均の幅 約 100 step、全体の約 16%）で取り、生の重みと EMA の両方を選定候補にする。外部レビューで、当初の 0.999 は平均の幅（約 1,000 step）が学習全体より長く、EMA が初期値に近いまま残ることが分かった。データを作ったあと step 数を数え直し、平均の幅が全体の 10〜20% になるように減衰率を合わせる。
- **損失**: 全 branch で `KL(q‖p) + 0.1·Brier`。Score の branch だけ `+ λ·soft-RPS`。soft-RPS は意味上の段階順（低→高）で `Σ_{k<K}(F_p(k)−F_q(k))²/(K−1)` を計算する。choice・score の一部の設問には 2 順列の一致 KL を足す（§9 の 5）。
- **選択肢の並べ方**: choice は毎回ランダムにシャッフルする。**score は低→高と高→低の 2 向きだけ**（E3 の決定。段階の順序に意味があるので完全なシャッフルはしない）。noul は No/Yes の 2 順。soft-RPS は学習で見せた向きから段階順に戻してから計算する。既存の `permuted_row` は score も完全シャッフルしているので直す。
- **学習後の温度スケーリング（kind 別）**: 候補を固定したあと、val で kind ごと（noul・choice・score）に温度を 1 つずつ fit する（NLL 最小）。選定と評価の指標は温度をかけた後の値で測る。serve 側は回答文字の logit を kind の温度で割る（§7）。公開されている再現実装は、どれも学習後に T>1 を当てている。
- **run の段組み**（計 5 run）:
  1. seed 20260928 で λ ∈ {0, 0.5, 2} を 1 run ずつ。val で Score の soft-RPS が改善し、全体の NLL が悪化しない λ を選ぶ。該当がなければ λ=0。
  2. 選んだ λ で seed 20261003、20261004 を追加。3 run それぞれで val NLL 最小の checkpoint（生の重み・EMA から）を選び、pooled val NLL が中央値の run を候補にする。

## 5. 評価と合格基準（05、07）

比べる相手は既存候補 `checkpoints/bonsai/candidate`。凍結 base と Qwen E2 は報告だけ。推論は正準順 1 回（公開 serve と同じ経路）、MLX、Mac で 1 プロセスずつ（メモリ対策。キャッシュ上限 512MB、リクエストごとに解放）。順番平均は報告だけ。

### 5.1 汎用側

- **改善**: typed-decisions test（400 case、2,000 問）で、正解率の差（新 LoRA − 既存候補）の 95% 区間の下限 > 0、かつ KL の差の 95% 区間の上限が §9 で決める許容幅以下。区間はどちらも case 単位の paired bootstrap 10,000 回。入力は公開 serve の `to_request` と同じ変換。
  - 既存候補の基準点: 正解率 0.598、KL 0.485（`results/artifacts/typed_decisions/bonsai4b/summary.json`）。
- **非劣性**: 公開ベンチ 5 種（各 500 行、`mac/bench_public.py`）のそれぞれで、正解率の差の 95% 区間の下限 ≥ −0.03。

### 5.2 エージェント判断側

- **データ**: test は `data/speculation/test_human.jsonl`（500 行、人手ラベル）、判定に 1 回だけ使う。設問と checkpoint の選定は `data/speculation/calib_human.jsonl`（400 行、人手ラベル）。Flash のラベル（κ 0.27〜0.46）は評価・選定に使わない。
- **条件**: calib で選んだ設問 1 つについて、(1) AUROC が既存候補を上回る（差の 95% 区間の下限 > 0）、(2) AUROC が「回答の文字数だけ」の基準を上回る（同じく下限 > 0）。区間は同じ行どうしの bootstrap。
  - 参考（calib、人手ラベル）: 文字数だけ 0.703、既存候補の `noul_direct` 0.687。
- auto thinking・router は正解がないので、出力分布を報告するだけ。
- 不合格のときだけ、手元専用のエージェント判断 LoRA を新しいチケットで検討する（04）。

### 5.3 報告だけの指標

- Score（typed-decisions の score 設問）: soft-RPS、期待スコアの誤差、2 段以上外した割合。
- 選択肢数別: 公開ベンチの K=2（BoolQ）、3（MNLI）、4（AG）、5（SST-5）、18（MASSIVE）の NLL・top-1。
- kind 別の ECE と NLL（特に Noul）。
- 既存仕様 §5.3 のサイズ・メモリ（artifact・常駐 1.50 GiB 以下、peak 2.25 GiB 以下）も測る。r16 のままなので変わらない見込み。

## 6. 予算（14）

上限は Runpod と DeepSeek の合計 $50。実行中に $40 を超えたら止めてユーザーに確認する。見積もりは約 $36〜45（学習 5 run 約 $25、起動とやり直し約 $5、DeepSeek 約 $6〜11）。データを作った段階で state 数を数えて見積もりを更新する。学習前に Runpod の残高をチャージする。

## 7. 変更するコード

| 変更先 | 内容 |
|---|---|
| `data/` に新規モジュール | 公開データの取得、ライセンスでの絞り込み、§3.5 の除去、出典ごとの件数の記録、K 分布に合わせた選択肢の作り直し、train／val／dev への分割 |
| `e2/generate_teacher_data.py` または新規 | §2 の Flash 順番平均（noul 2・choice 4・score 2 順）と温度、言い換え文の生成 |
| `data/speculation_devset.py` | `LABEL_DEFINITION_V2` は追加済み（既定は v1 のまま） |
| `e2/train_lora.py` | §4 のレシピ（cosine、最大 2 epoch、EMA 0.99）、score を 2 向きだけにする `permuted_row` の修正、Score の soft-RPS 項と λ、新データの読み込み。既存 Qwen・Bonsai 経路の挙動は引数で残す |
| `e2/evaluate_lora.py`、`mac/bench.py` | 新しい val での選定指標（Score の soft-RPS を含む）、kind 別温度の fit と適用 |
| `mac/jev_mlx.py` と公開 serve | kind 別の温度を adapter と一緒に保存し、回答文字の logit を割ってから softmax する |
| 評価スクリプト | typed-decisions（`/tmp` の試作を `mac/` に正式化）、speculation の `test_human`／`calib_human` での AUROC と bootstrap、§5.3 の指標 |

## 8. 実装の順番

1. §3.5 の除去を含むデータ作成と件数の記録（学習前に件数と除去数をユーザーに見せる）。
2. §2 の dev（speculation 約 300、tasksource の choice・score 各約 300）で kind 別の T と smoothing を決める。
3. Flash での付け直しと言い換え文の生成（ユーザーは任意で言い換え文と Score の段階文を目で確認する）。
4. 学習コードの改修と短い smoke run（既存仕様 §6.1 と同じ確認）。
5. 段階 1 の 3 run → λ の決定 → 段階 2 の 2 run → 候補の固定。
6. §5 の評価を 1 回だけ実施して報告する。

## 9. 外部レビュー（2026-10-13）を受けた決定

反映済み: RAGTruth を Mistral の回答だけにする（§3.3）、EMA の減衰率と step 数（§4）、score を 2 向きだけにする（§4）、学習後の kind 別温度（§4、§7）、T を kind 別に決める（§2）、KL を区間で判定する形（§5.1）。

残り（学習前に決める）:
1. ~~§5.1 と §5.2 の片方だけ合格したときに LoRA を採用するか~~ → **§5.1 に合格すれば採用する。§5.2 は報告し、不合格なら手元専用のエージェント判断 LoRA を別チケットで検討する（04）。**（2026-10-13 決定）
2. ~~§5.1 の KL の許容幅 ε~~ → **ε = +0.05。typed-decisions で KL の差（新 LoRA − 既存候補）の 95% 区間の上限 ≤ +0.05。**（2026-10-13 決定）
3. ~~公開ベンチの非劣性~~ → **SST-5 だけ test 全量（2,210 行）で測り、マージンは 5 種とも −0.03 のまま。** 他の 4 種は 500 行のまま。base と既存候補の不一致率から見た 95% 区間の片幅は AG 0.018、MASSIVE 0.022、MNLI 0.023、BoolQ 0.026、SST-5 0.033（500 行）で、全量なら SST-5 は約 0.016。（2026-10-13 決定）
4. ~~RAGTruth の件数~~ → **Mistral-7B-Instruct の回答から 1,200 件。正例・負例を半々にし、長さの分布も正負でそろえて抜き出す。届かなければその件数で使い、不足分は AgentHallu（04 の規則を満たす行）で埋める。** GPT（OpenAI の規約）と Llama-2（ライセンス §1(b)(v)）の回答を除くと Mistral だけが残るため。（2026-10-13 決定）
5. ~~追加の手当て~~ → **3 つとも入れる。全 5 run に共通で、λ の比較とは掛け合わせない。**（2026-10-13 決定）
   - **Brier 項**: 損失に `0.1·Σ_i (p_i − q_i)²` を足す（Open-Jev・Clef の実績）。
   - **2 順列の一致 KL**: choice と score の一部の設問を、別の並び順で同じ step に 2 回入れ、意味の順に戻した 2 つの分布の対称 KL を足す。1 step の計算が増えるので、入れる割合はデータを作ったあと予算（§6）に収まる範囲で決める。
   - **speculation の対照ペア**: RAGTruth の幻覚 span を含む回答（正例）と、その span を削った回答（負例）を組にする。正解は人手の span から規則で決まり、Flash は使わない。RAGTruth の種別のうち「Baseless Information」だけを使い、「Conflict」（矛盾）は `LABEL_DEFINITION_V2` の範囲外なので除く。仕上げに、学習集合で「回答の文字数だけ」の AUROC を計算し、0.5 付近になるよう長さ帯ごとに正負比をそろえる。
