# ローカルJevの振る舞い上のゴール

## 目的

このプロジェクトの目標は、本家Jevの内部実装を複製することではない。

同じ入力契約に対して、妥当な確率分布を低レイテンシで返す**ローカルJev-style実装**を作る。

内部のbackbone、重み、tokenizer、学習法、attention実装が本家と異なることは許容する。評価対象は観測可能な振る舞いである。

**言語の前提**: state・instructions・optionsはすべて英語を前提とする。本家Jevも英語を主な学習言語とし、CJKを含む他言語は「扱えるが同等ではない」としている（docs.typesafe.ai/models「Language support」）。日本語は参考評価に留め、品質を主張しない。

## 完了条件

次の三条件を満たすこと。

1. **入力契約**: 自由文stateと複数のtyped branchを1 requestとして受け、1回の`forward()`で結果を返す。
2. **確率品質**: 人手Goldの投票分布に整合する確率分布を返す。
3. **レイテンシ**: 複数branchを個別にbackboneへ通す方式より、packed inferenceが有意に低コストである。

## 1. 入力契約

対応対象:

- 1つの自由文state。
- 同じstateに紐づく複数branch。
- Choice、Score、Noulの混在。
- Choice / Scoreの可変個数の選択肢。
- Noulの真偽出力。

出力契約:

- Choice: 選択肢と同じ順序・長さの確率ベクトル。和は1。
- Score: 選択肢確率と、その確率から得る期待スコア。
- Noul: `p(true)`。
- branch IDはcaller側で入力branchへ対応付けられる。

## 2. 確率品質

正解データは、凍結済みのprivate Gold V1の英語版を主評価に使う。

```text
evaluation/private/gold_v1_adjudicated_en_mt.jsonl   # 主評価（英語）
evaluation/private/gold_v1_adjudicated.jsonl         # 原本（日本語）・参考評価
```

- 60 cases / 90 branches。
- 3人の人手投票から作ったtarget分布。投票は日本語原本に対して行われた。
- 英語版は原本のstate・instructions・optionsをLLMで機械翻訳したもの。case/branch ID、option数・順序、target分布は原本と同一。翻訳版を主評価として採用することは2026-09-24にユーザー判断で決定した。
- 翻訳でラベルの意味がずれたbranchを見つけた場合は、翻訳を直してV1-en revisionとして記録する。target分布は変更しない。
- モデルbaselineはGold投票から除外する。
- Gold V1を変更しない。設問・選択肢・labelを変更する場合はV2を作る。

### Choice / Score

測定する:

- Brier score
- NLL
- top-1 accuracy
- Score expected-score MAE
- Score normalized EMD

### Noul

測定する:

- Brier score
- NLL
- threshold accuracy
- 人手yes率（0/3、1/3、2/3、3/3）ごとの平均予測確率

### 人手不一致の扱い

metricsは全体平均だけでなく、次のsliceに分ける。

- 3/3一致branch: 明確なケースへ適切に高い確率を置けるか。
- 2/1分割branch: 不当に確率1.0へ飽和せず、人手の割れを表現できるか。
- flag付きbranch: 人手が曖昧としたbranchで過信しないか。
- revision branch: 修正済みの入力契約に従うか。

branchごとの予測をprivate artifactへ残す。少なくとも次を記録する。

```json
{
  "case_id": "…",
  "branch_id": "…",
  "kind": "choice",
  "gold_target": [0.6667, 0.3333, 0.0],
  "prediction": [0.61, 0.35, 0.04],
  "top1_match": true,
  "brier": 0.004
}
```

## 3. 振る舞いの頑健性

以下を測定する。

1. **single vs packed**: 同一branchを単独で実行した場合と複数branchにpackした場合の差。
2. **branch reorder**: branch順序を変えても、各branchの出力が変わらないこと。
3. **branch add/remove**: 無関係branchの追加・削除が既存branchを変えないこと。
4. **option reversal**: 選択肢順序を反転後、option identityに戻して比較する。位置だけで答えていないことを確認する。
5. **混在型**: Choice / Score / Noulを同一requestへ混在させ、型・shape・確率の不変条件を確認する。
6. **可変option数**: 2、3、4以上の選択肢数で正しいshapeと確率分布を返すこと。

single vs packedとbranch reorderは、数値ノイズの範囲に収まることを受け入れ条件とする。

## 4. レイテンシ

同一backboneの`naive re-encode`を基準に、packed inferenceを比較する。

測定する:

- warm end-to-end p50 / p95 / p99 latency。
- 1、2、4、8、16 branchにおけるlatency。
- packed対naive re-encodeのlatency比。
- state / branch / optionのtoken長に対するlatency。
- peak memory。
- 同時request処理を実装する場合だけthroughput。

測定条件を固定する:

```text
同じcheckpoint
同じtokenizer
同じdevice / dtype
eval mode
warm-up後
同じ入力token数
複数回測定
```

目的は本家と同じkernelを使うことではない。利用者から見て、複数質問を個別評価より低コストで処理できることを示す。

## 5. 本家の公開例との比較

`experiments.official_compare`で、公式に掲載されたstate・question・出力例へ同じ入力を与える。

測定する:

- Choiceのtop-1一致。
- Score期待値差。
- Noul真偽一致。
- 公式確率が掲載される例ではJS divergenceまたはBrier。
- typed outputの形式と確率不変条件。

これは重みや内部実装の同一性を検証するものではない。公開された観測可能な振る舞いへの近さを測る。

## 6. 多ドメイン・日英比較

Gold V1英語版は英語サポート判断における主評価であり、別ドメインへの汎化を直接証明しない。

この評価はGold V1の代替ではない。Gold V1を通過したcheckpointが、特定のドメイン・設問語彙へ過適合せず、typed multi-question inferenceとして一般化するかを測る。

### 多ドメイン

既存または明確に分離したholdoutを使い、Choice・Score・Noulをそれぞれ評価する。

- AG News: topic Choice。
- banking77: intent Choice。
- DBPedia 14: entity / topic Choice。
- SST-5: 5段階のordinal Score。
- emotion: 感情のChoiceまたはScore。
- Noul holdout: escalation、urgency、repeat contact、payment problemの明示・非明示を含む未学習case。

各domainでBrier、NLL、top-1 accuracyを個別に報告する。Scoreはexpected-score MAEとnormalized EMD、Noulはthreshold accuracyを加える。

### 日本語（参考）

日本語は主張対象ではない。英語版と日本語原本の同一case・branchで次を参考値として報告し、差を隠さない。

- Choice top-1一致。
- Score期待値差。
- Noul真偽一致。
- option identityへ戻した確率分布のJS divergence。

### 学習データ

- 公開データセットと、LLMで合成したデータを使ってよい。
- 合成データはGold V1（原本・英語版）のstate・instructions・options文面を入力に使わず、コピーもしない。
- **Gold V1を見て学習タスクを決めない。** 設問の種類・選択肢の集合・段階の中身をGold V1からなぞることも禁止する（文面の一致だけでなく概念のなぞりも対象）。合成データの設問は、Gold V1を参照しない手順（例: LLMに業務判断の種類を広く列挙させる）で決める。
- 言い換えや段階数の変更など、特定の文言に依存させない工夫は許可する。ただし言い換えの中身をGold V1の文言に似せない。
- 合成データの生成model、prompt、seed、件数をmanifestへ記録する。
- 学習前にlinear probeまたはzero-shot probeで、ラベルがテキストから読めることを確認する。

### 分離規則

- Gold V1（原本・英語版）、公式例、多ドメインholdout、reference model出力を学習データに含めない。
- ゼロショット評価と、そのdomainを学習へ含めた評価を混ぜない。
- 単独分類だけでなく、複数branchを同一stateへpackしたrequestでも評価する。
- **Gold V1は最終試験としてだけ使う。** データ・ハイパラ・checkpointの選択にはGold V1と無関係な開発用セットを使い、Gold V1は大きな区切りでだけ測る。
- 合成データv1（2026-09-24）は9タスクがGold V1の9設問をなぞっており、この規則に反する。v1で学習したcheckpointのGold V1数値は、汎化の証拠として扱わない。

## 7. 強力な自己回帰モデルとのreference agreement

人手Goldを置き換えず、正解ラベルが作りにくい複雑なworkflowに対する補助評価として、強力な自己回帰モデルのstructured decision出力との一致を測る。

同一のworkflow、state、typed questionsを複数のreference modelへ与え、各modelのChoice / Score / Noul確率を保存する。referenceが複数ある場合は、平均分布を集約referenceとして使う。

測定する:

- Choice / Score: top-1 agreement、JS divergence。
- Score: reference期待スコアとのMAE。
- Noul: reference `p(true)`との平均絶対誤差。
- reference model間の相互JS divergenceまたは相互差。reference自身の不一致を隠さない。

この評価は「強い自己回帰モデル群の集約判断への近さ」を測るだけであり、真実性・人手との一致・確率校正の証明ではない。Gold V1のBrier / NLL / 人手分布評価を置換しない。

reference predictionは各branchを正確に1行ずつ含むJSONLとしてprivate領域へ保存し、使用model、version、prompt / adapter、temperature、seed、timestampをmanifestへ記録する。

## 最小評価セット

ローカルJev-style実装と、英語・複数domainへ汎化する実装を主張する前に、次を実施する。

1. Gold V1英語版の全90 branchでの分布評価。
2. 3/3、2/1、flag付き、revision別のmetrics。
3. single-vs-packed、reorder、option reversalの頑健性評価。
4. 1〜16 branchのwarm p50 / p95 / p99とnaive re-encode比較。
5. 公式掲載例のChoice / Score / Noul比較。
6. 分離済み多domain holdoutのdomain別確率品質評価。
7. 日本語原本との参考比較（主張には使わない）。
8. 複雑なworkflowにおける複数の強力な自己回帰reference modelとのagreement評価。

## 主張できること

上の最小評価セットが基準を満たした場合、次を主張できる。

> 英語の固定したtyped multi-question inputに対し、人手投票に整合する確率分布を返し、複数質問を単独評価より低いコストで処理し、公開Jev例と同じ種類の出力挙動を示し、複数domainで同じ入力契約を保つローカルJev-style実装である。日本語など英語以外は、本家と同様に同等の品質を保証しない。
