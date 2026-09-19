# 公式Jev掲載例との比較 — AG News 10,000件訓練済みPointerReadout（2026-09-19）

## 条件

- 仮想Jev: Qwen2.5-0.5B backbone（凍結）＋ `checkpoints/pointer_agnews_s10000_e1.pt`
- checkpoint: AG News 10,000件×1 epoch、readout初期化・エポック内シャッフルともseed=0固定
- checkpoint評価: accuracy 0.2260→0.8040、ECE 0.5664→0.1980（held-out 500件）
- 比較入力: `docs.typesafe.ai` のQuickstart / Choice / Scoreページ掲載例
- **Noulタスクは除外**: `NoulReadout`はAG Newsにyes/no訓練データがなく、別の未訓練線形層のため
- 公式側はドキュメント掲載値。live APIは使用していない

## 実行出力

```text
readout: 訓練済みreadout (checkpoints/pointer_agnews_s10000_e1.pt, train_size=10000, epochs=1)
NoulReadoutは未訓練（訓練データにyes/noなし）のため、Noulタスクはこの比較から除外

=== quickstart_ticket  (source: docs.typesafe.ai/introduction/quickstart) ===
state: Hi, I've been trying to connect my Stripe account for 3 days and it keep...
1 request / 2 questions -> 1 forward
  [department] choice
    official: choice=billing  billing=0.840 technical=0.159 sales=0.001  confidence=0.596
    ours    : choice=technical  billing=0.000 technical=1.000 sales=0.000
  [frustration] score
    official: score=1.035    confidence=0.842
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000

=== easy_ticket  (source: docs.typesafe.ai/primitives/choice) ===
state: My running shoes arrived in the wrong size. Can I swap them for a size 1...
1 request / 1 questions -> 1 forward
  [department] choice
    official: choice=returns  returns=1.000 shipping=0.000 billing=0.000  confidence=1.0
    ours    : choice=billing  returns=0.000 shipping=0.000 billing=1.000

=== ambiguous_ticket (5 questions in 1 call)  (source: docs.typesafe.ai/primitives/choice) ===
state: Shoes arrived two weeks late and in the wrong size. Also I see two charg...
1 request / 5 questions -> 1 forward
  [department] choice
    official: choice=returns  returns=0.600 shipping=0.020 billing=0.380  confidence=0.39
    ours    : choice=billing  returns=0.000 shipping=0.000 billing=1.000
  [return_reason] choice
    official: choice=wrong_size  wrong_size=1.000 wrong_item=0.000 damaged=0.000 changed_mind=0.000 other=0.000  confidence=1.0
    ours    : choice=damaged  wrong_size=0.000 wrong_item=0.000 damaged=1.000 changed_mind=0.000 other=0.000
  [shipping_issue] choice
    official: choice=delayed  not_delivered=0.000 delayed=0.630 wrong_address=0.000 damaged_in_transit=0.000 other=0.370  confidence=0.53
    ours    : choice=wrong_address  not_delivered=0.000 delayed=0.000 wrong_address=1.000 damaged_in_transit=0.000 other=0.000
  [requested_resolution] choice
    official: choice=exchange  exchange=0.370 refund=0.290 replacement=0.240 information=0.100  confidence=0.16
    ours    : choice=replacement  exchange=0.000 refund=0.000 replacement=1.000 information=0.000
  [tone] choice
    official: choice=frustrated  calm=0.000 frustrated=0.920 angry=0.080  confidence=0.88
    ours    : choice=frustrated  calm=0.000 frustrated=1.000 angry=0.000

=== return_topic (structured criteria)  (source: docs.typesafe.ai/primitives/choice) ===
state: I sent the shoes back a week ago. When do I get my money?...
1 request / 1 questions -> 1 forward
  [return_topic] choice
    official: choice=return_status  return_policy=0.000 return_status=1.000  confidence=1.0
    ours    : choice=return_status  return_policy=0.000 return_status=1.000

=== severity_table (same Score question, 5 states)  (source: docs.typesafe.ai/primitives/score) ===
  state: The export button is misaligned by a few pixels on the setti...
    official: score=0.0  L0=1.000 L1=0.000 L2=0.000  confidence=1.0
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000
  state: The PDF export button does nothing when clicked. I can still...
    official: score=1.0  L0=0.000 L1=1.000 L2=0.000  confidence=1.0
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000
  state: Export to PDF fails with a spinner that never finishes. Some...
    official: score=1.12  L0=0.000 L1=0.880 L2=0.120  confidence=0.81
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000
  state: The export button crashes the settings page in Safari. It wo...
    official: score=1.3  L0=0.000 L1=0.700 L2=0.300  confidence=0.54
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000
  state: Nobody on our team can log in since this morning. We get a 5...
    official: score=2.0  L0=0.000 L1=0.000 L2=1.000  confidence=1.0
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000

=== spinner_ticket (3 Score questions in 1 call)  (source: docs.typesafe.ai/primitives/score) ===
state: Export to PDF fails with a spinner that never finishes. Some of our team...
1 request / 3 questions -> 1 forward
  [severity] score
    official: score=1.24  L0=0.000 L1=0.760 L2=0.240  confidence=0.63
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000
  [frustration] score
    official: score=1.45  L0=0.000 L1=0.550 L2=0.450  confidence=0.33
    ours    : score=2.000  L0=0.000 L1=0.000 L2=1.000
  [report_quality] score
    official: score=3.0  L0=0.000 L1=0.000 L2=0.000 L3=1.000  confidence=1.0
    ours    : score=3.000  L0=0.000 L1=0.000 L2=0.000 L3=1.000

=== 構造チェック ===
requests=10, question outputs=17, probability values=56
全確率が[0,1]内: OK
```

## 結果

- **構造は一致**: 10リクエスト・17 questionをtyped出力として処理。複数question（最大5問）も1回のforwardで回答し、56確率値はすべて[0,1]内。選択肢集合外の値は出ない。
- **Choiceのtop-1一致は8問中2問**: `tone=frustrated`、`return_topic=return_status`のみ一致。サポートチケットdomainへの意味的な転移は見られない。
- **Scoreの最高確率レベル一致は9問中2問**: 「全員ログイン不能」のseverity=L2とreport_quality=L3のみ。残りはすべて最高レベルへ飽和した。
- **確率はほぼ完全に0/1へ飽和**: 公式Jevが0.84/0.159、0.63/0.37、0.76/0.24など中間分布を返すのに対し、仮想Jevはほぼone-hot。AG News内ではaccuracy/ECEが改善しても、domain外ではcalibrationと意味的汎化が成立していない。
- **結論**: このcheckpointが示すのはJev風のtyped/listwise構造とAG Newsでのlinear-probe性能。公式Jevの汎用判断能力やcalibrationは再現できていない。

## 再現性についての注記

旧scaling実行の10,000件ベスト値（accuracy 0.8300 / ECE 0.1706）はreadout初期化のみseed=0、エポック内シャッフルは未固定だった。今回のcheckpointは初期化・シャッフルともseed=0固定した再現可能な再訓練で、結果はaccuracy 0.8040 / ECE 0.1980。したがって同一データ・同一epoch数でも旧実行と完全同一の重みではなく、旧ベスト値のcheckpoint復元ではない。

再現コマンド:

```bash
python3 -m experiments.official_compare checkpoints/pointer_agnews_s10000_e1.pt
```
