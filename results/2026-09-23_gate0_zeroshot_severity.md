# Gate 0: severity_tableドメインでのzero-shot天井測定

設計: `internal-docs/EXP_HEAD_VS_BACKBONE.md`。目的: Q1（severity_tableドメインの順序尺度情報が
凍結backboneの隠れ状態に既に存在するか）をこのドメイン固有に判定する。sst5では天井0.900が
既知だが、severity_tableでは未測定だった（項目5はargmax 4/5一致のみ、tau未測定）。

## 実装

`experiments/backbone_probe.py`を拡張:

- `probe_severity`: `model/packer.py::pack()`と同一の地の文プロンプト形式
  （`"{instructions} Answer with one word — {opt1}, {opt2}, {opt3}.\nAnswer:"`）に切り替え、
  選択肢テキストも`official_compare.py`の公式`SEVERITY_LEVELS`（説明節、K=3）をそのまま使用
  （旧実装は`cosmetic`/`degraded`/`blocking`という短い代理語だった — 本番入力と非同一だったため
  今回置き換えた）。
- 候補スコアリングを先頭トークンのみの比較から、**teacher-forcingによる平均トークンlog確率**
  （`_option_avg_log_probs`）に変更。report_qualityの選択肢が先頭数トークンを共有する
  （"Steps to reproduce or..." / "Steps to reproduce and..."）ため、先頭トークンのみでは
  区別できないことが判明し、全トークンをスコアする方式に統一した。
- Kendall tau・p値・定数予測器フラグ・予測幅（max−min）を追加（`official_compare.py::run_severity_table`
  と同じ指標定義を流用）。
- frustration・report_qualityは公式ドキュメントに独立した既知順序state集合が存在しない
  （`official_compare.py`には各1件のみ）ため、severityと同じ5 statesに別の質問を当てて
  分布のみ定性観察する`probe_ordinal_no_ground_truth`を追加。tauは計算しない
  （§3.1: 未検証のground truthを捏造しない）。

## 結果

```
=== Gate 0: severity（LM head zero-shot, K=3, 地の文プロンプト） ===
tau: 校正前 +0.400 (p=0.483) / 校正後 +0.400 (p=0.483)
定数予測器: 校正前 YES / 校正後 no
予測幅(max-min): 校正前 0.089 / 校正後 0.114
argmax: 校正前 [1,1,1,1,1] / 校正後 [0,2,2,2,2]  (公式=[0,1,1,1,2])
```

対照（sst5最遠ペア二値、LM head zero-shot）: accuracy 校正前0.900 / 校正後0.890（既知の高天井、再確認）。

frustration/report_qualityは同5statesで定性観察のみ（tau未計算）: いずれも3〜4値の分布が
0.2〜0.4のレンジに収まり、明確な単調傾向は目視では確認できない（既知順序が無いため定量判定不可）。

## 判定（P1検証）

`EXP_HEAD_VS_BACKBONE.md` P1の予測「Gate 0でseverity_tableのzero-shot順序は正方向に出る」は
**的中**。校正後tau +0.400、定数予測器でない、予測幅0.114（退化していない）。n=5のため
統計的有意性（p=0.483）は判断材料にしない（設計のP1注記どおり、方向のみ見る）。

→ **Q1真・Q2偽**: severity_tableドメインの順序信号は凍結backboneの隠れ状態に既に存在する。
不足しているのは抽出側（readout/訓練データ）。

→ **`EXP_HEAD_VS_BACKBONE.md` §6の決定木に従い、Gate 1（backbone凍結のまま、readout側の
安価な改善）に進む。Gate 2/3（部分解凍・LoRA）は現時点では正当化されない。**

## 次

Gate 1-a（`h_opts`のpooling変更: 末尾トークン→区間mean-pool）と、Gate 1-b（severity寄り訓練データの
追加）のどちらから着手するか、作業範囲をユーザーに確認してから実行する。
