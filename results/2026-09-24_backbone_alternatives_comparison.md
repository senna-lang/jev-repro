# 2026-09-24: backbone代替案の比較（Qwen2.5-0.5B-Instruct / Qwen3-0.6B）

## 目的

ユーザーの指示で、現行backbone（Qwen2.5-0.5B base、instruction-tuningなし）に対する代替案として
Qwen2.5-0.5B-Instruct・Qwen3-0.6Bをzero-shot probeのみ（訓練なし）で比較した。日本語特化backbone
（Sarashina2.2・llm-jp等）はユーザーが明示的に却下（多言語性を保ちたいため）、比較対象から除外。

## 手法

`experiments/backbone_probe.py`（Gate 0で構築、prose-cueプロンプト＋teacher-forced複数トークン
option scoring）の`MODEL_NAME`をモンキーパッチし、各backboneで以下2つのzero-shot probeを実行。
readoutは一切使わず、凍結backbone＋LM headのみで判定（訓練は行っていない）。`model/backbone.py`
へのコード変更は行っていない（本番backboneは変更なし）。

- **感情二値判定**（sst5の最遠2レベル、易しい対照タスク）
- **severity_table**（Gate 0の公式5-state表、n=5、正解順序ドキュメント記載）
- **bugzilla severity**（`experiments/bugzilla_severity_probe.py`、実バグトリアージのSeverity Label、
  校正後Kendall tau、n=60）

## 結果

| backbone | 感情accuracy | severity_table tau (n=5) | bugzilla severity tau (n=60, 校正後) |
|---|---|---|---|
| **Qwen2.5-0.5B（現行・base）** | 0.900 | **+0.400** | **+0.392 (p<0.001)** |
| Qwen2.5-0.5B-Instruct | 0.960 | **-0.800**（符号反転） | +0.099 (p=0.294, 非有意) |
| Qwen3-0.6B | 0.960 | +0.000 | +0.142 (p=0.133, 非有意) |

- 両代替案とも単純な二値感情判定では現行より高精度（0.900→0.960）。
- しかし本プロジェクトが必要とする**順序尺度判定（severity等）では両方とも現行より明確に劣る**。
  Instructモデルはseverity_table taxで符号が反転（+0.400→-0.800）、bugzilla severityでは統計的
  有意性を失う（p<0.001→p=0.294）。Qwen3-0.6Bも同様にbugzilla severityで非有意（p=0.133）。
- Qwen3-0.6Bは`Qwen3Model`クラス（hidden_size=1024、28層）で`Qwen2Model`とアーキテクチャ上は
  別物だが、forwardシグネチャ（`input_ids`/`attention_mask`/`position_ids`）は互換。

## 決定

**現行のQwen2.5-0.5B baseを維持する。** instruction-tuningやより新しい世代のモデルは、単純な
二値分類では有利でも、本プロジェクトの核心タスクである順序尺度判定（Score branch）では一貫して
劣化する。コード変更なし。
