import json
from pathlib import Path
import torch
from e2.evaluate_lora import evaluate
from e2.train_lora import load_lora_model

ADAPTER = Path('/workspace/e2-runs/lr-1e-4/epoch-3-quarter-3')
model, tokenizer = load_lora_model(ADAPTER)
model.cuda().eval()
metrics = evaluate(model, tokenizer, Path('data/e2/teacher_branches.jsonl'), Path('data/e2/states.jsonl'), 'eval', torch.device('cuda'))
Path('/workspace/e2-results').mkdir(exist_ok=True)
Path('/workspace/e2-results/selected_eval.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
