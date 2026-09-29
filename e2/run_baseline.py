import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2.evaluate_lora import evaluate

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
metrics = evaluate(model, tokenizer, Path('data/e2/teacher_branches.jsonl'), Path('data/e2/states.jsonl'), 'eval', torch.device('cuda'))
Path('results').mkdir(exist_ok=True)
Path('results/e2_baseline_eval.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
