import time
from pathlib import Path
import random
import torch
from e2.packed_data import load_requests
from e2.train_lora import load_lora_model, packed_kl_loss, permuted_row

model, tokenizer = load_lora_model()
model.cuda().train()
rows, request = next((rows, request) for rows, request in load_requests(Path('data/e2/teacher_branches.jsonl'), Path('data/e2/states.jsonl')) if rows[0]['split'] == 'train')
started = time.perf_counter()
loss = packed_kl_loss(model, tokenizer, permuted_row(rows, request, random.Random(20260928)), torch.device('cuda'))
loss.backward()
print({'loss': loss.item(), 'forward_backward_seconds': time.perf_counter() - started, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
