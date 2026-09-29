import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2.chat_packer import chat_prompt_ids, pack_chat_request
from e2.eval_permutations import expanded_branches
from e2.packed_data import load_requests
from model.packer import PackedRequest
from model.tree_mask import build_position_ids, build_tree_attention_mask

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
LETTERS = 'ABCDEFGHIJKLMNOPQRST'

def labels(tokenizer, count):
    return torch.tensor([tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS[:count]], device='cuda')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--eval', action='store_true'); args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    checked = flips = material = 0
    for rows, request in load_requests(Path('data/e2/teacher_branches.jsonl'), Path('data/e2/states.jsonl')):
        if args.eval:
            if rows[0]['split'] != 'eval': continue
            items = expanded_branches(rows, request.branches)
            branches = [branch for _, branch, _ in items]
            item_rows = [row for row, _, _ in items]
        else:
            if rows[0]['split'] != 'train': continue
            branches = request.branches; item_rows = rows
        packed = pack_chat_request(PackedRequest(request.state, branches), tokenizer)
        with torch.inference_mode():
            packed_logits = model(
                input_ids=torch.tensor([packed.input_ids], device='cuda'),
                attention_mask=build_tree_attention_mask(packed).cuda()[None, None],
                position_ids=torch.tensor([build_position_ids(packed)], device='cuda'),
            ).logits[0]
        for row, branch, span in zip(item_rows, branches, packed.branch_spans):
            allowed = labels(tokenizer, len(row['options']))
            p = packed_logits[span.decision_index, allowed]
            single_ids = chat_prompt_ids(tokenizer, request.state, branch)
            with torch.inference_mode():
                s = model(input_ids=torch.tensor([single_ids], device='cuda')).logits[0, -1, allowed]
            flipped = p.argmax() != s.argmax(); flips += int(flipped)
            material += int(flipped and (p.topk(2).values.diff().abs() >= .1) and (s.topk(2).values.diff().abs() >= .1))
            checked += 1
            if checked >= 64:
                print({'mode': 'eval' if args.eval else 'train', 'checked': checked, 'top1_flips': flips, 'material_flips': material})
                return
if __name__ == '__main__': main()
