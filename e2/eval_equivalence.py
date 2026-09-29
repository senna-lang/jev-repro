import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2.eval_permutations import expanded_branches
from e2.packed_data import load_requests
from model.packer import PackedRequest, pack
from model.tree_mask import build_position_ids, build_tree_attention_mask
MODEL='Qwen/Qwen3-4B-Instruct-2507'
def main():
 t=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval(); checked=flips=material=0
 for rows,r in load_requests(Path('data/e2/teacher_branches.jsonl'),Path('data/e2/states.jsonl')):
  if rows[0]['split']!='eval': continue
  x=expanded_branches(rows,r.branches); p=pack(PackedRequest(r.state,[b for _,b,_ in x]),t); ids=torch.tensor([p.input_ids],device='cuda'); pos=torch.tensor([build_position_ids(p)],device='cuda'); mask=build_tree_attention_mask(p).cuda()[None,None]
  with torch.inference_mode(): out=m(input_ids=ids,attention_mask=mask,position_ids=pos).logits
  for i,(row,b,_) in enumerate(x):
   labels=torch.tensor([t.encode(z,add_special_tokens=False)[0] for z in 'ABCDEFGHIJKLMNOPQRST'][:len(row['options'])],device='cuda'); single=pack(PackedRequest(r.state,[b]),t)
   with torch.inference_mode(): solo=m(input_ids=torch.tensor([single.input_ids],device='cuda')).logits[0,single.branch_spans[0].decision_index,labels]
   q=out[0,p.branch_spans[i].decision_index,labels]; flip=q.argmax()!=solo.argmax(); flips+=int(flip); material+=int(flip and (q.topk(2).values.diff().abs()>=.1) and (solo.topk(2).values.diff().abs()>=.1)); checked+=1
   if checked>=64: print({'branches':checked,'top1_flips':flips,'material_flips':material}); return
if __name__=='__main__': main()
