import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2.packed_data import load_requests
from model.packer import PackedRequest, pack
from model.tree_mask import build_position_ids, build_tree_attention_mask
MODEL='Qwen/Qwen3-4B-Instruct-2507'
def main():
 t=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval(); checked=0; flips=0; bad=0
 for rows,request in load_requests(Path('data/e2/teacher_branches.jsonl'),Path('data/e2/states.jsonl')):
  packed=pack(request,t); ids=torch.tensor([packed.input_ids],device='cuda'); pos=torch.tensor([build_position_ids(packed)],device='cuda'); mask=build_tree_attention_mask(packed).cuda()[None,None]
  with torch.inference_mode(): out=m(input_ids=ids,attention_mask=mask,position_ids=pos).logits
  for row,branch,span in zip(rows,request.branches,packed.branch_spans):
   labels=torch.tensor([t.encode(x,add_special_tokens=False)[0] for x in 'ABCDEFGHIJKLMNOPQRST'][:len(row['options'])],device='cuda'); p=out[0,span.decision_index,labels]; one=pack(PackedRequest(request.state,[branch]),t)
   with torch.inference_mode(): s=m(input_ids=torch.tensor([one.input_ids],device='cuda')).logits[0,one.branch_spans[0].decision_index,labels]
   d=(p.float()-s.float()).abs(); flips+=int(p.argmax()!=s.argmax()); bad+=int(p.argmax()!=s.argmax() and abs(float(p.topk(2).values[0]-p.topk(2).values[1]))>=.1 and abs(float(s.topk(2).values[0]-s.topk(2).values[1]))>=.1); checked+=1
   if checked==64: print({'checked':checked,'top1_flips':flips,'material_flips':bad}); return
if __name__=='__main__': main()
