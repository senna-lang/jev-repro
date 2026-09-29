"""Controlled packed-versus-naive latency measurements for Jev checkpoints."""
from __future__ import annotations
import argparse, json, platform, statistics, time
from pathlib import Path
import torch
from legacy.baselines.naive_reencode import naive_reencode_forward
from legacy.model_readout.forward import JevModel
from model.packer import Branch, PackedRequest


def _pct(xs:list[float], q:float)->float: return sorted(xs)[int((len(xs)-1)*q)]
def _request(n:int, state_words:int, options:int)->PackedRequest:
    state=" ".join(["context"]*state_words)
    branches=[Branch("choice", f"Select the best category for item {i}.", [f"option {j}" for j in range(options)]) for i in range(n)]
    return PackedRequest(state, branches)
def _measure(fn, warmups:int, repeats:int)->dict:
    for _ in range(warmups): fn()
    xs=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); xs.append((time.perf_counter()-t)*1000)
    return {"p50_ms":statistics.median(xs),"p95_ms":_pct(xs,.95),"p99_ms":_pct(xs,.99),"samples_ms":xs}
def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--branch-counts",type=int,nargs="+",default=[1,2,4,8,16]);p.add_argument("--state-words",type=int,nargs="+",default=[32,128]);p.add_argument("--option-counts",type=int,nargs="+",default=[3,6]);p.add_argument("--warmups",type=int,default=3);p.add_argument("--repeats",type=int,default=10)
    a=p.parse_args(); ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=JevModel();model.pointer.load_state_dict(ckpt["state_dict"])
    rows=[]
    for n in a.branch_counts:
      for s in a.state_words:
       for k in a.option_counts:
        req=_request(n,s,k); packed=_measure(lambda:model.forward(req),a.warmups,a.repeats); naive=_measure(lambda:naive_reencode_forward(model,req),a.warmups,a.repeats)
        rows.append({"branches":n,"state_words":s,"options":k,"packed":packed,"naive_reencode":naive,"p50_speedup":naive["p50_ms"]/packed["p50_ms"]})
    out={"measurement":"warm end-to-end forward; fixed synthetic text", "manifest":{"checkpoint":str(a.checkpoint),"platform":platform.platform(),"torch":torch.__version__,"warmups":a.warmups,"repeats":a.repeats},"rows":rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n");print(json.dumps({"output":str(a.output),"conditions":len(rows)},ensure_ascii=False))
if __name__=="__main__": main()
