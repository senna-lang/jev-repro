"""Generate Terra reference JSONL through OMP headless print mode; no API key required."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from evaluation.reference_terra import DEFAULT_MODEL, normalise, prompt_for
from evaluation.schema import load_cases

def invoke(model:str, prompt:str, cwd:Path)->dict:
    result=subprocess.run(["omp","-p","--no-session","--no-tools","--model",model,prompt],cwd=cwd,text=True,capture_output=True,check=True)
    text=result.stdout.strip()
    start=text.find('{'); end=text.rfind('}')
    if start < 0 or end < start: raise ValueError(f"OMP returned no JSON: {text[-500:]}")
    return json.loads(text[start:end+1])
def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model',default='openai-codex/gpt-5.6-terra');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); cases=load_cases(a.dataset)
    if a.dry_run: print(prompt_for(cases[0])); return
    rows=[]
    for case in cases: rows.extend(normalise(case,invoke(a.model,prompt_for(case),Path.cwd())))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps({'role':'reference-only; never Gold evidence','runner':'omp -p --no-session --no-tools','model':a.model,'dataset':str(a.dataset),'cases':len(cases),'branches':len(rows)},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'branches':len(rows),'model':a.model},ensure_ascii=False))
if __name__=='__main__': main()
