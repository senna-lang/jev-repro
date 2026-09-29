from __future__ import annotations
import argparse, json, math, os, random, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
LABELS = tuple(chr(65 + n) for n in range(20))
SEED = 20260927
PII = (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), re.compile(r"(?x)(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]\d{4}(?!\w)"))
DOMAINS = {"github_archive_filtered": ("software services", "quality assurance", "cybersecurity operations"), "nasa_asrs": ("air transportation", "emergency management"), "ami_icsi": ("engineering services", "scientific research", "software services", "forestry operations", "residential care"), "nhtsa": ("transport equipment manufacturing", "repair and maintenance"), "regulations": ("public administration", "legal services", "environmental compliance")}

def lse(values):
    m=max(values); return m + math.log(sum(math.exp(x-m) for x in values))

def call(session, key, payload, case_id):
    error=None
    for attempt in range(5):
        try:
            response=session.post(API_URL,headers={"Authorization":f"Bearer {key}"},json=payload,timeout=90)
            response.raise_for_status(); return response.json()["choices"][0]
        except (requests.RequestException,KeyError,ValueError) as exc:
            error=exc; time.sleep(2**attempt)
    raise RuntimeError(f"request failed for {case_id}: {error}")

def select(tasks, state):
    candidates=[t for t in tasks if t["split"]==state["split"] and t["domain"] in DOMAINS[state["corpus"]] and not t["type_id"].endswith("_required_condition_present")]
    by_kind={k:[t for t in candidates if t["kind"]==k] for k in ("choice","score","noul")}
    if any(not x for x in by_kind.values()): candidates=[t for t in tasks if t["split"]==state["split"] and not t["type_id"].endswith("_required_condition_present")]; by_kind={k:[t for t in candidates if t["kind"]==k] for k in ("choice","score","noul")}
    rng=random.Random(f"{SEED}:{state['state_id']}"); picked=[rng.choice(by_kind[k]) for k in ("choice","score","noul")]
    picked.append(rng.choice([t for t in candidates if t["type_id"] not in {x["type_id"] for x in picked}]))
    return picked

def augmentation(state, task):
    if state["split"] == "eval":
        return "none"
    value=random.Random(f"{SEED}:{state['state_id']}:{task['type_id']}").random()
    if value < .5: return "none"
    if value < .7: return "paraphrase"
    if value < .8 and task["kind"] == "choice": return "distractor"
    if value < .9 and task["kind"] != "noul": return "k_shift"
    return "unsupported"

def create_prompt(state, task, aug):
    constraint={"choice":"Return 3 to 5 concise mutually exclusive answer options.","score":"Return exactly 4 concise ordered levels from low to high.","noul":"Return exactly [\"No\", \"Yes\"] as options. State the concrete condition in the question; never use the vague phrase 'required condition'."}[task["kind"]]
    augmentation_instruction={"none":"", "paraphrase":"Use non-default, natural wording for the answer options.", "distractor":"Include one plausible option that the record does not support.", "k_shift":"Use one more or one fewer answer level than usual, while keeping 2 to 20 options.", "unsupported":"Ask about a concrete condition for which the record supports none of the answer options."}[aug]
    return f"Write one English record-based question answerable from the record alone and testing exactly the task judgment. The question itself must name the concrete event, decision, condition, or requirement being judged from this record; never ask only about a generic risk, urgency, condition, procedure, or follow-up. Do not repeat names, organizations, identifiers, or source names. Do not invent facts. {constraint} {augmentation_instruction} Return only JSON with keys question and options.\n\nTask type:\ntype_id: {task['type_id']}\ndomain: {task['domain']}\nkind: {task['kind']}\njudgment: {task['judgment']}\n\nRecord:\n{state['text']}"

def parse_question(choice, task, aug):
    obj=json.loads(choice["message"]["content"]); question=obj.get("question"); options=obj.get("options")
    allowed={"choice":range(3,6),"score":(range(3,6) if aug=="k_shift" else range(4,5)),"noul":range(2,3)}[task["kind"]]
    if not isinstance(question,str) or not isinstance(options,list) or len(options) not in allowed or not all(isinstance(x,str) and x.strip() for x in options): raise ValueError("invalid JSON")
    options=[x.strip() for x in options]
    if len(set(x.casefold() for x in options))!=len(options) or (task["kind"]=="noul" and options != ["No","Yes"]): raise ValueError("invalid options")
    if task["kind"] == "noul" and "required condition" in question.casefold():
        raise ValueError("vague Noul condition")
    if any(p.search(question) or any(p.search(x) for x in options) for p in PII): raise ValueError("PII")
    return question.strip(),options

def score_prompt(state,question,options):
    answers="\n".join(f"{LABELS[i]}. {x}" for i,x in enumerate(options))
    return f"Read the record and answer the question. Respond with exactly one listed letter and no other text.\n\nRecord:\n{state['text']}\n\nQuestion: {question}\nAnswers:\n{answers}\n\nAnswer:"

def probs(choice,count):
    alts=choice["logprobs"]["content"][0]["top_logprobs"]; buckets={x:[] for x in LABELS[:count]}; minimum=min(float(x["logprob"]) for x in alts)
    for x in alts:
        label=x["token"].strip()
        if label in buckets: buckets[label].append(float(x["logprob"]))
    missing=[k for k,v in buckets.items() if not v]; values=[lse(v) if v else minimum for v in buckets.values()]; normalizer=lse(values)
    return [math.exp(x-normalizer) for x in values],missing,(math.exp(minimum) if missing else None)

def allocate_augmentations(jobs):
    labels={case:"none" for case,_,_ in jobs}; eligible=[x for x in jobs if x[1]["split"]!="eval"]; rng=random.Random(SEED); rng.shuffle(eligible)
    for aug,count,allowed in (("paraphrase",1000,{"choice","score","noul"}),("distractor",500,{"choice"}),("k_shift",500,{"choice","score"}),("unsupported",500,{"choice","score","noul"})):
        pool=[x for x in eligible if labels[x[0]]=="none" and x[2]["kind"] in allowed]; rng.shuffle(pool)
        if len(pool)<count: raise RuntimeError(f"not enough eligible branches for {aug}")
        for case,_,_ in pool[:count]: labels[case]=aug
    return labels

def generate_one(state,task,key,aug):
    case=f"{state['state_id']}__{task['type_id']}"
    with requests.Session() as session:
        for _ in range(10):
            first=call(session,key,{"model":MODEL,"messages":[{"role":"user","content":create_prompt(state,task,aug)}],"temperature":1,"response_format":{"type":"json_object"},"thinking":{"type":"disabled"}},case)
            try: question,options=parse_question(first,task,aug); break
            except (ValueError,json.JSONDecodeError): pass
        else: raise RuntimeError(f"invalid question {case}")
        second=call(session,key,{"model":MODEL,"messages":[{"role":"user","content":score_prompt(state,question,options)}],"max_tokens":1,"temperature":1,"logprobs":True,"top_logprobs":20,"thinking":{"type":"disabled"}},case)
    values,missing,bound=probs(second,len(options)); record={k:state[k] for k in ("state_id","split","corpus","source_id","window","license")}
    record.update({"case_id":case,"type_id":task["type_id"],"kind":task["kind"],"domain":task["domain"],"question":question,"options":options,"probabilities":values,"missing_labels":missing,"missing_prob_bound":bound,"aug":aug,"teacher":MODEL,"question_temperature":1.0,"score_temperature":1.0})
    return record

def generate(states,tasks,output,key,workers):
    partial=output.with_suffix(".partial.jsonl"); output.parent.mkdir(parents=True,exist_ok=True)
    done={json.loads(x)["case_id"] for x in partial.read_text().splitlines() if x} if partial.exists() else set()
    jobs=[(f"{state['state_id']}__{task['type_id']}",state,task) for state in states for task in select(tasks,state) if f"{state['state_id']}__{task['type_id']}" not in done]
    augs=allocate_augmentations(jobs)
    with partial.open("a") as sink, ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(generate_one,state,task,key,augs[case]) for case,state,task in jobs]
        for future in as_completed(futures):
            record=future.result()
            if record["case_id"] in done: raise RuntimeError(f"duplicate case {record['case_id']}")
            sink.write(json.dumps(record)+"\n"); sink.flush(); os.fsync(sink.fileno()); done.add(record["case_id"])
    rows=[json.loads(x) for x in partial.read_text().splitlines() if x]; output.write_text("".join(json.dumps(x)+"\n" for x in rows)); return {"branches":len(rows),"by_kind":dict(Counter(x["kind"] for x in rows)),"missing":sum(bool(x["missing_labels"]) for x in rows)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--states",type=Path,default=Path("data/e2/states.jsonl")); p.add_argument("--types",type=Path,default=Path("data/e2/task_types.jsonl")); p.add_argument("--output",type=Path,default=Path("data/e2/pilot/teacher_branches.jsonl")); p.add_argument("--limit-states",type=int,default=5); p.add_argument("--workers",type=int,default=8); a=p.parse_args(); key=os.environ.get("DEEPSEEK_API_KEY")
    if not key or a.workers < 1: raise RuntimeError("DEEPSEEK_API_KEY must be set and workers must be positive")
    states=[json.loads(x) for x in a.states.read_text().splitlines() if x][:a.limit_states]; tasks=[json.loads(x) for x in a.types.read_text().splitlines() if x]; print(json.dumps(generate(states,tasks,a.output,key,a.workers)))
if __name__=="__main__": main()
