"""Generate dev set B (backbone selection) with terra via `omp -p`.

One call per task type in `internal-docs/devset_task_types_v0.json`; each call writes
one record per option so labels are balanced. Gold V1 is never given to the generator;
it is read only for the 8-gram leakage check.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "openai-codex/gpt-5.6-terra"
WORKERS = 6
ROOT = Path(__file__).resolve().parent.parent
TYPES = ROOT / "internal-docs/devset_task_types_v0.json"
OUT_DIR = ROOT / "data/devset"
PARTIAL = OUT_DIR / "devset_b_v0.partial.jsonl"
OUT = OUT_DIR / "devset_b_v0.jsonl"
MANIFEST = OUT_DIR / "devset_b_v0.manifest.json"


def prompt_for(spec: dict) -> str:
    options = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(spec["options"]))
    order = (
        "The answers are ordered from lowest to highest. Each record must fit its level, not the neighbouring ones.\n"
        if spec["kind"] == "score"
        else ""
    )
    return f'''Domain: {spec["domain"]}
Question a worker answers from a short written record: "{spec["question"]}"
Possible answers:
{options}
{order}
Write exactly one realistic English record (60-160 words: an email, note, form, log, or report) for EACH answer, in the listed order.
Rules:
- A careful human reader must reach exactly that answer from the record alone.
- Show the answer through facts in the record. Do not state or paraphrase the answer text.
- Invent distinct names, places, and identifiers per record.
- Write only the record text. No labels or metadata.
Return only JSON: {{"records": ["record for answer 1", "record for answer 2", ...]}} with exactly {len(spec["options"])} strings.
'''


def generate(spec: dict) -> list[dict]:
    last_error = None
    for _ in range(3):
        try:
            result = subprocess.run(
                ["omp", "-p", "--no-session", "--no-tools", "--model", MODEL, prompt_for(spec)],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            text = result.stdout
            records = json.loads(text[text.find("{") : text.rfind("}") + 1])["records"]
            records = [r.strip() for r in records if isinstance(r, str) and r.strip()]
            if len(records) != len(spec["options"]):
                raise ValueError(f"got {len(records)}")
            return [
                {
                    "case_id": f"{spec['id']}__{i}",
                    "type_id": spec["id"],
                    "domain": spec["domain"],
                    "kind": spec["kind"],
                    "question": spec["question"],
                    "options": spec["options"],
                    "label": i,
                    "state": record,
                }
                for i, record in enumerate(records)
            ]
        except Exception as error:  # retry malformed output
            last_error = error
    raise RuntimeError(f"{spec['id']}: {last_error}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = json.loads(TYPES.read_text())
    done = set()
    if PARTIAL.exists():
        done = {json.loads(line)["type_id"] for line in PARTIAL.open()}
    work = [s for s in specs if s["id"] not in done]
    print(f"todo={len(work)} already={len(done)} model={MODEL}", flush=True)
    with PARTIAL.open("a") as handle, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(generate, s) for s in work]
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            handle.flush()
            print(f"  {i}/{len(work)} {rows[0]['type_id']}", flush=True)

    order = {s["id"]: n for n, s in enumerate(specs)}
    rows = sorted((json.loads(line) for line in PARTIAL.open()), key=lambda r: (order[r["type_id"]], r["label"]))
    gold = [json.loads(line)["state"] for line in open(ROOT / "evaluation/private/gold_v1_adjudicated_en_mt.jsonl")]

    def grams(s: str, n: int = 8) -> set[str]:
        words = re.findall(r"\w+", s.lower())
        return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}

    gold_grams = set().union(*[grams(s) for s in gold])
    overlap = sum(1 for r in rows if grams(r["state"]) & gold_grams)
    leaked = sum(1 for r in rows if r["options"][r["label"]].lower() in r["state"].lower() and len(r["options"][r["label"]]) > 4)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "version": 0,
        "created": "2026-09-25",
        "role": "backbone-selection dev set B; never training data",
        "generator": f"omp -p --no-session --no-tools --model {MODEL}",
        "types_file": str(TYPES.relative_to(ROOT)),
        "labels_from": "generator target (one record per option)",
        "rows": len(rows),
        "types": len({r["type_id"] for r in rows}),
        "by_kind": collections.Counter(r["kind"] for r in rows),
        "gold_v1_used_as_input": False,
        "gold_v1_8gram_overlap_rows": overlap,
        "rows_containing_label_text": leaked,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))
    if overlap:
        raise SystemExit("Gold V1 8-gram overlap is not 0")


if __name__ == "__main__":
    main()
