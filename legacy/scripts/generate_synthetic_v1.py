"""Generate synthetic support v1 with openai-codex/gpt-5.6-terra via `omp -p`.

Does not use Gold V1 text. Writes incrementally so a crash can resume.
"""

from __future__ import annotations

import json
import re
import subprocess
import hashlib
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synthetic_support import TASKS, DATA_DIR

MODEL = "openai-codex/gpt-5.6-terra"
N_PER_CALL = 10
N_BATCHES = 10
WORKERS = 6
ROOT = Path(__file__).resolve().parent.parent
PARTIAL = DATA_DIR / "support_v1.partial.jsonl"
OUT = DATA_DIR / "support_v1.jsonl"
MANIFEST = DATA_DIR / "support_v1.manifest.json"

SECTORS = [
    "online clothing store", "SaaS productivity app", "mobile bank", "airline",
    "food delivery", "home furniture", "consumer electronics", "telecom ISP",
    "fitness wearable", "streaming video service",
]
STYLES = [
    "1-2 short sentences, clipped",
    "3-4 sentences, polite",
    "a rambling 5-6 sentence paragraph",
    "all-caps frustration then a calm recap",
    "bullet-like but still prose, numbered facts",
    "non-native English, simple words",
    "very formal, almost legal",
    "chatty slang, contractions",
    "copy-pasted error text plus one sentence",
    "a follow-up to a previous ticket, referencing dates",
]
ID_FORMATS = [
    "order codes like ORD-7F2K", "ticket ids like TCK-4419", "no numeric ids at all",
    "ISO dates only (2026-03-12)", "last-4 of a card", "IMEI-style 15 digits",
    "SKU like SKU-BLU-M", "tracking like 1Z999AA10123456784", "invoice INV-2026-118",
    "a username handle only, no numbers",
]
FORBIDDEN = (
    "Do not reuse these tokens anywhere: #48213, #58213, FitPulse, FitBand, "
    "StreamMax, SoundMax, Nimbus, Trailblaze, Aria 2, Maple Ave, Columbus, Portland."
)


def prompt_for(task: str, label: str, batch: int) -> str:
    spec = TASKS[task]
    answers = "\n".join(f"- {l}" for l in spec.labels)
    note = (
        "(The answers are ordered from lowest to highest; match exactly this level, not the neighbouring ones.)\n"
        if spec.kind == "score"
        else ""
    )
    li = spec.labels.index(label)
    return f'''Write {N_PER_CALL} realistic, distinct English messages that customers send to a company's support inbox.

Question a support system will ask about each message: "{spec.question}"
All possible answers:
{answers}

Every message you write must have this answer, clearly for a careful human reader: "{label}"
{note}This batch's setting:
- company type: {SECTORS[(li + batch) % len(SECTORS)]}
- writing style: {STYLES[(li * 3 + batch) % len(STYLES)]}
- identifiers: {ID_FORMATS[(batch + li) % len(ID_FORMATS)]}

Rules:
- Vary product names, cities, and people across the {N_PER_CALL} messages. Invent unique names; do not reuse a product name inside the batch.
- {FORBIDDEN}
- Do not copy or paraphrase the answer text itself; show it through the situation.
- Each message must plausibly fit only this answer, not the neighbouring answers.
- Write only the customer's message. No metadata, no labels.
- Return only JSON of the form {{"messages": ["...", "..."]}} with exactly {N_PER_CALL} strings.
'''


def parse(text: str) -> list[str]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON")
    msgs = json.loads(text[start : end + 1])["messages"]
    msgs = [s.strip() for s in msgs if isinstance(s, str) and s.strip()]
    if len(msgs) != N_PER_CALL:
        raise ValueError(f"got {len(msgs)}")
    return msgs


def invoke(prompt: str) -> str:
    result = subprocess.run(
        ["omp", "-p", "--no-session", "--no-tools", "--model", MODEL, prompt],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def job_key(task: str, label: str, batch: int) -> str:
    return f"{task}\t{label}\t{batch}"


def load_done() -> set[str]:
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.open():
            row = json.loads(line)
            done.add(job_key(row["task"], row["label"], row["batch"]))
    return done


def generate_one(task: str, label: str, batch: int) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            msgs = parse(invoke(prompt_for(task, label, batch)))
            return {"task": task, "label": label, "batch": batch, "messages": msgs}
        except Exception as error:
            last_error = error
    raise RuntimeError(f"{task}/{label[:40]}/b{batch}: {last_error}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    done = load_done()
    work = []
    for task, spec in TASKS.items():
        for label in spec.labels:
            for batch in range(N_BATCHES):
                key = job_key(task, label, batch)
                if key not in done:
                    work.append((task, label, batch))
    print(f"todo={len(work)} already={len(done)} model={MODEL}", flush=True)
    with PARTIAL.open("a") as handle, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_one, t, lab, b): (t, lab, b) for t, lab, b in work}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"  {i}/{len(work)} {row['task']} b{row['batch']}", flush=True)

    rows = []
    for line in PARTIAL.open():
        packed = json.loads(line)
        for i, text in enumerate(packed["messages"]):
            rows.append({"task": packed["task"], "label": packed["label"], "idx": packed["batch"] * N_PER_CALL + i, "text": text})
    # drop exact duplicate texts, keep first
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["text"] in seen:
            continue
        seen.add(row["text"])
        unique.append(row)
    gold = [json.loads(line)["state"] for line in open(ROOT / "evaluation/private/gold_v1_adjudicated_en_mt.jsonl")]

    def grams(s: str, n: int = 8) -> set[str]:
        words = re.findall(r"\w+", s.lower())
        return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}

    gold_grams = set().union(*[grams(s) for s in gold])
    overlap = sum(1 for row in unique if grams(row["text"]) & gold_grams)
    counts = collections.Counter(n for row in unique for n in re.findall(r"#\d{4,}|FitPulse|FitBand", row["text"]))
    by_label = collections.Counter((row["task"], row["label"]) for row in unique)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique))
    manifest = {
        "version": 1,
        "created": "2026-09-24",
        "generator": f"omp -p --no-session --no-tools --model {MODEL}",
        "model": MODEL,
        "usage": "unavailable: omp -p does not expose completion().usage(); token counts not recorded",
        "n_per_label_target": N_PER_CALL * N_BATCHES,
        "n_batches": N_BATCHES,
        "workers": WORKERS,
        "rows": len(unique),
        "dropped_exact_duplicates": len(rows) - len(unique),
        "gold_v1_used_as_input": False,
        "gold_v1_8gram_overlap_rows": overlap,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "min_per_label": min(by_label.values()) if by_label else 0,
        "top_artifacts": counts.most_common(8),
        "tasks": {name: {"kind": spec.kind, "question": spec.question, "labels": list(spec.labels)} for name, spec in TASKS.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "dropped_exact_duplicates", "gold_v1_8gram_overlap_rows", "min_per_label", "top_artifacts")}, ensure_ascii=False))
    if overlap:
        raise SystemExit("Gold V1 8-gram overlap is not 0")
    if min(by_label.values()) < 90:
        raise SystemExit(f"some labels have <90 rows: {min(by_label.values())}")


if __name__ == "__main__":
    main()
