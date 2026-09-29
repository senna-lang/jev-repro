#!/usr/bin/env python3
"""Extract public-corpus E2 states without materializing full source datasets."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
MIN_TOKENS = 300
MAX_TOKENS = 1500
SEED = 20260926
SPLIT_TARGETS = {
    "train": {"github_archive_filtered": 355, "nasa_asrs": 296, "ami_icsi": 178, "nhtsa": 178, "regulations": 118},
    "val": {"github_archive_filtered": 39, "nasa_asrs": 33, "ami_icsi": 20, "nhtsa": 20, "regulations": 13},
    "eval": {"github_archive_filtered": 95, "nasa_asrs": 79, "ami_icsi": 47, "nhtsa": 47, "regulations": 32},
}
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
PHONE = re.compile(r"(?x)(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]\d{4}(?!\w)")
PII = (EMAIL, IPV4, PHONE)

@dataclass(frozen=True)
class Document:
    source_id: str
    text: str
    license: str


def redact(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = IPV4.sub("[REDACTED_IP]", text)
    return PHONE.sub("[REDACTED_PHONE]", text)


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in PII)


def metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def hf_rows(dataset: str, *, data_files: str | None = None) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": "train", "streaming": True}
    if data_files is None:
        return load_dataset(dataset, **kwargs).shuffle(seed=SEED, buffer_size=10_000)
    return load_dataset("parquet", data_files=data_files, **kwargs).shuffle(seed=SEED, buffer_size=10_000)


def github_documents() -> Iterator[Document]:
    for row in hf_rows("common-pile/github_archive"):
        info = metadata(row.get("metadata"))
        license_name = str(info.get("license", ""))
        # Common Pile records carry the repository license; keep only explicitly permissive records.
        if license_name.lower() not in {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "zlib", "unlicense", "cc0-1.0"}:
            continue
        text = str(row.get("text") or "").strip()
        source_id = str(row.get("id") or "")
        if source_id and text:
            yield Document(source_id, text, license_name)


def asrs_documents() -> Iterator[Document]:
    for row in hf_rows("elihoole/asrs-aviation-reports"):
        source_id = str(row.get("acn_num_ACN") or "")
        kept = {key: value for key, value in row.items() if value not in (None, "", [], {})}
        if source_id and kept:
            yield Document(source_id, json.dumps(kept, ensure_ascii=False, sort_keys=True), "NASA public data")


def qmsum_documents() -> Iterator[Document]:
    for row in hf_rows("pszemraj/qmsum-cleaned"):
        source_id = str(row.get("pid") or row.get("id") or "")
        text = str(row.get("input") or "").strip()
        if source_id and text:
            yield Document(source_id, text, "CC BY 4.0 (AMI/ICSI via QMSum)")


def regulations_documents() -> Iterator[Document]:
    for row in hf_rows("common-pile/regulations"):
        source_id = str(row.get("id") or "")
        text = str(row.get("text") or "").strip()
        info = metadata(row.get("metadata"))
        license_name = str(info.get("license") or "Public Domain")
        if source_id and text:
            yield Document(source_id, text, license_name)


def cfpb_documents() -> Iterator[Document]:
    url = "hf://datasets/vivekkopthsd/cfpb-consumer-complaints@280afed1c919ffc14802e0771b895da323b86a48/complaints.parquet"
    for row in hf_rows("unused", data_files=url):
        source_id = str(row.get("Complaint ID") or row.get("complaint_id") or "")
        text = str(row.get("Consumer complaint narrative") or row.get("consumer_complaint_narrative") or "").strip()
        if source_id and text:
            yield Document(source_id, text, "CC0 / U.S. federal government work")


def nhtsa_documents(cache_dir: Path) -> Iterator[Document]:
    archive = cache_dir / "FLAT_CMPL.zip"
    if not archive.exists():
        response = requests.get("https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip", stream=True, timeout=120)
        response.raise_for_status()
        with archive.open("wb") as destination:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    destination.write(chunk)
    with zipfile.ZipFile(archive) as zipped:
        member = next(name for name in zipped.namelist() if name.lower().endswith(".txt"))
        with zipped.open(member) as raw:
            for raw_line in raw:
                fields = raw_line.decode("latin-1", errors="replace").rstrip("\r\n").split("\t")
                # FLAT_CMPL.txt is headerless. CMPLID is column 0; CMPLDESC is column 19.
                if len(fields) <= 19:
                    continue
                source_id = fields[0].strip()
                text = fields[19].strip()
                if source_id and text:
                    yield Document(source_id, text, "U.S. federal government public-domain data")


def window_text(tokenizer: Any, text: str) -> Iterator[tuple[str, list[int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    for start in range(0, len(ids), MAX_TOKENS):
        end = min(start + MAX_TOKENS, len(ids))
        if end - start < MIN_TOKENS:
            break
        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        chunk = text[char_start:char_end]
        if chunk.strip():
            yield chunk, [char_start, char_end]


def documents(corpus: str, cache_dir: Path) -> Iterator[Document]:
    return {
        "github_archive_filtered": github_documents,
        "nasa_asrs": asrs_documents,
        "ami_icsi": qmsum_documents,
        "nhtsa": lambda: nhtsa_documents(cache_dir),
        "regulations": regulations_documents,
        "cfpb": cfpb_documents,
    }[corpus]()


def extract(corpora: set[str], output: Path, cache_dir: Path, limit_per_corpus: int | None) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    records: list[dict[str, Any]] = []
    used_source_ids: set[tuple[str, str]] = set()
    targets = {split: {corpus: count for corpus, count in quotas.items() if corpus in corpora} for split, quotas in SPLIT_TARGETS.items()}
    for corpus in sorted(corpora):
        remaining = {split: targets[split].get(corpus, 0) for split in targets}
        cap = limit_per_corpus if limit_per_corpus is not None else sum(remaining.values())
        emitted = 0
        for document in documents(corpus, cache_dir):
            if emitted >= cap or not any(remaining.values()):
                break
            source_key = (corpus, document.source_id)
            if source_key in used_source_ids:
                continue
            text = redact(document.text)
            if contains_pii(text):
                raise ValueError(f"PII survived redaction in {corpus}:{document.source_id}")
            for chunk, offsets in window_text(tokenizer, text):
                split = next((name for name in ("train", "val", "eval") if remaining[name] > 0), None)
                if split is None or emitted >= cap:
                    break
                record = {
                    "state_id": hashlib.sha256(f"{corpus}:{document.source_id}:{offsets[0]}:{offsets[1]}".encode()).hexdigest()[:20],
                    "split": split,
                    "corpus": corpus,
                    "source_id": document.source_id,
                    "window": offsets,
                    "license": document.license,
                    "text": chunk,
                }
                records.append(record)
                remaining[split] -= 1
                emitted += 1
                # Keep a source document in only one split even if it has several valid windows.
                used_source_ids.add(source_key)
                break
        missing = {split: count for split, count in remaining.items() if count}
        if missing and limit_per_corpus is None:
            raise RuntimeError(f"{corpus} did not yield enough states: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return {"records": len(records), "by_split": dict(Counter(record["split"] for record in records)), "by_corpus": dict(Counter(record["corpus"] for record in records))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/e2/states.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/e2"))
    parser.add_argument("--corpora", nargs="+", choices=sorted(SPLIT_TARGETS["train"]), default=sorted(SPLIT_TARGETS["train"]))
    parser.add_argument("--limit-per-corpus", type=int, help="Smoke-test cap; skips full quota enforcement.")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(extract(set(args.corpora), args.output, args.cache_dir, args.limit_per_corpus), ensure_ascii=False))

if __name__ == "__main__":
    main()
