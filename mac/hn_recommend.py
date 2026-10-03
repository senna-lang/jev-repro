"""Rank recent Hacker News links by personal relevance with local Jev.

Run `./hn` from the project root. Scores use HN titles, not article contents.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
QUERIES = ("LLM inference", "attention", "model evaluation", "local LLM", "transformer")
PROFILE = (
    "Reader interests: LLM inference and serving, attention mechanisms, language model "
    "evaluation, and running open-weight models locally. Prefer concrete engineering "
    "findings, experiments, implementation details, and careful comparisons. "
    "Do not recommend an article just because it mentions AI or an LLM."
)
QUESTION = (
    "Based only on this Hacker News title, is the linked article likely worth reading "
    "for this reader's interests? Judge relevance and practical substance, not popularity."
)


def fetch_candidates(days: int, per_query: int) -> list[dict]:
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    unique: dict[str, dict] = {}
    for query in QUERIES:
        params = urlencode({
            "query": query,
            "tags": "story",
            "numericFilters": f"created_at_i>{since}",
            "hitsPerPage": per_query,
        })
        with urlopen(f"{SEARCH_URL}?{params}", timeout=15) as response:
            hits = json.load(response)["hits"]
        for hit in hits:
            if hit.get("title") and hit.get("url") and hit.get("objectID"):
                unique.setdefault(hit["objectID"], hit)
    return list(unique.values())


def rank_candidates(candidates: list[dict]) -> list[tuple[float, dict]]:
    from mac.jev_mlx import JevMLX
    from model.packer import Branch

    print(f"Loading local Jev to rank {len(candidates)} HN links...", flush=True)
    model = JevMLX()
    ranked = []
    for hit in candidates:
        print(f"Scoring {len(ranked) + 1}/{len(candidates)}: {hit['title']}", flush=True)
        state = f"{PROFILE}\n\nHacker News article title: {hit['title']}"
        score = model.predict_request(state, [Branch("noul", QUESTION, ["No", "Yes"])])[0][1]
        ranked.append((score, hit))
    ranked.sort(key=lambda item: (item[0], item[1]["created_at_i"]), reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="lookback window (default: 7 days)")
    parser.add_argument("--per-query", type=int, default=2, help="most recent hits per topic (default: 2)")
    parser.add_argument("--limit", type=int, default=5, help="recommendations to show (default: 5)")
    args = parser.parse_args()
    if min(args.days, args.per_query, args.limit) < 1:
        parser.error("--days, --per-query, and --limit must be positive")

    candidates = fetch_candidates(args.days, args.per_query)
    if not candidates:
        print("No linked HN stories matched these topics in the selected window.")
        return
    ranked = rank_candidates(candidates)
    print(f"Ranked {len(ranked)} recent HN links from {len(QUERIES)} topic searches.")
    print("Model scores are uncalibrated and use HN titles only; read the source before commenting.\n")
    for index, (score, hit) in enumerate(ranked[:args.limit], start=1):
        print(f"{index}. {hit['title']}  [ranking score: {score:.3f}]")
        print(f"   Article: {hit['url']}")
        print(f"   HN discussion: https://news.ycombinator.com/item?id={hit['objectID']}\n")


if __name__ == "__main__":
    main()
