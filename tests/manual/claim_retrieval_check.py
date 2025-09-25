"""Quick manual script to sanity-check claim retrieval outputs."""

from __future__ import annotations

import argparse
from textwrap import shorten

from patent_rag_app.retrieval.service import PatentRetrievalService


def run_claim_listing(service: PatentRetrievalService, patent_id: str, top_k: int) -> None:
    print(f"\nTop {top_k} claim chunks for {patent_id}:")
    results = service.search(
        f"show me the claims from {patent_id}",
        patent_id=patent_id,
        top_k=top_k,
    )
    if not results:
        print("  (no chunks returned)")
        return

    for idx, result in enumerate(results, start=1):
        preview = shorten(result.text, width=120, placeholder="…")
        print(f"  {idx:02d}. {result.section_label}: {preview}")


def run_claim_count(service: PatentRetrievalService, patent_id: str) -> None:
    query = f"how many claims are in {patent_id}?"
    result = service.search(query, patent_id=patent_id, top_k=1)
    answer = result[0].text if result else "(no answer returned)"
    print(f"\nCount query -> {answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patent_id",
        default="EP1577413_A1",
        nargs="?",
        help="Patent identifier to check (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of claims to display",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = PatentRetrievalService()
    run_claim_listing(service, args.patent_id, args.top_k)
    run_claim_count(service, args.patent_id)


if __name__ == "__main__":
    main()
