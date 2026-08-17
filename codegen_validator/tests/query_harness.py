from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from statistics import mean

import httpx


@dataclass(slots=True)
class QueryCase:
    category: str
    name: str
    query: str
    manual_review: str = "REVIEW"


@dataclass(slots=True)
class QueryOutcome:
    category: str
    name: str
    passed: bool
    pass_at_1: bool
    pass_at_3: bool
    retries_taken: int
    latency_ms: float
    confidence: str
    manual_review: str


def default_query_set() -> list[QueryCase]:
    return [
        QueryCase("simple", "sum_list", "Write a Python function that returns the sum of a list of integers."),
        QueryCase("simple", "reverse_words", "Create a function that reverses words in a sentence."),
        QueryCase("medium", "top_k", "Implement a function that returns the k most frequent elements in a list."),
        QueryCase("medium", "merge_intervals", "Write a function that merges overlapping intervals."),
        QueryCase("hard", "dijkstra", "Implement Dijkstra's algorithm for a weighted graph."),
        QueryCase("hard", "lru_cache", "Implement an LRU cache class with get and put methods."),
        QueryCase("ambiguous", "sort_users", "Sort users by score and date."),
        QueryCase("ambiguous", "calculate_metrics", "Calculate metrics for this dataset and return results."),
        QueryCase(
            "external",
            "db_report",
            "Write a Python function that queries a database for user activity and returns daily aggregates.",
        ),
        QueryCase(
            "external",
            "api_sync",
            "Create a script that fetches paginated data from an API and writes transformed output to a file.",
        ),
    ]


def run_harness(base_url: str, cases: list[QueryCase]) -> list[QueryOutcome]:
    outcomes: list[QueryOutcome] = []
    with httpx.Client(timeout=180.0) as client:
        for case in cases:
            start = time.perf_counter()
            response = client.post(f"{base_url.rstrip('/')}/generate", json={"query": case.query})
            response.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000.0
            payload = response.json()

            passed = bool(payload["passed"])
            retries_taken = int(payload["retries_taken"])
            outcomes.append(
                QueryOutcome(
                    category=case.category,
                    name=case.name,
                    passed=passed,
                    pass_at_1=passed and retries_taken == 0,
                    pass_at_3=passed and retries_taken <= 3,
                    retries_taken=retries_taken,
                    latency_ms=latency_ms,
                    confidence=str(payload["confidence"]),
                    manual_review=case.manual_review,
                )
            )
    return outcomes


def print_report(outcomes: list[QueryOutcome]) -> None:
    print("name,category,passed,pass@1,pass@3,retries,latency_ms,confidence,manual_review")
    for row in outcomes:
        print(
            f"{row.name},{row.category},{row.passed},{row.pass_at_1},{row.pass_at_3},"
            f"{row.retries_taken},{row.latency_ms:.2f},{row.confidence},{row.manual_review}"
        )

    pass_at_1 = mean([1.0 if o.pass_at_1 else 0.0 for o in outcomes]) if outcomes else 0.0
    pass_at_3 = mean([1.0 if o.pass_at_3 else 0.0 for o in outcomes]) if outcomes else 0.0
    avg_retries = mean([o.retries_taken for o in outcomes]) if outcomes else 0.0
    avg_latency = mean([o.latency_ms for o in outcomes]) if outcomes else 0.0

    print()
    print(f"Summary: pass@1={pass_at_1:.3f} pass@3={pass_at_3:.3f} avg_retries={avg_retries:.2f} avg_latency_ms={avg_latency:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query harness against codegen_validator API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for API server")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outcomes = run_harness(base_url=args.base_url, cases=default_query_set())
    print_report(outcomes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

