from __future__ import annotations

import argparse
import json
import sys

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for codegen_validator API")
    parser.add_argument("query", help="Natural language query to generate code for")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for API server")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = f"{args.base_url.rstrip('/')}/generate"
    payload = {"query": args.query}
    response = httpx.post(endpoint, json=payload, timeout=120.0)
    response.raise_for_status()
    data = response.json()

    print("Generated Code:")
    print(data["code"])
    print()
    print(f"Description: {data['description']}")
    print(f"Assumptions: {json.dumps(data['assumptions'])}")
    print(
        f"PASS/FAIL: {'PASS' if data['passed'] else 'FAIL'} | "
        f"confidence={data['confidence']} | retries_taken={data['retries_taken']}"
    )
    if data["issues"]:
        print("Issues:")
        for issue in data["issues"]:
            print(f"- {issue}")
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

