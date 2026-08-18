from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import CodeAgent
from .generator import Generator
from .llm_client import LLMClient, LLMClientError
from .validator import Validator
from dotenv import load_dotenv

load_dotenv()

def create_agent() -> CodeAgent:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    model = os.getenv("OPENAI_MODEL", "olori-image")
    llm_client = LLMClient(api_key=api_key, model=model)
    generator = Generator(llm_client=llm_client)
    validator = Validator(llm_client=llm_client)

    return CodeAgent(generator=generator, validator=validator, max_retries=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct CLI for codegen_validator")
    parser.add_argument("query", help="Natural language query to generate code for")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        agent = create_agent()
    except RuntimeError as exc:
        print(f"Initialization Error: {exc}", file=sys.stderr)
        return 2

    try:
        result = agent.run(args.query)
    except LLMClientError as exc:
        print(f"Generation Error: {exc}", file=sys.stderr)
        print("This usually clears on a retry, try running the same command again.", file=sys.stderr)
        return 3

    if result.clarification_needed:
        print("\nMissing Information Detected:")
        print(f"The AI needs clarification: {result.clarification_needed}")
        print("Please run the CLI again with a more specific prompt.\n")
        return 1

    print("Generated Code:")
    print(result.code)
    print()

    if result.tests:
        print("Generated Tests:")
        print(result.tests)
        print()

    print(f"Description: {result.description}")

    if result.assumptions:
        print("Notice: The prompt was broad, so the following default assumptions were made:")
        for assumption in result.assumptions:
            print(f"  - {assumption}")
        print("  (If you want to change these, specify the exact parameters in your prompt!)")

    print(
        f"\nPASS/FAIL: {'PASS' if result.passed else 'FAIL'} | "
        f"confidence={result.confidence} | retries_taken={result.retries_taken}"
    )

    if result.issues:
        print("Issues:")
        for issue in result.issues:
            print(f"- {issue}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())