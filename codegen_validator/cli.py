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

    # Optional Groq-backed client, used ONLY as a fallback for the cleanup
    # step when local AST stripping can't fully clean the mock scaffolding.
    # Never used for generate/repair/logic_check, that path is always
    # re-validated by static_check regardless of which model produced it,
    # so this can't reduce overall accuracy. Opt-in: only activates if
    # GROQ_API_KEY is set.
    cleanup_llm_client = None
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        cleanup_llm_client = LLMClient(
            api_key=groq_api_key,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )

    # Optional lighter OpenAI model for logic_check review. Opt-in only via
    # OPENAI_REVIEW_MODEL, if unset this defaults to the same model as
    # everything else, so accuracy is unchanged unless explicitly configured.
    review_llm_client = None
    review_model = os.getenv("OPENAI_REVIEW_MODEL")
    if review_model:
        review_llm_client = LLMClient(api_key=api_key, model=review_model)

    # Set EXECUTION_TIMEOUT_SEC=none to disable the wall-clock execution
    # timeout entirely (e.g. for code with a slow pip install fallback).
    # On Windows this removes the ONLY runtime bound, see validator.py.
    timeout_env = os.getenv("EXECUTION_TIMEOUT_SEC")
    if timeout_env is not None and timeout_env.strip().lower() in ("none", "0", ""):
        execution_timeout_sec = None
    elif timeout_env is not None:
        execution_timeout_sec = int(timeout_env)
    else:
        execution_timeout_sec = 10

    generator = Generator(llm_client=llm_client, cleanup_llm_client=cleanup_llm_client)
    validator = Validator(
        llm_client=llm_client, review_llm_client=review_llm_client, execution_timeout_sec=execution_timeout_sec
    )

    return CodeAgent(generator=generator, validator=validator, max_retries=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct CLI for codegen_validator")
    parser.add_argument("query", help="Natural language query to generate code for")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print non-blocking reviewer commentary on a PASS (hidden by default).",
    )
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

    print("Cleaned, Ready-to-Run Code:")
    print(result.code)
    print()

    if result.tests:
        print("Sandbox Tests (Note: These may still contain mock objects used for internal validation):")
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

    if result.issues and (not result.passed or args.verbose):
        print("Issues:" if not result.passed else "Reviewer notes (non-blocking):")
        for issue in result.issues:
            print(f"- {issue}")
    elif result.issues and result.passed:
        print(f"({len(result.issues)} non-blocking reviewer note(s) hidden, rerun with --verbose to see them)")

    if result.pipeline_notes:
        print("Pipeline notes (cleanup step, not reviewer findings):")
        for note in result.pipeline_notes:
            print(f"- {note}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())