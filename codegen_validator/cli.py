from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import CodeAgent
from .generator import Generator
from .llm_client import LLMClient
from .validator import Validator


def create_agent() -> CodeAgent:
    # Use environment variable or default to the provided key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required.")
        
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
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

    # Run the agent directly with the provided query
    result = agent.run(args.query)
    
    if result.clarification_needed:
        print("\n🛑 Missing Information Detected:")
        print(f"The AI needs clarification: {result.clarification_needed}")
        print("Please run the CLI again with a more specific prompt.\n")
        return 1

    print("Generated Code:")
    print(result.code)
    print()
    print(f"Description: {result.description}")
    print(f"Assumptions: {json.dumps(result.assumptions)}")
    print(
        f"PASS/FAIL: {'PASS' if result.passed else 'FAIL'} | "
        f"confidence={result.confidence} | retries_taken={result.retries_taken}"
    )
    
    if result.issues:
        print("Issues:")
        for issue in result.issues:
            print(f"- {issue}")
            
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())