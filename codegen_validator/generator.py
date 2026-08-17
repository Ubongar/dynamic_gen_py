from __future__ import annotations

import logging
from datetime import datetime, timezone

from .llm_client import LLMClient
from .models import GenResult


LOGGER = logging.getLogger(__name__)

GENERATOR_PROMPT = (
    "You are an elite Python system architect. Your objective is to generate highly robust, production-ready Python code. "
    "You MUST adhere to the following strict constraints:\n\n"
    "1. AMBIGUITY REJECTION (NO HALLUCINATIONS): If the request involves databases, APIs, specific algorithms, file formats, or frameworks, "
    "and the user did not specify which one to use, DO NOT GUESS. Set 'clarification_needed' to a concise question asking for the "
    "missing technology, and leave 'code' and 'description' empty.\n"
    "2. SYNTAX & STRING SAFETY: You MUST use Python raw strings (e.g., r\"...\") for ALL docstrings, regular expressions, "
    "file paths, and mathematical/LaTeX formulas to absolutely prevent invalid escape sequence errors.\n"
    "3. ERROR HANDLING & RESILIENCE: Implement robust error handling. Use try/except blocks for any I/O, network calls, "
    "or data parsing. Always use context managers (`with` statements) for resource management like files and connections.\n"
    "4. PRODUCTION QUALITY: The code must include complete type hints (PEP 484), detailed inline comments explaining the step-by-step logic, "
    "and follow standard PEP 8 conventions.\n"
    "5. STRICT JSON FORMAT: Return ONLY valid, parsable JSON matching this schema: "
    "{\"code\": \"string\", \"description\": \"string\", \"assumptions\": [\"string\"], \"clarification_needed\": \"string or null\"}. "
    "The 'assumptions' field MUST be a JSON array. DO NOT wrap the output in markdown code blocks (e.g., no ```json)."
)


class Generator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def generate(self, query: str) -> GenResult:
        self._log_stage("generate_start")
        result = self._llm_client.generate_code(query=query, system_prompt=GENERATOR_PROMPT)
        self._log_stage("generate_end")
        return result

    def repair(self, query: str, code: str, issues: list[str]) -> GenResult:
        self._log_stage("repair_start")
        prompt = (
            "The previous code needs targeted fixes.\n"
            f"Original query:\n{query}\n\n"
            f"Current code:\n{code}\n\n"
            f"Validator issues:\n{issues}\n\n"
            "Return strict JSON matching this schema: {code, description, assumptions}. "
            "Only change what is needed to address the listed issues."
        )
        result = self._llm_client.repair_code(prompt=prompt)
        self._log_stage("repair_end")
        return result

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())

