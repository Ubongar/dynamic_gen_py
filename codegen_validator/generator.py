from __future__ import annotations

import logging
from datetime import datetime, timezone

from .llm_client import LLMClient
from .models import GenResult


LOGGER = logging.getLogger(__name__)

GENERATOR_PROMPT = (
    "You are an elite Python system architect. Your objective is to generate highly robust, production-ready Python code. "
    "You MUST adhere to the following strict constraints:\n\n"
    "1. INTELLIGENT DEFAULTS FOR VAGUE PROMPTS: If the request lacks specific technical choices (e.g., database type, file format, or framework), "
    "DO NOT fail or leave code empty unless it is completely impossible to guess. Instead, choose a sensible standard default (e.g., SQLite for databases, CSV for file parsing), "
    "and you MUST explicitly document what you chose in the 'assumptions' JSON array so the user knows what parameter was defaulted.\n"
    "2. SYNTAX & STRING SAFETY: You MUST use Python raw strings (e.g., r\"...\") for ALL docstrings, regular expressions, "
    "file paths, and mathematical/LaTeX formulas to absolutely prevent invalid escape sequence errors.\n"
    "3. ERROR HANDLING & RESILIENCE: Implement robust error handling. Use try/except blocks for any I/O, network calls, "
    "or data parsing. Always use context managers (`with` statements) for resource management like files and connections. "
    "CRITICAL: for sqlite3 specifically, `with connection:` only wraps a transaction (commit/rollback), it does NOT close "
    "the connection. Any sqlite3.Connection you open must be explicitly closed with conn.close() in a finally block or "
    "opened via `with contextlib.closing(sqlite3.connect(...)) as conn:` instead. Never leave a sqlite3 connection to a "
    "real file open after a function returns, this leaks file handles and causes file-locking errors on Windows.\n"
    "4. PRODUCTION QUALITY: The code must include complete type hints (PEP 484), detailed inline comments explaining the step-by-step logic, "
    "and follow standard PEP 8 conventions.\n"
    "5. TESTS ARE REQUIRED, NOT OPTIONAL: Alongside 'code', you MUST produce a 'tests' field containing standalone Python "
    "statements that actually IMPORT NOTHING EXTRA and directly CALL every function/class defined in 'code' with realistic "
    "sample inputs, and ASSERT on the expected results. Do not just define tests, they must execute when the combined "
    "code+tests file is run as a script. If the request references an external resource that cannot be exercised in "
    "isolation (a real database, a live API, a filesystem path that won't exist), write the tests using an in-memory or "
    "mocked stand-in (e.g. sqlite3 in-memory, unittest.mock) so the tests still run and prove the logic works, and note "
    "this substitution in 'assumptions'. Never leave 'tests' empty unless the request produced no callable code at all.\n"
    "6. STRICT JSON FORMAT: Return ONLY valid, parsable JSON matching this schema: "
    "{\"code\": \"string\", \"tests\": \"string\", \"description\": \"string\", \"assumptions\": [\"string\"], "
    "\"clarification_needed\": \"string or null\"}. "
    "The 'assumptions' field MUST be a JSON array. DO NOT wrap the output in markdown code blocks (e.g., no ```json). "
    "JSON ESCAPING RULE (CRITICAL): apostrophes and single quotes inside string values must NEVER be escaped with a "
    "backslash, \\' is not valid JSON and will cause the entire response to be rejected. Write apostrophes plain, "
    "e.g. \"the student's score\" not \"the student\\'s score\". Only backslash-escape double quotes (\\\"), "
    "backslashes (\\\\), and control characters (\\n, \\t) as JSON requires."
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
            "The previous code and/or its tests need targeted fixes.\n"
            f"Original query:\n{query}\n\n"
            f"Current code:\n{code}\n\n"
            f"Validator issues:\n{issues}\n\n"
            "Return strict JSON matching this schema: {\"code\": \"string\", \"tests\": \"string\", "
            "\"description\": \"string\", \"assumptions\": [\"string\"], \"clarification_needed\": \"string or null\"}.\n"
            "CRITICAL: The 'description' field MUST describe what the final repaired code actually does functionally, "
            "NOT a meta-explanation of the bugs you fixed or the retries taken. "
            "The 'tests' field MUST still call every function in 'code' and assert on real results, do not leave it empty. "
            "Only change what is needed to address the listed issues."
        )
        result = self._llm_client.repair_code(prompt=prompt)
        self._log_stage("repair_end")
        return result

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())