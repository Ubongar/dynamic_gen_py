from __future__ import annotations

import ast
import logging
from datetime import datetime, timezone
from typing import cast

from .llm_client import LLMClient, LLMClientError
from .models import GenResult


LOGGER = logging.getLogger(__name__)

_STUB_HELPER_NAME = "_stub_missing_package"


def _is_importerror(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Name) and node.id == "ImportError"


def _handler_calls_stub(handler: ast.ExceptHandler) -> bool:
    for stmt in ast.walk(handler):
        if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name) and stmt.func.id == _STUB_HELPER_NAME:
            return True
    return False


class _MockStripper(ast.NodeTransformer):
    """
    Deterministic, local removal of the exact mock-stubbing scaffolding the
    GENERATOR_PROMPT mandates (the _stub_missing_package helper plus the
    try/import/except-ImportError/stub pattern). Safe because we control the
    exact shape of what's being generated, no network call needed for the
    common case.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | None:
        if node.name == _STUB_HELPER_NAME:
            return None
        return cast(ast.FunctionDef, self.generic_visit(node))

    def visit_Try(self, node: ast.Try) -> ast.stmt:
        if (
            len(node.body) == 1
            and isinstance(node.body[0], (ast.Import, ast.ImportFrom))
            and len(node.handlers) == 1
            and _is_importerror(node.handlers[0].type)
            and _handler_calls_stub(node.handlers[0])
        ):
            # Collapse to just the real import, drop the stub fallback entirely.
            return node.body[0]
        return cast(ast.stmt, self.generic_visit(node))


def _strip_mocks_locally(code: str) -> str | None:
    """
    Returns cleaned source with mock scaffolding removed, or None if the
    code didn't match the expected template closely enough to safely strip
    (still references the stub helper afterward), signaling the caller to
    fall back to an LLM-based cleanup instead.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    cleaned_tree = _MockStripper().visit(tree)
    ast.fix_missing_locations(cleaned_tree)
    cleaned = ast.unparse(cleaned_tree)
    if _STUB_HELPER_NAME in cleaned:
        return None
    return cleaned

GENERATOR_PROMPT = (
    "You are an elite Python system architect. Your objective is to generate highly robust, production-ready Python code. "
    "You MUST adhere to the following strict constraints:\n\n"
    "1. INTELLIGENT DEFAULTS FOR VAGUE PROMPTS: If the request lacks specific technical choices (e.g., database type, file format, or framework), "
    "DO NOT fail or leave code empty unless it is completely impossible to guess. Instead, choose a sensible standard default (e.g., SQLite for databases, CSV for file parsing), "
    "and you MUST explicitly document what you chose in the 'assumptions' JSON array so the user knows what parameter was defaulted.\n"
    "1b. NEVER SILENTLY SUBSTITUTE A DIFFERENT TECHNOLOGY: If the user explicitly names a specific technology (e.g. 'using MySQL', 'with Redis', "
    "'via a REST API'), the generated 'code' MUST use that exact technology, this is non-negotiable and does not fall under rule 1's defaulting. "
    "This holds even though this sandbox has no live external services (no real MySQL/Postgres/Redis server, no network access) to test against. "
    "Do NOT swap the requested technology for something locally runnable (e.g. writing MySQL-requested code as sqlite3) just to make the 'tests' "
    "field pass execution, that produces code the user did not ask for and will not work against their real MySQL setup. Instead, satisfy rule 5's "
    "requirement for real, calling tests by mocking the external dependency: use unittest.mock (e.g. unittest.mock.patch / MagicMock) to stub the "
    "connection/client object so the actual requested driver code (mysql-connector-python, psycopg2, redis-py, requests, etc.) is exercised and "
    "asserted on without needing a live server. If the driver PACKAGE ITSELF is not installed in this sandbox (e.g. `import mysql.connector`, "
    "`import psycopg2`, `import redis`, `import pymongo`, `import boto3` etc. raise ModuleNotFoundError, not just a connection error), you MUST stub "
    "it using this EXACT generic helper, copied verbatim, at the top of 'code' before any import of the missing package. It correctly handles both "
    "flat packages (psycopg2, redis) and dotted ones (mysql.connector, google.cloud.storage) by registering every parent level in sys.modules with "
    "the child attached as an attribute, do not hand-write a package-specific version of this, ALWAYS use this helper for ANY missing driver:\n"
    "    import sys, types, unittest.mock as mock\n"
    "    def _stub_missing_package(dotted_name):\n"
    "        parent = None\n"
    "        accumulated = ''\n"
    "        parts = dotted_name.split('.')\n"
    "        for i, part in enumerate(parts):\n"
    "            accumulated = f'{accumulated}.{part}' if accumulated else part\n"
    "            if accumulated not in sys.modules:\n"
    "                is_leaf = (i == len(parts) - 1)\n"
    "                mod = mock.MagicMock() if is_leaf else types.ModuleType(accumulated)\n"
    "                sys.modules[accumulated] = mod\n"
    "                if parent is not None:\n"
    "                    setattr(parent, part, mod)\n"
    "            parent = sys.modules[accumulated]\n"
    "        return sys.modules[dotted_name]\n"
    "    mysql_connector = _stub_missing_package('mysql.connector')  # call once per missing package, before importing it\n"
    "Do this BEFORE the `import` statement that needs it. If the repair loop reports "
    "an execution failure that stems from a missing live external service "
    "rather than a real bug in your logic, fix it by adding/adjusting the mock, not by changing which technology the code targets.\n"
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
    "statements that directly CALL every function/class defined in 'code'. CRITICAL: The 'tests' code will be appended "
    "directly to the bottom of the 'code' script during execution. DO NOT write import statements in 'tests' to import the "
    "generated functions or classes (e.g., do not write `from my_module import MyClass`), they will already be in scope! "
    "Write realistic sample inputs and ASSERT on the expected results.\n"
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
    def __init__(self, llm_client: LLMClient, cleanup_llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client
        # Only used as a fallback when local mock-stripping can't fully clean
        # the code (model deviated from the exact template). This path is
        # always re-validated by static_check in agent.py regardless of
        # which model produced it, so using a faster/weaker model here can't
        # reduce overall accuracy, it never touches generate/repair/logic_check.
        self._cleanup_llm_client = cleanup_llm_client or llm_client

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
            "CRITICAL: If the original query named a specific technology (e.g. MySQL, Redis, a REST API), do NOT "
            "change 'code' to use a different, more locally-runnable technology to make an execution failure go "
            "away, that is a wrong fix even if it makes the tests pass. If the failure is because this sandbox has "
            "no live external service to connect to, fix it by mocking that dependency (unittest.mock) instead.\n"
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

    def cleanup(self, code: str) -> str:
        self._log_stage("cleanup_start")
        local_clean = _strip_mocks_locally(code)
        if local_clean is not None:
            self._log_stage("cleanup_end_local")
            return local_clean
        try:
            clean_code = self._cleanup_llm_client.cleanup_code(code)
            self._log_stage("cleanup_end_llm_fallback")
            return clean_code
        except LLMClientError as exc:
            # If cleanup fails due to rate limits or parsing, we gracefully 
            # fall back to returning the original mocked code instead of crashing.
            self._log_stage(f"cleanup_failed: {exc}")
            return code

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())