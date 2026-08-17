from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from io import StringIO

from pyflakes.api import check as pyflakes_check
from pyflakes.reporter import Reporter

from .llm_client import LLMClient
from .models import CheckResult

LOGGER = logging.getLogger(__name__)


class Validator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def static_check(self, code: str) -> CheckResult:
        self._log_stage("static_check_start")
        try:
            ast.parse(code)
        except SyntaxError as exc:
            issue = f"SyntaxError: {exc.msg} (line {exc.lineno}, offset {exc.offset})"
            self._log_stage("static_check_fail")
            return CheckResult(
                passed=False,
                issues=[issue],
                confidence="high",
                error=issue,
            )

        notes = self._run_pyflakes(code)
        self._log_stage("static_check_pass")
        return CheckResult(
            passed=True,
            issues=[],
            confidence="high",
            notes=notes,
        )

    def execution_check(self, code: str) -> CheckResult:
        self._log_stage("execution_check_start")

        # Explicitly force utf-8 encoding to prevent UnicodeEncodeError on Windows CP1252 systems
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                check=False,
            )

            if result.returncode == 0:
                self._log_stage("execution_check_pass")
                return CheckResult(
                    passed=True,
                    issues=[],
                    confidence="high",
                    notes=["Execution successful. Output:\n" + result.stdout.strip()],
                )

            self._log_stage("execution_check_fail")
            issue = f"Runtime Error (Exit Code {result.returncode}):\n{result.stderr.strip()}"
            return CheckResult(
                passed=False,
                issues=[issue],
                confidence="high",
                error="Execution failed.",
            )

        except subprocess.TimeoutExpired:
            self._log_stage("execution_check_timeout")
            return CheckResult(
                passed=False,
                issues=["Execution timed out after 10 seconds. Check for infinite loops."],
                confidence="high",
                error="Timeout Error",
            )
        except Exception as exc:  # noqa: BLE001
            self._log_stage("execution_check_system_error")
            return CheckResult(
                passed=False,
                issues=[f"Sandbox environment failure: {exc!s}"],
                confidence="low",
                error="Sandbox Error",
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def logic_check(self, code: str, description: str, query: str) -> CheckResult:
        self._log_stage("logic_check_start")
        prompt = (
            "You are a rigorous code reviewer. You are NOT executing this code, you "
            "are reading it. Given the original request and the code, determine: "
            "(1) does the code's logic actually accomplish what was asked, "
            "(2) are there any logical errors, off-by-one mistakes, incorrect "
            "algorithm choices, or mismatches between the request and the "
            "implementation, (3) are the stated assumptions reasonable given the "
            "request. Do not comment on whether external resources (DBs, APIs, files) "
            "exist or are reachable, that is out of scope, assume they exist as "
            "referenced. Return JSON: {correct: bool, issues: [...], confidence: "
            "'high'|'medium'|'low'}.\n\n"
            f"Original request:\n{query}\n\n"
            f"Description:\n{description}\n\n"
            f"Code:\n{code}"
        )
        review = self._llm_client.review_logic(prompt=prompt)
        passed = review.correct and review.confidence in ("high", "medium")

        if passed:
            error_msg = None
        elif review.issues:
            error_msg = "; ".join(review.issues)
        else:
            error_msg = "Logic validation failed."

        self._log_stage("logic_check_end")
        return CheckResult(
            passed=passed,
            issues=review.issues,
            confidence=review.confidence,
            error=error_msg,
        )

    @staticmethod
    def _run_pyflakes(code: str) -> list[str]:
        output = StringIO()
        reporter = Reporter(output, output)
        pyflakes_check(code, filename="<generated>", reporter=reporter)
        findings = [line.strip() for line in output.getvalue().splitlines() if line.strip()]
        return findings

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(UTC).isoformat())