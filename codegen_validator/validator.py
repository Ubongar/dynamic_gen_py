from __future__ import annotations

import ast
import functools
import logging
import os
import platform
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

_DEFAULT_TIMEOUT_SEC = 10
_UNBOUNDED_TIMEOUT_CPU_BACKSTOP_SEC = 300
_DEFAULT_MEMORY_LIMIT_MB = 256
_MAX_OUTPUT_CHARS = 4000

_ENVIRONMENT_ISSUE_PREFIX = "ENVIRONMENT ISSUE (not a code bug): "

# Exported so agent.py can detect this specific failure and skip sending it
# to the LLM repair loop, repairing code that was never actually broken.
ENVIRONMENT_ERROR_MARKER = "Environment Error (missing pip in sandbox)."


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Cap captured stdout/stderr so a pathological program can't balloon the result."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... [truncated, {omitted} more characters]"


def _apply_resource_limits(memory_limit_mb: int, cpu_limit_sec: int) -> None:
    """
    preexec_fn target: runs inside the forked child, before exec, POSIX only.
    Caps address space and CPU time as a defense-in-depth backstop behind the
    subprocess timeout. This is process-level resource limiting, not a full
    security sandbox (no filesystem or network isolation).
    """
    import resource  # POSIX-only stdlib module; not resolvable by static analysis on Windows

    memory_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))  # type: ignore[attr-defined]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_sec, cpu_limit_sec))  # type: ignore[attr-defined]


def _is_missing_dependency_failure(stderr: str) -> bool:
    """
    Detects the specific pattern where generated code correctly tries an
    import, falls back to auto-installing the package on ImportError, and
    that install fails because the execution sandbox itself has no working
    pip (a broken/minimal venv), not because the generated code is wrong.

    Deliberately strict: a missing-module error alone is not enough (that's
    the normal "forgot to import X" case), and neither is a generic
    CalledProcessError or the word "pip" appearing anywhere in stderr, since
    generated code can raise CalledProcessError or mention pip for reasons
    that have nothing to do with the sandbox's own pip being broken. The
    pip-failure indicator must appear on the SAME line as "pip" to count.
    """
    if not stderr:
        return False

    lower = stderr.lower()
    has_missing_module = "modulenotfounderror" in lower or "no module named " in lower
    if not has_missing_module:
        return False

    for line in stderr.splitlines():
        line_lower = line.lower()
        if "pip" not in line_lower:
            continue
        if "no module named pip" in line_lower:
            return True
        if "command not found" in line_lower:
            return True
        if "calledprocesserror" in line_lower:
            return True

    return False


class Validator:
    def __init__(
        self,
        llm_client: LLMClient,
        execution_timeout_sec: int | None = _DEFAULT_TIMEOUT_SEC,
        memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
        review_llm_client: LLMClient | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._execution_timeout_sec = execution_timeout_sec
        self._memory_limit_mb = memory_limit_mb
        # Opt-in only: defaults to the SAME client as everything else, so
        # accuracy is unchanged unless the caller explicitly wires in a
        # lighter model for review_logic.
        self._review_llm_client = review_llm_client or llm_client

    def static_check(self, code: str) -> CheckResult:
        self._log_stage("static_check_start")

        if not code.strip():
            self._log_stage("static_check_fail")
            return CheckResult(
                passed=False,
                issues=["Generated code was empty."],
                confidence="high",
                error="Empty code.",
            )

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
        # Pyflakes findings like unused imports are style, not correctness,
        # so they stay informational. "undefined name" means the code WILL
        # crash at runtime, that is a real bug and must not slip through as
        # a silent pass.
        critical = [n for n in notes if "undefined name" in n]
        if critical:
            self._log_stage("static_check_fail")
            return CheckResult(
                passed=False,
                issues=critical,
                confidence="high",
                error="Pyflakes detected undefined name(s).",
                notes=notes,
            )
        self._log_stage("static_check_pass")
        return CheckResult(
            passed=True,
            issues=[],
            confidence="high",
            notes=notes,
        )

    def execution_check(self, code: str, tests: str = "") -> CheckResult:
        """
        Runs generated code, together with its accompanying tests, as a
        subprocess to confirm the code actually behaves correctly when
        exercised, not just that it parses. If `tests` is empty this
        degrades to a syntax/import-only smoke test and CANNOT catch bugs
        inside functions that are never called, callers should treat an
        empty-tests pass with reduced confidence.

        This provides process isolation via subprocess plus resource limits
        (CPU time / memory, POSIX only) but is NOT a full security sandbox:
        it has no filesystem or network restriction. Do not point this at
        code from an untrusted or adversarial source without a real
        container or restricted-user sandbox in front of it. The argv
        passed to subprocess.run is a fixed two-element list (interpreter
        path + temp file path), never a shell string, so this is not
        vulnerable to shell/argument injection; the generated code itself
        is executed by design, which is the intended behavior of this check.
        """
        self._log_stage("execution_check_start")

        if tests.strip():
            try:
                ast.parse(tests)
            except SyntaxError as exc:
                # `code` already passed static_check by the time execution_check runs,
                # so a syntax error here is unambiguously in the generated tests, not
                # the code, tell the repair loop that precisely instead of a vague
                # "SyntaxError in temp file" it can't attribute to either part.
                issue = (
                    f"The generated TESTS (not the code) contain a SyntaxError: "
                    f"{exc.msg} (line {exc.lineno}, offset {exc.offset}). "
                    f"The code itself is syntactically valid. Fix only the 'tests' field."
                )
                self._log_stage("execution_check_tests_syntax_error")
                return CheckResult(
                    passed=False,
                    issues=[issue],
                    confidence="high",
                    error="Tests SyntaxError.",
                )

        combined = code if not tests.strip() else f"{code}\n\n# --- tests ---\n{tests}"

        temp_path: str | None = None
        try:
            temp_path = self._write_temp_file(combined)
        except OSError as exc:
            self._log_stage("execution_check_write_error")
            return CheckResult(
                passed=False,
                issues=[f"Could not write temporary file for execution: {exc!s}"],
                confidence="low",
                error="Execution Setup Error",
            )

        preexec_fn = None
        if platform.system() != "Windows":
            # CPU-seconds and wall-clock seconds measure different things:
            # I/O-bound code can run past its CPU budget well before the
            # wall-clock timeout below fires. Giving the CPU limit a few
            # seconds of headroom means the wall-clock timeout (which is
            # caught cleanly as TimeoutExpired) is the one that normally
            # fires first, rather than an abrupt SIGKILL from RLIMIT_CPU.
            # If the wall-clock timeout is disabled (None), this CPU limit
            # becomes the ONLY backstop, so fall back to a generous but
            # still bounded value instead of leaving it fully unbounded.
            cpu_limit_sec = (
                self._execution_timeout_sec + 5
                if self._execution_timeout_sec is not None
                else _UNBOUNDED_TIMEOUT_CPU_BACKSTOP_SEC
            )
            preexec_fn = functools.partial(
                _apply_resource_limits,
                self._memory_limit_mb,
                cpu_limit_sec,
            )
        elif self._execution_timeout_sec is None:
            # Windows has no RLIMIT_CPU backstop at all, so a disabled
            # wall-clock timeout here means truly no bound on runtime,
            # including for genuinely infinite loops.
            LOGGER.warning(
                "execution_timeout_sec is disabled on Windows with no CPU-limit "
                "backstop available, generated code (including infinite loops) "
                "can run indefinitely."
            )

        try:
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._execution_timeout_sec,
                check=False,
                preexec_fn=preexec_fn,
            )

            stdout = _truncate(result.stdout.strip())
            stderr = _truncate(result.stderr.strip())

            if result.returncode == 0:
                self._log_stage("execution_check_pass")
                confidence = "high" if tests.strip() else "low"
                notes = ["Execution successful. Output:\n" + stdout]
                if not tests.strip():
                    notes.append(
                        "No tests were provided, this only confirms the code parses "
                        "and runs without raising, it does NOT confirm the logic is correct."
                    )
                return CheckResult(
                    passed=True,
                    issues=[],
                    confidence=confidence,
                    notes=notes,
                )

            self._log_stage("execution_check_fail")

            if _is_missing_dependency_failure(stderr):
                # The generated code's own logic never got to run, it failed
                # while trying to auto-install a missing package because this
                # sandbox's venv has no working pip. Surface this distinctly
                # so it isn't mistaken for a real logic bug in the generated
                # code, and don't send it into the repair loop as if the code
                # itself needs changing, low confidence reflects that we did
                # not actually verify the code's behavior.
                self._log_stage("execution_check_environment_issue")
                issue = (
                    f"{_ENVIRONMENT_ISSUE_PREFIX}the generated code's dependency "
                    f"auto-install fallback failed because this execution sandbox "
                    f"has no working pip. This does not indicate a problem with the "
                    f"generated code's logic. Fix by running 'python -m ensurepip "
                    f"--upgrade' in the sandbox venv, or pre-installing the needed "
                    f"packages there.\nOriginal error:\n{stderr}"
                )
                return CheckResult(
                    passed=False,
                    issues=[issue],
                    confidence="low",
                    error=ENVIRONMENT_ERROR_MARKER,
                )

            issue = f"Runtime Error (Exit Code {result.returncode}):\n{stderr}"
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
                issues=[
                    f"Execution timed out after {self._execution_timeout_sec} "
                    "seconds. Check for infinite loops."
                ],
                confidence="high",
                error="Timeout Error",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            LOGGER.exception("Execution environment failure while running generated code")
            self._log_stage("execution_check_system_error")
            return CheckResult(
                passed=False,
                issues=[f"Execution environment failure: {exc!s}"],
                confidence="low",
                error="Execution Environment Error",
            )
        finally:
            if temp_path and os.path.exists(temp_path):
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
        review = self._review_llm_client.review_logic(prompt=prompt)
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
    def _write_temp_file(code: str) -> str:
        # Explicitly force utf-8 encoding to prevent UnicodeEncodeError on Windows CP1252 systems
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(code)
            return temp_file.name

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