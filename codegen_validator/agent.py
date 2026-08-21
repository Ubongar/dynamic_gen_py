from __future__ import annotations

import logging
from datetime import datetime, UTC

from .generator import Generator
from .llm_client import LLMClientError
from .models import AgentResult, GenResult
from .validator import ENVIRONMENT_ERROR_MARKER, Validator


LOGGER = logging.getLogger(__name__)


class CodeAgent:
    def __init__(self, generator: Generator, validator: Validator, max_retries: int = 3) -> None:
        if max_retries < 1:
            # max_retries=0 used to make the validation loop's `for` never
            # execute, silently returning an UNVALIDATED result marked
            # passed=False with no issues, no static/execution/logic check
            # ever ran. Fail loudly at construction instead.
            raise ValueError(f"max_retries must be >= 1, got {max_retries}.")
        self._generator = generator
        self._validator = validator
        self._max_retries = max_retries

    def run(self, query: str) -> AgentResult:
        self._log_stage("agent_run_start")
        try:
            result = self._generator.generate(query)
        except LLMClientError as exc:
            return self._llm_failure_result(exc, retries_taken=0)

        clarification = self._clarification_result(result)
        if clarification is not None:
            return clarification

        retries_taken = 0
        last_issues: list[str] = []
        last_confidence = "low"

        for attempt in range(self._max_retries):
            # 1. Static Syntax Check
            static = self._validator.static_check(result.code)
            if not static.passed:
                last_issues = static.issues
                last_confidence = static.confidence
                if attempt == self._max_retries - 1:
                    break
                retries_taken += 1
                result, failure = self._safe_repair(
                    query, result.code, [static.error or "Syntax validation failed."], retries_taken
                )
                if failure is not None:
                    return failure
                assert result is not None
                clarification = self._clarification_result(result, retries_taken=retries_taken)
                if clarification is not None:
                    return clarification
                continue

            # 2. Execution Check (runs code + tests together, real dynamic verification)
            execution = self._validator.execution_check(result.code, result.tests)
            if not execution.passed:
                last_issues = execution.issues
                last_confidence = execution.confidence
                if execution.error == ENVIRONMENT_ERROR_MARKER:
                    # The sandbox itself is broken (no pip), not the generated
                    # code. Sending this to repair would waste an LLM call
                    # asking the model to "fix" code that was never wrong, and
                    # it has no way to fix a problem in the local environment.
                    self._log_stage("agent_run_environment_issue")
                    break
                if attempt == self._max_retries - 1:
                    break
                retries_taken += 1
                result, failure = self._safe_repair(query, result.code, execution.issues, retries_taken)
                if failure is not None:
                    return failure
                assert result is not None
                clarification = self._clarification_result(result, retries_taken=retries_taken)
                if clarification is not None:
                    return clarification
                continue

            # 3. Logic Check
            logic = self._validator.logic_check(result.code, result.description, query)
            last_issues = logic.issues
            last_confidence = logic.confidence
            if logic.passed:
                self._log_stage("agent_run_pass")
                
                # --- CLEANUP STEP ---
                self._log_stage("agent_run_cleanup_start")
                clean_code = self._generator.cleanup(result.code)
                pipeline_notes: list[str] = []

                if clean_code != result.code:
                    # Validate the cleaned code to prevent shipping broken syntax
                    clean_static = self._validator.static_check(clean_code)
                    if not clean_static.passed:
                        clean_code = result.code
                        pipeline_notes.append("Post-validation cleanup introduced a syntax error and was rolled back.")
                elif "unittest.mock" in result.code or "_stub_missing_package" in result.code:
                    # Only flag a failure if there was actually something to clean
                    pipeline_notes.append("Post-validation cleanup failed; mock objects may still be present.")
                # ------------------------

                return AgentResult(
                    code=clean_code,
                    description=result.description,
                    assumptions=result.assumptions,
                    passed=True,
                    confidence=logic.confidence,
                    issues=logic.issues,
                    retries_taken=retries_taken,
                    tests=result.tests,
                    clarification_needed=None,
                    pipeline_notes=pipeline_notes,
                )

            if attempt == self._max_retries - 1:
                break
            retries_taken += 1
            result, failure = self._safe_repair(query, result.code, logic.issues, retries_taken)
            if failure is not None:
                return failure
            assert result is not None
            clarification = self._clarification_result(result, retries_taken=retries_taken)
            if clarification is not None:
                return clarification

        self._log_stage("agent_run_fail")
        return AgentResult(
            code=result.code,
            description=result.description,
            assumptions=result.assumptions,
            passed=False,
            confidence=last_confidence,
            issues=last_issues,
            retries_taken=retries_taken,
            tests=result.tests,
            clarification_needed=None,
        )

    def _safe_repair(
        self, query: str, code: str, issues: list[str], retries_taken: int
    ) -> tuple[GenResult | None, AgentResult | None]:
        """
        Wraps generator.repair() so a provider outage or unrecoverable LLM
        failure returns a normal failed AgentResult instead of an unhandled
        exception killing the whole run. Returns (new_result, None) on
        success or (None, failure_result) on failure, exactly one is set.
        """
        try:
            return self._generator.repair(query, code, issues), None
        except LLMClientError as exc:
            return None, self._llm_failure_result(exc, retries_taken=retries_taken, partial_code=code)

    def _clarification_result(self, result: GenResult, retries_taken: int = 0) -> AgentResult | None:
        """Bug fix: clarification_needed must be honored after repair(), not just the initial generate()."""
        if not result.clarification_needed:
            return None
        self._log_stage("agent_run_clarification_needed")
        return AgentResult(
            code=result.code,
            description=result.description or "Clarification required before generation.",
            assumptions=result.assumptions,
            passed=False,
            confidence="low",
            issues=[result.clarification_needed],
            retries_taken=retries_taken,
            tests=result.tests,
            clarification_needed=result.clarification_needed,
        )

    def _llm_failure_result(
        self, exc: LLMClientError, retries_taken: int, partial_code: str = ""
    ) -> AgentResult:
        self._log_stage("agent_run_llm_failure")
        return AgentResult(
            code=partial_code,
            description="Generation failed: the LLM provider could not be reached or returned an unusable response.",
            assumptions=[],
            passed=False,
            confidence="low",
            issues=[f"LLM request failed: {exc}"],
            retries_taken=retries_taken,
            tests="",
            clarification_needed=None,
        )

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(UTC).isoformat())