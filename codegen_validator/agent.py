from __future__ import annotations

import logging
from datetime import datetime, timezone

from .generator import Generator
from .models import AgentResult, GenResult
from .validator import Validator


LOGGER = logging.getLogger(__name__)


class CodeAgent:
    def __init__(self, generator: Generator, validator: Validator, max_retries: int = 3) -> None:
        self._generator = generator
        self._validator = validator
        self._max_retries = max_retries

    def run(self, query: str) -> AgentResult:
        self._log_stage("agent_run_start")
        result = self._generator.generate(query)

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
                result = self._generator.repair(query, result.code, [static.error or "Syntax validation failed."])
                clarification = self._clarification_result(result, retries_taken=retries_taken)
                if clarification is not None:
                    return clarification
                continue

            # 2. Execution Check (runs code + tests together, real dynamic verification)
            execution = self._validator.execution_check(result.code, result.tests)
            if not execution.passed:
                last_issues = execution.issues
                last_confidence = execution.confidence
                if attempt == self._max_retries - 1:
                    break
                retries_taken += 1
                result = self._generator.repair(query, result.code, execution.issues)
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
                return AgentResult(
                    code=result.code,
                    description=result.description,
                    assumptions=result.assumptions,
                    passed=True,
                    confidence=logic.confidence,
                    issues=logic.issues,
                    retries_taken=retries_taken,
                    tests=result.tests,
                    clarification_needed=None,
                )

            if attempt == self._max_retries - 1:
                break
            retries_taken += 1
            result = self._generator.repair(query, result.code, logic.issues)
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

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())