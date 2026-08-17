from __future__ import annotations

import logging
from datetime import datetime, timezone

from .generator import Generator
from .models import AgentResult
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
        if result.clarification_needed:
            self._log_stage("agent_run_clarification_needed")
            return AgentResult(
                code="",
                description="Clarification required before generation.",
                assumptions=[],
                passed=False,
                confidence="low",
                issues=[result.clarification_needed],
                retries_taken=0,
                clarification_needed=result.clarification_needed
            )
        retries_taken = 0
        last_issues = []
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
                continue

            # 2. Execution Check (The Sandbox)
            execution = self._validator.execution_check(result.code)
            if not execution.passed:
                last_issues = execution.issues
                last_confidence = execution.confidence
                if attempt == self._max_retries - 1:
                    break
                retries_taken += 1
                result = self._generator.repair(query, result.code, execution.issues)
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
                    clarification_needed=None,
                )

            if attempt == self._max_retries - 1:
                break
            retries_taken += 1
            result = self._generator.repair(query, result.code, logic.issues)

        self._log_stage("agent_run_fail")
        return AgentResult(
            code=result.code,
            description=result.description,
            assumptions=result.assumptions,
            passed=False,
            confidence=last_confidence,
            issues=last_issues,
            retries_taken=retries_taken,
            clarification_needed=None,
        )

    @staticmethod
    def _log_stage(stage: str) -> None:
        LOGGER.info("%s %s", stage, datetime.now(timezone.utc).isoformat())