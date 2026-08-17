from __future__ import annotations

from codegen_validator.models import GenResult
from codegen_validator.validator import Validator


class _DummyLLMClient:
    def review_logic(self, prompt: str) -> object:  # pragma: no cover - not used in this test
        raise NotImplementedError


def test_static_check_accepts_valid_python() -> None:
    validator = Validator(llm_client=_DummyLLMClient())  # type: ignore[arg-type]
    result = validator.static_check("def add(a, b):\n    return a + b\n")
    assert result.passed is True


def test_gen_result_defaults() -> None:
    result = GenResult(code="print('ok')", description="prints ok")
    assert result.assumptions == []

