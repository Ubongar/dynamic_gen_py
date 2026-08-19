from __future__ import annotations

from unittest.mock import Mock

from codegen_validator.llm_client import LLMClient
from codegen_validator.models import GenResult
from codegen_validator.validator import Validator


def test_static_check_accepts_valid_python() -> None:
    # Use a proper Mock spec instead of a custom dummy class to satisfy typing
    dummy_llm = Mock(spec=LLMClient)
    validator = Validator(llm_client=dummy_llm)
    
    result = validator.static_check("def add(a, b):\n    return a + b\n")
    
    assert result.passed is True
    assert result.issues == []


def test_gen_result_defaults() -> None:
    # Verify all defaults match the current GenResult model in models.py
    result = GenResult(code="print('ok')", description="prints ok")
    
    assert result.assumptions == []
    assert result.tests == ""
    assert result.clarification_needed is None


def test_llm_client_uses_openai_compatible_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "olori-image")

    client = LLMClient()

    assert client._model == "olori-image"
    assert str(client._client.base_url).rstrip("/") == "http://example.com/v1"