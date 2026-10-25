import pytest
from pydantic import ValidationError

from llm_gateway.models import AskRequest


def test_valid_request():
    request = AskRequest(question="What is the capital of France?")
    assert request.question == "What is the capital of France?"
    assert request.max_output_tokens == 512
    assert request.model == "gpt-4.1-mini"


def test_invalid_token_limit():
    with pytest.raises(ValidationError):
        AskRequest(question="What is the capital of France?", max_output_tokens=3000)


def test_unknown_field():
    with pytest.raises(ValidationError):
        AskRequest(
            question="What is the capital of France?",
            unknown_field="This should not be allowed",
        )


def test_whitespace_question():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")  # Question with only whitespace


def test_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")  # Empty question


def test_negative_token_limit():
    with pytest.raises(ValidationError):
        AskRequest(question="What is the capital of France?", max_output_tokens=-1)


def test_invalid_assignment():
    request = AskRequest(question="What is the capital of France?")
    with pytest.raises(ValidationError):
        request.max_output_tokens = 3000  # Invalid assignment
