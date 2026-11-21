import httpx
import pytest

from kernl.providers import validate_provider

SECRET = "sk-test-secret-value"


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gemini_validation_sends_key_in_header_and_reports_observed_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        return httpx.Response(200, json={"models": [{"name": "models/gemini-test"}]})

    result = validate_provider("gemini", SECRET, client=client_for(handler))

    assert result.status == "valid"
    assert result.ok is True
    assert result.http_status == 200
    assert "1 model(s) visible" in result.detail
    assert SECRET not in str(result)
    assert "generativelanguage.googleapis.com" in seen["url"]
    assert SECRET not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == SECRET


def test_groq_validation_uses_bearer_header_and_read_only_models_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        return httpx.Response(200, json={"data": [{"id": "test-model"}]})

    result = validate_provider("groq", SECRET, client=client_for(handler))

    assert result.status == "valid"
    assert "api.groq.com/openai/v1/models" in seen["url"]
    assert seen["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_rejected_credential_is_redacted_and_exit_worthy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": f"API key not valid: {SECRET}"}},
        )

    result = validate_provider("gemini", SECRET, client=client_for(handler))

    assert result.status == "invalid"
    assert result.ok is False
    assert result.http_status == 401
    assert "***redacted***" in result.detail
    assert SECRET not in str(result)


def test_timeout_is_reported_as_unreachable_not_invalid():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    result = validate_provider("groq", SECRET, client=client_for(handler))

    assert result.status == "unreachable"
    assert "timed out" in result.detail


def test_server_error_is_reported_as_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "try later"}})

    result = validate_provider("gemini", SECRET, client=client_for(handler))

    assert result.status == "unreachable"
    assert result.http_status == 503


def test_unexpected_success_shape_is_not_claimed_as_valid():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    result = validate_provider("groq", SECRET, client=client_for(handler))

    assert result.status == "unexpected"
    assert "response shape" in result.detail


def test_unsupported_provider_is_rejected():
    with pytest.raises(ValueError, match="unsupported provider"):
        validate_provider("openai", SECRET)
