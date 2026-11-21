from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, quote_plus

import httpx

REMOTE_PROVIDERS = ("gemini", "groq")
ALL_PROVIDERS = ("local", *REMOTE_PROVIDERS)

ValidationStatus = Literal["valid", "invalid", "unreachable", "unexpected"]


@dataclass(frozen=True)
class ProviderValidation:
    """Observed result of one live credential check."""

    provider: str
    status: ValidationStatus
    detail: str
    http_status: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "valid"


@dataclass(frozen=True)
class _Endpoint:
    url: str
    list_key: str
    header_name: str

    def header_value(self, api_key: str) -> str:
        if self.header_name == "Authorization":
            return f"Bearer {api_key}"
        return api_key


_ENDPOINTS = {
    "gemini": _Endpoint(
        url="https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
        list_key="models",
        header_name="x-goog-api-key",
    ),
    "groq": _Endpoint(
        url="https://api.groq.com/openai/v1/models",
        list_key="data",
        header_name="Authorization",
    ),
}


def validate_provider(
    provider: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> ProviderValidation:
    """Validate a stored provider key with one live, read-only API request.

    The key is sent in a header, never in the URL, and is redacted from every
    returned detail string. No model is executed and no weight is downloaded.
    """
    normalized = provider.strip().lower()
    if normalized not in _ENDPOINTS:
        raise ValueError(f"unsupported provider: {provider}")
    secret = api_key.strip()
    if not secret:
        raise ValueError("API key cannot be empty")
    endpoint = _ENDPOINTS[normalized]
    headers = {endpoint.header_name: endpoint.header_value(secret)}

    try:
        if client is not None:
            response = client.get(endpoint.url, headers=headers)
        else:
            with httpx.Client(timeout=timeout, follow_redirects=False) as owned:
                response = owned.get(endpoint.url, headers=headers)
    except httpx.TimeoutException:
        return ProviderValidation(normalized, "unreachable", "request timed out")
    except httpx.TransportError as exc:
        return ProviderValidation(
            normalized,
            "unreachable",
            f"network error: {redact_secret(str(exc), secret)}",
        )

    status = response.status_code
    if 200 <= status < 300:
        try:
            payload = response.json()
        except ValueError:
            return ProviderValidation(
                normalized,
                "unexpected",
                "credential was accepted but the response was not JSON",
                status,
            )
        if not isinstance(payload, dict) or not isinstance(
            payload.get(endpoint.list_key), list
        ):
            return ProviderValidation(
                normalized,
                "unexpected",
                "credential was accepted but the response shape was not recognized",
                status,
            )
        count = len(payload[endpoint.list_key])
        return ProviderValidation(
            normalized,
            "valid",
            f"credential accepted; {count} model(s) visible",
            status,
        )

    error_detail = _response_error(response, secret)
    suffix = f": {error_detail}" if error_detail else ""
    if status in (401, 403):
        return ProviderValidation(
            normalized,
            "invalid",
            f"credential rejected (HTTP {status}){suffix}",
            status,
        )
    if status in (408, 429) or status >= 500:
        return ProviderValidation(
            normalized,
            "unreachable",
            f"provider unavailable (HTTP {status}){suffix}",
            status,
        )
    return ProviderValidation(
        normalized, "invalid", f"request rejected (HTTP {status}){suffix}", status
    )


def _response_error(response: httpx.Response, secret: str) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
    elif isinstance(error, str):
        message = error
    else:
        message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    return redact_secret(_truncate(message), secret)


def redact_secret(text: str, secret: str) -> str:
    redacted = text
    for variant in (secret, quote(secret), quote_plus(secret)):
        if variant:
            redacted = redacted.replace(variant, "***redacted***")
    return _truncate(redacted)


def _truncate(text: str, limit: int = 200) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."
