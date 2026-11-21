from types import SimpleNamespace

from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.providers import ProviderValidation

runner = CliRunner()
SECRET = "sk-cli-secret-value"


def stub_store(provider):
    return SimpleNamespace(get=lambda key: provider)


def test_provider_test_reports_live_validation_for_selected_provider(monkeypatch):
    calls = {}

    def fake_validate(provider, secret, *, timeout):
        calls.update(provider=provider, secret=secret, timeout=timeout)
        return ProviderValidation(
            provider, "valid", "credential accepted; 2 model(s) visible", 200
        )

    monkeypatch.setattr("kernl.cli.app.get_store", lambda: stub_store("gemini"))
    monkeypatch.setattr("kernl.cli.app.get_secret", lambda provider: SECRET)
    monkeypatch.setattr("kernl.cli.app.validate_provider", fake_validate)

    result = runner.invoke(app, ["provider", "test"])

    assert result.exit_code == 0, result.output
    assert "Verified" in result.output
    assert "HTTP 200" in result.output
    assert "2 model(s)" in result.output
    assert "live provider API response" in result.output
    assert SECRET not in result.output
    assert calls["provider"] == "gemini"
    assert calls["secret"] == SECRET


def test_provider_test_can_validate_a_provider_other_than_selected(monkeypatch):
    seen = []

    def fake_validate(provider, secret, *, timeout):
        seen.append(provider)
        return ProviderValidation(provider, "valid", "credential accepted", 200)

    monkeypatch.setattr("kernl.cli.app.get_store", lambda: stub_store("local"))
    monkeypatch.setattr("kernl.cli.app.get_secret", lambda provider: SECRET)
    monkeypatch.setattr("kernl.cli.app.validate_provider", fake_validate)

    result = runner.invoke(app, ["provider", "test", "--provider", "groq"])

    assert result.exit_code == 0, result.output
    assert seen == ["groq"]


def test_provider_test_requires_a_stored_credential(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("validator must not be called without a credential")

    monkeypatch.setattr("kernl.cli.app.get_store", lambda: stub_store("groq"))
    monkeypatch.setattr("kernl.cli.app.get_secret", lambda provider: None)
    monkeypatch.setattr("kernl.cli.app.validate_provider", fail)

    result = runner.invoke(app, ["provider", "test"])

    assert result.exit_code == 1
    assert "No credential" in result.output


def test_provider_test_redacts_secret_from_rejection_detail(monkeypatch):
    def fake_validate(provider, secret, *, timeout):
        return ProviderValidation(
            provider, "invalid", f"credential rejected: {secret}", 401
        )

    monkeypatch.setattr("kernl.cli.app.get_store", lambda: stub_store("gemini"))
    monkeypatch.setattr("kernl.cli.app.get_secret", lambda provider: SECRET)
    monkeypatch.setattr("kernl.cli.app.validate_provider", fake_validate)

    result = runner.invoke(app, ["provider", "test"])

    assert result.exit_code == 1
    assert "Rejected" in result.output
    assert "***redacted***" in result.output
    assert SECRET not in result.output


def test_provider_test_local_provider_makes_no_network_request(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("local provider must not call the validator")

    monkeypatch.setattr("kernl.cli.app.get_store", lambda: stub_store("local"))
    monkeypatch.setattr("kernl.cli.app.validate_provider", fail)

    result = runner.invoke(app, ["provider", "test"])

    assert result.exit_code == 0, result.output
    assert "Local-only provider is ready" in result.output
