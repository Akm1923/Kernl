from types import SimpleNamespace

from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.core.types import HardwareProfile, Provenance
from kernl.models.discovery import GGUFVariant, ModelCandidate, RepoInspection

runner = CliRunner()


class StubDiscovery:
    def __init__(self, candidates=(), inspections=None):
        self.candidates = list(candidates)
        self.inspections = inspections or {}

    def search(self, query="", task=None, limit=20):
        return self.candidates

    def inspect(self, repo_id, *, revision=None, token=None):
        return self.inspections[repo_id]


def make_profile(available_bytes: int) -> HardwareProfile:
    return HardwareProfile(
        platform=Provenance(value="Windows 11"),
        architecture=Provenance(value="AMD64"),
        cpu_model=Provenance(value="Test CPU"),
        cpu_cores=Provenance(value=6),
        cpu_threads=Provenance(value=12),
        ram_total_bytes=Provenance(value=available_bytes),
        ram_available_bytes=Provenance(value=available_bytes),
        gpu_names=Provenance(),
        gpu_driver_versions=Provenance(),
        gpu_memory_bytes=Provenance(),
        storage_free_bytes=Provenance(value=100 * 1024**3),
        backends=Provenance(value=["cpu"]),
        scanned_at="now",
    )


def test_models_inspect_shows_repo_metadata_and_gguf_variants(monkeypatch):
    inspection = RepoInspection(
        repo_id="org/model",
        revision="abc123",
        license="apache-2.0",
        gated=False,
        variants=(
            GGUFVariant("model.Q4_K_M.gguf", 2 * 1024**3, "Q4_K_M"),
            GGUFVariant("model.f16.gguf", None, "F16"),
        ),
    )
    monkeypatch.setattr(
        "kernl.cli.models.HuggingFaceDiscovery",
        lambda: StubDiscovery(inspections={"org/model": inspection}),
    )

    result = runner.invoke(app, ["models", "inspect", "org/model"])

    assert result.exit_code == 0, result.output
    assert "org/model" in result.output
    assert "abc123" in result.output
    assert "apache-2.0" in result.output
    assert "Q4_K_M" in result.output
    assert "2.00 GiB" in result.output
    assert "UNKNOWN" in result.output


def test_models_recommend_uses_saved_profile_and_keeps_unknown_unknown(monkeypatch):
    candidates = [
        ModelCandidate(
            "small/model", "small", 10, 1, "text-generation", None, (), False
        ),
        ModelCandidate(
            "unknown/model", "unknown", 10, 1, "text-generation", None, (), False
        ),
    ]
    inspections = {
        "small/model": RepoInspection(
            "small/model",
            None,
            None,
            False,
            (GGUFVariant("s.Q4_K_M.gguf", 1024**3, "Q4_K_M"),),
        ),
        "unknown/model": RepoInspection(
            "unknown/model", None, None, False, (GGUFVariant("u.gguf", None, None),)
        ),
    }
    monkeypatch.setattr(
        "kernl.cli.models.HuggingFaceDiscovery",
        lambda: StubDiscovery(candidates=candidates, inspections=inspections),
    )
    monkeypatch.setattr(
        "kernl.cli.app.get_store",
        lambda: SimpleNamespace(latest_profile=lambda: make_profile(2 * 1024**3)),
    )

    result = runner.invoke(app, ["models", "recommend", "test"])

    assert result.exit_code == 0, result.output
    assert "small/model" in result.output
    assert "unknown/model" in result.output
    assert "YES" in result.output
    assert "UNKNOWN" in result.output
    assert "GGUF file size only" in result.output


def test_models_recommend_requires_memory_information(monkeypatch):
    monkeypatch.setattr(
        "kernl.cli.models.HuggingFaceDiscovery", lambda: StubDiscovery()
    )
    monkeypatch.setattr(
        "kernl.cli.app.get_store",
        lambda: SimpleNamespace(
            latest_profile=lambda: make_profile(1024).model_copy(
                update={"ram_available_bytes": Provenance()}
            )
        ),
    )

    result = runner.invoke(app, ["models", "recommend", "test"])

    assert result.exit_code != 0
    assert "available RAM" in result.output
