import hashlib
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.models.discovery import GGUFVariant, RepoInspection
from kernl.models.download import DownloadError, ModelDownloader

runner = CliRunner()


def make_api(filename="model.gguf", size=10, sha256=None):
    lfs = {"size": size, "oid": sha256} if sha256 else None
    sibling = SimpleNamespace(rfilename=filename, size=size, lfs=lfs)
    info = SimpleNamespace(siblings=[sibling])
    return SimpleNamespace(model_info=lambda *args, **kwargs: info)


def test_download_requires_explicit_approval(tmp_path):
    downloader = ModelDownloader(api=object())
    with pytest.raises(PermissionError, match="approval"):
        downloader.download(
            "org/model", "model.gguf", destination=tmp_path, approve_download=False
        )


def test_download_rejects_path_traversal(tmp_path):
    downloader = ModelDownloader(api=object())
    with pytest.raises(ValueError, match="filename"):
        downloader.download(
            "org/model",
            "../secret.gguf",
            destination=tmp_path,
            approve_download=True,
        )


def test_download_verifies_size_and_sha256(tmp_path):
    payload = b"model-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "model.gguf"
    target.write_bytes(payload)

    def fake_download(**kwargs):
        return str(target)

    downloader = ModelDownloader(
        api=make_api(size=len(payload), sha256=sha), download_fn=fake_download
    )
    result = downloader.download(
        "org/model",
        "model.gguf",
        destination=tmp_path,
        approve_download=True,
    )

    assert result.verified is True
    assert result.size_bytes == len(payload)
    assert result.sha256 == sha
    assert result.path == target


def test_download_rejects_hash_mismatch(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"actual")
    downloader = ModelDownloader(
        api=make_api(size=6, sha256="0" * 64),
        download_fn=lambda **kwargs: str(target),
    )

    with pytest.raises(DownloadError, match="SHA-256"):
        downloader.download(
            "org/model",
            "model.gguf",
            destination=tmp_path,
            approve_download=True,
        )


def test_download_without_repository_hash_is_labelled_unverified(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"actual")
    downloader = ModelDownloader(
        api=make_api(size=6),
        download_fn=lambda **kwargs: str(target),
    )

    result = downloader.download(
        "org/model",
        "model.gguf",
        destination=tmp_path,
        approve_download=True,
    )

    assert result.verified is False


def test_cli_download_requires_approval():
    result = runner.invoke(app, ["download", "org/model"])

    assert result.exit_code != 0
    assert "approval" in result.output.lower()


def test_cli_download_selects_smallest_gguf_and_reports_verification(
    monkeypatch, tmp_path
):
    inspection = RepoInspection(
        repo_id="org/model",
        revision="abc123",
        license=None,
        gated=False,
        variants=(
            GGUFVariant("model.Q8_0.gguf", 2_000, "Q8_0"),
            GGUFVariant("model.Q4_K_M.gguf", 1_000, "Q4_K_M"),
            GGUFVariant("README.md", 100, None),
        ),
    )
    captured = {}

    class FakeDownloader:
        def download(self, repo_id, filename, **kwargs):
            captured.update(repo_id=repo_id, filename=filename, kwargs=kwargs)
            from kernl.models.download import DownloadResult

            path = tmp_path / filename
            path.write_bytes(b"x")
            return DownloadResult(repo_id, filename, path, 1, "0" * 64, True)

    monkeypatch.setattr(
        "kernl.cli.app.HuggingFaceDiscovery",
        lambda: SimpleNamespace(inspect=lambda *args, **kwargs: inspection),
    )
    monkeypatch.setattr("kernl.cli.app.ModelDownloader", FakeDownloader)

    result = runner.invoke(app, ["download", "org/model", "--approve"])

    assert result.exit_code == 0, result.output
    assert captured["filename"] == "model.Q4_K_M.gguf"
    assert captured["kwargs"]["approve_download"] is True
    assert "Downloaded" in result.output
    assert "Verified" in result.output
    assert "SHA-256" in result.output
