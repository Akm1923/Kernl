from pathlib import Path

import pytest

from kernl.runtime.installer import InstallationError, RuntimeArtifact, RuntimeInstaller


def test_install_requires_approval(tmp_path: Path):
    artifact = RuntimeArtifact(
        "https://example.com/runtime.exe", "0" * 64, "runtime.exe"
    )
    with pytest.raises(PermissionError):
        RuntimeInstaller().install(artifact, tmp_path)


def test_artifact_validation_is_strict(tmp_path: Path):
    artifact = RuntimeArtifact(
        "http://example.com/runtime.exe", "0" * 64, "runtime.exe"
    )
    with pytest.raises(InstallationError):
        RuntimeInstaller().install(artifact, tmp_path, approve_install=True)


def test_filename_traversal_rejected(tmp_path: Path):
    artifact = RuntimeArtifact(
        "https://example.com/runtime.exe", "0" * 64, "../runtime.exe"
    )
    with pytest.raises(InstallationError):
        RuntimeInstaller().install(artifact, tmp_path, approve_install=True)
