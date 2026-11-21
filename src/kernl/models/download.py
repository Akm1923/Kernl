from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


class DownloadError(RuntimeError):
    """Raised when a model artifact cannot be downloaded or verified."""


@dataclass(frozen=True)
class ArtifactMetadata:
    filename: str
    size_bytes: int | None
    sha256: str | None


@dataclass(frozen=True)
class DownloadResult:
    repo_id: str
    filename: str
    path: Path
    size_bytes: int
    sha256: str
    verified: bool


class ModelDownloader:
    """Download one Hugging Face artifact after explicit approval.

    Artifacts are verified against repository metadata when Hugging Face exposes
    a SHA-256 LFS oid. No download happens without approval.
    """

    def __init__(
        self,
        api: HfApi | None = None,
        download_fn: Callable[..., str] | None = None,
    ) -> None:
        self.api = api or HfApi()
        self._download_fn = download_fn or hf_hub_download

    def artifact(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        token: str | None = None,
    ) -> ArtifactMetadata:
        _validate_repo_id(repo_id)
        _validate_filename(filename)
        try:
            info = self.api.model_info(
                repo_id,
                revision=revision,
                files_metadata=True,
                token=token,
            )
        except Exception as exc:
            raise DownloadError(f"Hugging Face metadata lookup failed: {exc}") from exc

        for sibling in getattr(info, "siblings", None) or []:
            name = _filename(sibling)
            if name != filename:
                continue
            return ArtifactMetadata(
                filename=name,
                size_bytes=_file_size(sibling),
                sha256=_lfs_sha256(sibling),
            )
        raise DownloadError(f"artifact not found in repository: {filename}")

    def download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        destination: Path | None = None,
        approve_download: bool = False,
        token: str | None = None,
    ) -> DownloadResult:
        if not approve_download:
            raise PermissionError("model download requires explicit approval")
        metadata = self.artifact(repo_id, filename, revision=revision, token=token)
        target_dir = destination or Path.home() / ".kernl" / "models"
        try:
            downloaded = self._download_fn(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
                local_dir=target_dir,
            )
        except Exception as exc:
            raise DownloadError(f"model artifact download failed: {exc}") from exc

        path = Path(downloaded)
        if not path.is_file():
            raise DownloadError(f"downloaded artifact not found: {path}")
        actual_size = path.stat().st_size
        if metadata.size_bytes is not None and actual_size != metadata.size_bytes:
            raise DownloadError(
                f"artifact size mismatch: expected {metadata.size_bytes}, got {actual_size}"
            )
        actual_sha256 = _sha256(path)
        verified = False
        if metadata.sha256 is not None:
            if actual_sha256.lower() != metadata.sha256.lower():
                raise DownloadError("artifact SHA-256 verification failed")
            verified = True
        return DownloadResult(
            repo_id=repo_id,
            filename=filename,
            path=path,
            size_bytes=actual_size,
            sha256=actual_sha256,
            verified=verified,
        )


def _validate_repo_id(repo_id: str) -> None:
    value = repo_id.strip()
    if not value or value.startswith("/") or "\\" in value or ".." in value:
        raise ValueError("repo_id must be a valid Hugging Face repository id")


def _validate_filename(filename: str) -> None:
    path = PurePosixPath(filename)
    if not filename or path.is_absolute() or ".." in path.parts or "\\" in filename:
        raise ValueError("filename must be a relative repository path")


def _filename(file: Any) -> str:
    return str(getattr(file, "rfilename", None) or getattr(file, "path", "") or "")


def _file_size(file: Any) -> int | None:
    size = getattr(file, "size", None)
    lfs = getattr(file, "lfs", None)
    if size is None:
        size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if isinstance(size, bool) or not isinstance(size, (int, float)) or size < 0:
        return None
    return int(size)


def _lfs_sha256(file: Any) -> str | None:
    lfs = getattr(file, "lfs", None)
    oid = lfs.get("oid") if isinstance(lfs, dict) else getattr(lfs, "oid", None)
    if not isinstance(oid, str) or len(oid) != 64:
        return None
    if any(char not in "0123456789abcdefABCDEF" for char in oid):
        return None
    return oid.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
