from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import tempfile
from urllib.parse import urlparse

import httpx


class InstallationError(RuntimeError):
    """Raised when a runtime artifact cannot be safely installed."""


@dataclass(frozen=True)
class RuntimeArtifact:
    url: str
    sha256: str
    filename: str


class RuntimeInstaller:
    """Download and verify runtime artifacts; installation always needs approval."""

    def install(self, artifact: RuntimeArtifact, destination: Path, *, approve_install: bool = False) -> Path:
        if not approve_install:
            raise PermissionError("runtime installation requires explicit approval")
        self._validate_artifact(artifact)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / artifact.filename
        temporary = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        try:
            with httpx.stream("GET", artifact.url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        digest.update(chunk)
                        handle.write(chunk)
            actual = digest.hexdigest().lower()
            if actual != artifact.sha256.lower():
                temporary.unlink(missing_ok=True)
                raise InstallationError("runtime artifact SHA-256 verification failed")
            os.replace(temporary, target)
            return target
        except (OSError, httpx.HTTPError) as exc:
            temporary.unlink(missing_ok=True)
            raise InstallationError(f"runtime artifact download failed: {exc}") from exc

    @staticmethod
    def _validate_artifact(artifact: RuntimeArtifact) -> None:
        parsed = urlparse(artifact.url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise InstallationError("runtime artifact URL must use HTTPS")
        if len(artifact.sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in artifact.sha256):
            raise InstallationError("runtime artifact SHA-256 must be 64 hexadecimal characters")
        if not artifact.filename or Path(artifact.filename).name != artifact.filename:
            raise InstallationError("runtime artifact filename must not contain directories")
