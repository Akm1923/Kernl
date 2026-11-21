from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from huggingface_hub import HfApi

_QUANTIZATION_RE = re.compile(
    r"(?i)\b(iq[1-9][a-z0-9_]*|q[1-8][a-z0-9_]*|bf16|fp16|fp32|f16|f32)\b"
)


@dataclass(frozen=True)
class ModelCandidate:
    repo_id: str
    author: str | None
    downloads: int
    likes: int
    pipeline_tag: str | None
    library_name: str | None
    tags: tuple[str, ...]
    gated: bool | None


class DiscoveryError(RuntimeError):
    """Raised when Hugging Face discovery cannot complete."""


@dataclass(frozen=True)
class GGUFVariant:
    filename: str
    size_bytes: int | None
    quantization: str | None


@dataclass(frozen=True)
class RepoInspection:
    repo_id: str
    revision: str | None
    license: str | None
    gated: bool | None
    variants: tuple[GGUFVariant, ...]


class HuggingFaceDiscovery:
    def __init__(self, api: HfApi | None = None) -> None:
        self.api = api or HfApi()

    def search(
        self, query: str = "", task: str | None = None, limit: int = 20
    ) -> list[ModelCandidate]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            models = self.api.list_models(
                search=query or None, filter=task or None, limit=limit, full=True
            )
            return [self._map(model) for model in models]
        except Exception as exc:
            raise DiscoveryError(f"Hugging Face discovery failed: {exc}") from exc

    def inspect(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
    ) -> RepoInspection:
        """Read GGUF file metadata without downloading model weights."""
        if not repo_id.strip():
            raise ValueError("repo_id cannot be empty")
        try:
            info = self.api.model_info(
                repo_id,
                revision=revision,
                files_metadata=True,
                token=token,
            )
        except Exception as exc:
            raise DiscoveryError(f"Hugging Face inspection failed: {exc}") from exc

        card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        license_value = (
            card.get("license")
            if isinstance(card, dict)
            else getattr(card, "license", None)
        )
        variants = [
            GGUFVariant(
                filename=name,
                size_bytes=self._file_size(file),
                quantization=self._quantization(name),
            )
            for file in (getattr(info, "siblings", None) or [])
            if (name := self._filename(file)).lower().endswith(".gguf")
        ]
        variants.sort(
            key=lambda variant: (
                variant.size_bytes is None,
                variant.size_bytes if variant.size_bytes is not None else 0,
                variant.filename,
            )
        )
        gated = getattr(info, "gated", None)
        revision_value = getattr(info, "sha", None) or revision
        return RepoInspection(
            repo_id=repo_id,
            revision=str(revision_value) if revision_value else None,
            license=str(license_value) if license_value is not None else None,
            gated=bool(gated) if gated is not None else None,
            variants=tuple(variants),
        )

    @staticmethod
    def _filename(file: Any) -> str:
        """Support both the live RepoSibling field and older dict-style shapes."""
        return str(getattr(file, "rfilename", None) or getattr(file, "path", "") or "")

    @staticmethod
    def _quantization(filename: str) -> str | None:
        match = _QUANTIZATION_RE.search(PurePosixPath(filename).name)
        return match.group(0).upper() if match else None

    @staticmethod
    def _file_size(file: Any) -> int | None:
        size = getattr(file, "size", None)
        lfs = getattr(file, "lfs", None)
        if size is None:
            size = (
                lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
            )
        if isinstance(size, bool) or not isinstance(size, (int, float)) or size < 0:
            return None
        return int(size)

    @staticmethod
    def _map(model: Any) -> ModelCandidate:
        tags = getattr(model, "tags", None) or []
        return ModelCandidate(
            repo_id=str(model.id),
            author=getattr(model, "author", None),
            downloads=int(getattr(model, "downloads", 0) or 0),
            likes=int(getattr(model, "likes", 0) or 0),
            pipeline_tag=getattr(model, "pipeline_tag", None),
            library_name=getattr(model, "library_name", None),
            tags=tuple(str(tag) for tag in tags),
            gated=getattr(model, "gated", None),
        )
