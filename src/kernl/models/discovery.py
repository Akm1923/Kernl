from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from huggingface_hub import HfApi

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

class HuggingFaceDiscovery:
    def __init__(self, api: HfApi | None = None) -> None:
        self.api = api or HfApi()

    def search(self, query: str = "", task: str | None = None, limit: int = 20) -> list[ModelCandidate]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            models = self.api.list_models(search=query or None, filter=task or None, limit=limit, full=True)
            return [self._map(model) for model in models]
        except Exception as exc:
            raise DiscoveryError(f"Hugging Face discovery failed: {exc}") from exc

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
