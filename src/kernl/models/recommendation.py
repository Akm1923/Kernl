from __future__ import annotations
from dataclasses import dataclass

from kernl.models.compatibility import CompatibilityResult, assess_model
from kernl.models.discovery import ModelCandidate
from kernl.models.memory import estimate_memory

@dataclass(frozen=True)
class Recommendation:
    model: ModelCandidate
    compatibility: CompatibilityResult
    reason: str


def recommend(models: list[ModelCandidate], *, available_ram_bytes: int, task: str | None = None) -> list[Recommendation]:
    results: list[Recommendation] = []
    for model in models:
        if task and model.pipeline_tag and model.pipeline_tag != task:
            continue
        compatibility = assess_model(model, available_ram_bytes=available_ram_bytes)
        reason = "metadata is incomplete; inspect GGUF before execution" if compatibility.compatible else "; ".join(compatibility.reasons)
        results.append(Recommendation(model=model, compatibility=compatibility, reason=reason))
    return sorted(results, key=lambda item: (item.compatibility.compatible, item.compatibility.score, item.model.downloads), reverse=True)
