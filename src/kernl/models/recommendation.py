from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kernl.models.compatibility import CompatibilityResult, assess_model
from kernl.models.discovery import ModelCandidate


@dataclass(frozen=True)
class Recommendation:
    model: ModelCandidate
    compatibility: CompatibilityResult
    reason: str


def recommend(
    models: list[ModelCandidate],
    *,
    available_ram_bytes: int,
    task: str | None = None,
    required_ram_bytes_by_repo: Mapping[str, int] | None = None,
) -> list[Recommendation]:
    results: list[Recommendation] = []
    for model in models:
        if task and model.pipeline_tag and model.pipeline_tag != task:
            continue
        required_bytes = (required_ram_bytes_by_repo or {}).get(model.repo_id)
        compatibility = assess_model(
            model,
            available_ram_bytes=available_ram_bytes,
            required_ram_bytes=required_bytes,
        )
        reason = (
            "; ".join(compatibility.reasons) or "estimated memory fits available RAM"
        )
        results.append(
            Recommendation(model=model, compatibility=compatibility, reason=reason)
        )
    return sorted(
        results,
        key=lambda item: (
            {True: 2, None: 1, False: 0}[item.compatibility.compatible],
            item.compatibility.score,
            item.model.downloads,
        ),
        reverse=True,
    )
