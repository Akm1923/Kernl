from __future__ import annotations

from dataclasses import dataclass

from kernl.models.discovery import ModelCandidate


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reasons: tuple[str, ...]
    score: float


def assess_model(
    model: ModelCandidate,
    *,
    available_ram_bytes: int,
    required_ram_bytes: int | None = None,
) -> CompatibilityResult:
    """Conservative compatibility assessment; unknown metadata never becomes a false positive."""
    reasons: list[str] = []
    if required_ram_bytes is not None and required_ram_bytes > available_ram_bytes:
        reasons.append("estimated model memory exceeds available RAM")
    if model.gated is True:
        reasons.append("model is gated and requires access approval")
    if not model.repo_id:
        reasons.append("model repository id is missing")
    compatible = not reasons
    score = 1.0 if compatible else max(0.0, 1.0 - 0.35 * len(reasons))
    return CompatibilityResult(
        compatible=compatible, reasons=tuple(reasons), score=score
    )
