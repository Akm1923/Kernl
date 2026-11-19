from __future__ import annotations

from dataclasses import dataclass

from kernl.models.discovery import ModelCandidate


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool | None
    reasons: tuple[str, ...]
    score: float


def assess_model(
    model: ModelCandidate,
    *,
    available_ram_bytes: int,
    required_ram_bytes: int | None = None,
) -> CompatibilityResult:
    """Conservative compatibility assessment; unknown metadata never becomes a false positive."""
    if available_ram_bytes < 0:
        raise ValueError("available_ram_bytes cannot be negative")
    if required_ram_bytes is not None and required_ram_bytes < 0:
        raise ValueError("required_ram_bytes cannot be negative")

    reasons: list[str] = []
    compatible: bool | None = None
    if required_ram_bytes is None:
        reasons.append("memory requirements are unknown; inspect GGUF metadata")
    elif required_ram_bytes > available_ram_bytes:
        reasons.append("estimated model memory exceeds available RAM")
        compatible = False
    else:
        compatible = True

    if model.gated is True:
        reasons.append("model is gated and requires access approval")
        compatible = False
    if not model.repo_id:
        reasons.append("model repository id is missing")
        compatible = False

    score = {True: 1.0, None: 0.5, False: 0.0}[compatible]
    return CompatibilityResult(
        compatible=compatible, reasons=tuple(reasons), score=score
    )
