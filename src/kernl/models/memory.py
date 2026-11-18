from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum


class Quantization(StrEnum):
    F32 = "F32"
    F16 = "F16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q4_K_M = "Q4_K_M"
    Q4_0 = "Q4_0"
    UNKNOWN = "UNKNOWN"


# Approximate bytes per parameter. These are estimates, not measurements.
_BYTES_PER_PARAMETER: dict[Quantization, float] = {
    Quantization.F32: 4.0,
    Quantization.F16: 2.0,
    Quantization.Q8_0: 1.0625,
    Quantization.Q6_K: 0.8125,
    Quantization.Q5_K_M: 0.6875,
    Quantization.Q4_K_M: 0.5625,
    Quantization.Q4_0: 0.5625,
}


@dataclass(frozen=True)
class MemoryEstimate:
    model_bytes: int | None
    kv_cache_bytes: int | None
    runtime_overhead_bytes: int | None
    total_bytes: int | None
    available_bytes: int | None
    safety_margin_bytes: int | None
    estimated: bool
    warnings: tuple[str, ...] = ()

    @property
    def fits(self) -> bool | None:
        if self.total_bytes is None or self.available_bytes is None:
            return None
        return self.total_bytes <= self.available_bytes


def parse_quantization(value: str | None) -> Quantization:
    if not value:
        return Quantization.UNKNOWN
    normalized = value.strip().upper().replace("-", "_")
    aliases = {"Q4_K_S": Quantization.Q4_K_M, "Q5_K_S": Quantization.Q5_K_M}
    if normalized in aliases:
        return aliases[normalized]
    try:
        return Quantization(normalized)
    except ValueError:
        return Quantization.UNKNOWN


def estimate_memory(
    *,
    parameter_count: int | None,
    quantization: str | Quantization | None,
    context_tokens: int = 4096,
    layers: int | None = None,
    embedding_length: int | None = None,
    available_bytes: int | None = None,
    safety_fraction: float = 0.15,
) -> MemoryEstimate:
    """Estimate runtime memory and label all unknown inputs explicitly.

    The result is intentionally conservative. It is not a benchmark and must not
    be presented as measured RAM/VRAM consumption.
    """
    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    if not 0 <= safety_fraction < 1:
        raise ValueError("safety_fraction must be between 0 and 1")

    quant = parse_quantization(quantization) if isinstance(quantization, str) else (quantization or Quantization.UNKNOWN)
    warnings: list[str] = []
    if parameter_count is None or parameter_count < 1:
        warnings.append("parameter count is unknown; model memory cannot be estimated")
    if quant == Quantization.UNKNOWN:
        warnings.append("quantization is unknown; model memory cannot be estimated")

    model_bytes: int | None = None
    if parameter_count and parameter_count > 0 and quant in _BYTES_PER_PARAMETER:
        model_bytes = int(parameter_count * _BYTES_PER_PARAMETER[quant])

    kv_cache_bytes: int | None = None
    if layers and embedding_length:
        # Simplified K/V fp16 estimate: 2 tensors × 2 bytes × layers × hidden × tokens.
        kv_cache_bytes = int(2 * 2 * layers * embedding_length * context_tokens)
    else:
        warnings.append("layers or embedding length is unknown; KV cache is excluded")

    known_components = [value for value in (model_bytes, kv_cache_bytes) if value is not None]
    runtime_overhead_bytes: int | None = None
    total_bytes: int | None = None
    safety_margin_bytes: int | None = None
    if known_components:
        runtime_overhead_bytes = int(sum(known_components) * 0.10)
        subtotal = sum(known_components) + runtime_overhead_bytes
        safety_margin_bytes = int(subtotal * safety_fraction)
        total_bytes = subtotal + safety_margin_bytes
    else:
        warnings.append("no known memory components; total estimate is unknown")

    return MemoryEstimate(
        model_bytes=model_bytes,
        kv_cache_bytes=kv_cache_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        safety_margin_bytes=safety_margin_bytes,
        estimated=True,
        warnings=tuple(warnings),
    )
