from __future__ import annotations

from dataclasses import dataclass

from kernl.core.types import HardwareProfile
from kernl.models.gguf import GGUFMetadata


@dataclass(frozen=True)
class HardwareMemoryPlan:
    """Estimated host/GPU memory requirements for one GGUF runtime plan."""

    model_bytes: int
    kv_cache_bytes: int | None
    gpu_layers: int
    cpu_required_bytes: int
    cpu_available_bytes: int | None
    cpu_fits: bool | None
    gpu_name: str | None
    gpu_required_bytes: int | None
    gpu_budget_bytes: int | None
    gpu_fits: bool | None
    overall_compatible: bool | None
    warnings: tuple[str, ...]


_GPU_MEMORY_FRACTION = 0.80
_CPU_RESIDENT_WEIGHT_FRACTION = 0.20
_RUNTIME_OVERHEAD_FRACTION = 0.10
_SAFETY_MARGIN_FRACTION = 0.15
_KV_BYTES_PER_ELEMENT = {
    "f16": 2.0,
    "f32": 4.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}


def _with_overhead(byte_count: int) -> int:
    runtime = int(byte_count * _RUNTIME_OVERHEAD_FRACTION)
    subtotal = byte_count + runtime
    return subtotal + int(subtotal * _SAFETY_MARGIN_FRACTION)


def _fits(required: int, capacity: int | None, *, complete: bool) -> bool | None:
    if capacity is None:
        return None
    if required > capacity:
        return False
    return True if complete else None


def plan_model_memory(
    metadata: GGUFMetadata,
    profile: HardwareProfile,
    *,
    gpu_layers: int = 0,
    gpu_index: int = 0,
    context_tokens: int | None = None,
    kv_cache_type: str = "f16",
    kv_cache_on_gpu: bool = True,
) -> HardwareMemoryPlan:
    """Estimate a GGUF model's host RAM and selected GPU VRAM requirements.

    The estimate uses the GGUF file size as a weight-size proxy. GPU layer
    placement is approximate: transformer blocks are assumed to be uniform,
    with 20% of weights retained on the CPU for non-block tensors. VRAM budget
    is capped at 80% of detected total memory to leave room for the desktop and
    other processes. These are planning estimates, not runtime measurements.
    """
    if metadata.file_size_bytes < 1:
        raise ValueError("GGUF file size must be positive")
    if gpu_layers < -1:
        raise ValueError("gpu_layers must be -1 (all) or non-negative")
    if gpu_index < 0:
        raise ValueError("GPU index cannot be negative")
    normalized_cache = kv_cache_type.strip().lower()
    if normalized_cache not in _KV_BYTES_PER_ELEMENT:
        raise ValueError("kv_cache_type must be f16, f32, q8_0, or q4_0")

    context_metadata_missing = metadata.context_length is None
    context = context_tokens if context_tokens is not None else metadata.context_length
    if context is None:
        context = 4096
    if context < 1:
        raise ValueError("context_tokens must be positive")
    if metadata.context_length is not None and context > metadata.context_length:
        raise ValueError("requested context exceeds the GGUF model context")

    layers = metadata.block_count
    if gpu_layers != 0 and layers is None:
        raise ValueError("GGUF block count is required for GPU layer offload")
    if layers is not None and layers < 1:
        raise ValueError("GGUF block count must be positive")
    resolved_gpu_layers = layers if gpu_layers == -1 else gpu_layers
    if layers is not None and resolved_gpu_layers > layers:
        raise ValueError("gpu_layers cannot exceed the GGUF block count")

    embedding = metadata.embedding_length
    kv_cache_bytes: int | None = None
    warnings: list[str] = []
    if context_metadata_missing:
        warnings.append(
            "GGUF context metadata is missing; the requested context cannot be checked against the model limit."
        )
    if layers is not None and embedding is not None and embedding > 0:
        kv_embedding = embedding
        heads = metadata.attention_head_count
        kv_heads = metadata.attention_head_count_kv
        if heads is not None and kv_heads is not None:
            if heads < 1 or kv_heads < 1 or kv_heads > heads:
                raise ValueError("GGUF attention head counts are invalid")
            kv_embedding = embedding * kv_heads // heads
        else:
            warnings.append(
                "KV head metadata is unavailable; full embedding width is used conservatively."
            )
        kv_cache_bytes = int(
            2
            * _KV_BYTES_PER_ELEMENT[normalized_cache]
            * layers
            * kv_embedding
            * context
        )
    else:
        warnings.append(
            "GGUF layer or embedding metadata is missing; KV-cache memory is unknown."
        )

    model_bytes = metadata.file_size_bytes
    gpu_name: str | None = None
    gpu_required: int | None = None
    gpu_budget: int | None = None
    gpu_fits: bool | None = None

    if resolved_gpu_layers > 0:
        names = profile.gpu_names.value or []
        if gpu_index >= len(names):
            raise ValueError(
                f"GPU index {gpu_index} is not present in the hardware profile"
            )
        gpu_name = names[gpu_index]
        if layers is not None:
            offload_fraction = resolved_gpu_layers / layers
            gpu_weight_bytes = int(
                model_bytes * (1 - _CPU_RESIDENT_WEIGHT_FRACTION) * offload_fraction
            )
        else:
            gpu_weight_bytes = 0
        gpu_components = gpu_weight_bytes
        if kv_cache_on_gpu and kv_cache_bytes is not None:
            gpu_components += kv_cache_bytes
        gpu_required = _with_overhead(gpu_components)

        memories = profile.gpu_memory_bytes.value
        total_vram = (
            memories[gpu_index] if memories and gpu_index < len(memories) else None
        )
        if total_vram is not None and total_vram < 0:
            raise ValueError("GPU memory cannot be negative")
        gpu_budget = (
            int(total_vram * _GPU_MEMORY_FRACTION) if total_vram is not None else None
        )
        gpu_complete = not kv_cache_on_gpu or kv_cache_bytes is not None
        gpu_fits = _fits(gpu_required, gpu_budget, complete=gpu_complete)
        if total_vram is None:
            warnings.append(
                f"VRAM is unknown for {gpu_name}; GPU fit cannot be confirmed."
            )
        else:
            warnings.append(
                f"{gpu_name} budget is capped at 80% of detected total VRAM; current free VRAM is not measured."
            )
        warnings.append(
            "GPU weight allocation is approximate and assumes uniform transformer block sizes."
        )

    cpu_weight_bytes = model_bytes
    if resolved_gpu_layers > 0 and layers is not None:
        gpu_weight_bytes = int(
            model_bytes
            * (1 - _CPU_RESIDENT_WEIGHT_FRACTION)
            * (resolved_gpu_layers / layers)
        )
        cpu_weight_bytes = model_bytes - gpu_weight_bytes
    cpu_components = cpu_weight_bytes
    if (resolved_gpu_layers == 0 or not kv_cache_on_gpu) and kv_cache_bytes is not None:
        cpu_components += kv_cache_bytes
    cpu_required = _with_overhead(cpu_components)
    cpu_available = profile.ram_available_bytes.value
    if cpu_available is not None and cpu_available < 0:
        raise ValueError("available system RAM cannot be negative")
    cpu_kv_required = resolved_gpu_layers == 0 or not kv_cache_on_gpu
    cpu_fits = _fits(
        cpu_required,
        cpu_available,
        complete=not cpu_kv_required or kv_cache_bytes is not None,
    )

    if cpu_available is None:
        warnings.append("Available system RAM is unknown; CPU fit cannot be confirmed.")
    if kv_cache_bytes is None:
        warnings.append(
            "Memory fit remains unknown until KV-cache metadata is available."
        )
    warnings.append(
        "GGUF file size and runtime overhead are estimates; actual residency depends on the runtime and workload."
    )
    warnings.append(
        "Prompt-processing compute buffers vary with batch and microbatch settings and are not modeled separately."
    )

    statuses = [cpu_fits]
    if resolved_gpu_layers > 0:
        statuses.append(gpu_fits)
    if False in statuses:
        overall = False
    elif all(status is True for status in statuses):
        overall = True
    else:
        overall = None

    return HardwareMemoryPlan(
        model_bytes=model_bytes,
        kv_cache_bytes=kv_cache_bytes,
        gpu_layers=resolved_gpu_layers,
        cpu_required_bytes=cpu_required,
        cpu_available_bytes=cpu_available,
        cpu_fits=cpu_fits,
        gpu_name=gpu_name,
        gpu_required_bytes=gpu_required,
        gpu_budget_bytes=gpu_budget,
        gpu_fits=gpu_fits,
        overall_compatible=overall,
        warnings=tuple(warnings),
    )
