from pathlib import Path

import pytest

from kernl.core.types import HardwareProfile, Provenance
from kernl.models.gguf import GGUFMetadata
from kernl.models.planner import plan_model_memory, recommend_gpu_layers


def make_profile(*, ram_bytes=16 * 1024**3, gpu_memory=(4 * 1024**3,)):
    return HardwareProfile(
        platform=Provenance(value="Windows 11"),
        architecture=Provenance(value="AMD64"),
        cpu_model=Provenance(value="Test CPU"),
        cpu_cores=Provenance(value=6),
        cpu_threads=Provenance(value=12),
        ram_total_bytes=Provenance(value=ram_bytes),
        ram_available_bytes=Provenance(value=ram_bytes),
        gpu_names=Provenance(value=[f"GPU {i}" for i in range(len(gpu_memory))]),
        gpu_driver_versions=Provenance(),
        gpu_memory_bytes=Provenance(value=list(gpu_memory)),
        storage_free_bytes=Provenance(value=100 * 1024**3),
        backends=Provenance(value=["cpu", "cuda"] if gpu_memory else ["cpu"]),
        scanned_at="now",
    )


def make_metadata(
    *,
    file_size=4 * 1024**3,
    layers=20,
    context=8192,
    embedding=4096,
    kv_heads=8,
):
    return GGUFMetadata(
        path=Path("model.gguf"),
        version=3,
        tensor_count=100,
        metadata_count=10,
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=context,
        embedding_length=embedding,
        block_count=layers,
        attention_head_count=32,
        attention_head_count_kv=kv_heads,
        quantization_version=2,
        file_size_bytes=file_size,
        metadata={},
    )


def auto(metadata, profile, **kwargs):
    return recommend_gpu_layers(metadata, profile, **kwargs)


def test_cpu_only_hardware_recommends_zero_layers():
    result = auto(make_metadata(), make_profile(gpu_memory=()))

    assert result.status == "fit"
    assert result.gpu_layers == 0
    assert result.gpu_name is None
    assert result.gpu_budget_bytes is None
    assert "No GPU" in result.reason
    assert result.warnings


def test_single_gpu_recommends_largest_layer_count_that_both_fits():
    metadata = make_metadata()
    profile = make_profile(ram_bytes=16 * 1024**3, gpu_memory=(4 * 1024**3,))
    result = auto(metadata, profile)

    assert result.status == "fit"
    assert result.gpu_name == "GPU 0"
    assert result.gpu_layers is not None
    assert 0 < result.gpu_layers <= metadata.block_count
    assert result.gpu_budget_bytes == int(4 * 1024**3 * 0.8)
    chosen = plan_model_memory(metadata, profile, gpu_layers=result.gpu_layers)
    assert chosen.overall_compatible is True
    if result.gpu_layers < metadata.block_count:
        rejected = plan_model_memory(
            metadata, profile, gpu_layers=result.gpu_layers + 1
        )
        assert rejected.overall_compatible is False
    assert "estimated" in result.reason.lower()


def test_unknown_vram_never_produces_confident_layer_count():
    result = auto(make_metadata(), make_profile(gpu_memory=(None,)))

    assert result.status == "unknown"
    assert result.gpu_layers is None
    assert result.gpu_budget_bytes is None
    assert "VRAM" in result.reason


def test_insufficient_vram_falls_back_to_zero_offload_layers():
    result = auto(
        make_metadata(file_size=4 * 1024**3),
        make_profile(ram_bytes=32 * 1024**3, gpu_memory=(512 * 1024**2,)),
    )

    assert result.status == "fit"
    assert result.gpu_layers == 0
    assert "VRAM" in result.reason


def test_cpu_ram_constraint_can_require_gpu_offload():
    metadata = make_metadata()
    profile = make_profile(ram_bytes=2 * 1024**3, gpu_memory=(8 * 1024**3,))
    result = auto(metadata, profile)

    assert result.status == "fit"
    assert result.gpu_layers is not None
    assert result.gpu_layers > 0
    assert (
        plan_model_memory(metadata, profile, gpu_layers=result.gpu_layers).cpu_fits
        is True
    )


def test_insufficient_cpu_ram_reports_no_safe_layer_count():
    result = auto(
        make_metadata(),
        make_profile(ram_bytes=64 * 1024**2, gpu_memory=(8 * 1024**3,)),
    )

    assert result.status == "infeasible"
    assert result.gpu_layers is None
    assert "CPU RAM" in result.reason


def test_missing_block_count_is_unknown_not_a_confident_fit():
    result = auto(make_metadata(layers=None), make_profile())

    assert result.status == "unknown"
    assert result.gpu_layers is None
    assert "block count" in result.reason


def test_invalid_block_count_and_negative_vram_are_rejected():
    with pytest.raises(ValueError, match="block count"):
        auto(make_metadata(layers=0), make_profile())

    with pytest.raises(ValueError, match="GPU memory"):
        auto(make_metadata(), make_profile(gpu_memory=(-1,)))


def test_kv_cache_placement_changes_recommendation():
    metadata = make_metadata()
    profile = make_profile(ram_bytes=16 * 1024**3, gpu_memory=(4 * 1024**3,))
    on_gpu = auto(metadata, profile, kv_cache_on_gpu=True)
    on_cpu = auto(metadata, profile, kv_cache_on_gpu=False)

    assert on_gpu.status == on_cpu.status == "fit"
    assert on_gpu.gpu_layers is not None
    assert on_cpu.gpu_layers is not None
    assert on_gpu.gpu_layers <= on_cpu.gpu_layers
    assert on_gpu.kv_cache_bytes == on_cpu.kv_cache_bytes


def test_quantized_kv_cache_allows_at_least_as_many_layers():
    metadata = make_metadata()
    profile = make_profile(ram_bytes=16 * 1024**3, gpu_memory=(3 * 1024**3,))
    fp16 = auto(metadata, profile, kv_cache_type="f16")
    q4 = auto(metadata, profile, kv_cache_type="q4_0")

    assert fp16.status == q4.status == "fit"
    assert q4.kv_cache_bytes < fp16.kv_cache_bytes
    assert q4.gpu_layers >= fp16.gpu_layers


def test_selected_gpu_index_is_used_and_vram_is_not_summed():
    metadata = make_metadata()
    profile = make_profile(
        ram_bytes=32 * 1024**3,
        gpu_memory=(512 * 1024**2, 8 * 1024**3),
    )
    small = auto(metadata, profile, gpu_index=0)
    large = auto(metadata, profile, gpu_index=1)

    assert small.gpu_name == "GPU 0"
    assert large.gpu_name == "GPU 1"
    assert large.gpu_budget_bytes == int(8 * 1024**3 * 0.8)
    assert small.gpu_layers is not None
    assert large.gpu_layers is not None
    assert large.gpu_layers > small.gpu_layers

    with pytest.raises(ValueError, match="GPU index"):
        auto(metadata, profile, gpu_index=2)


def test_unknown_cpu_ram_prevents_confident_recommendation():
    profile = make_profile(gpu_memory=())
    profile.ram_available_bytes = Provenance()

    result = auto(make_metadata(), profile)

    assert result.status == "unknown"
    assert result.gpu_layers is None
    assert "RAM" in result.reason


def test_unknown_kv_metadata_prevents_confident_recommendation():
    result = auto(make_metadata(embedding=None), make_profile())

    assert result.status == "unknown"
    assert result.gpu_layers is None
    assert "KV-cache" in result.reason


def test_context_length_changes_the_recommendation():
    metadata = make_metadata()
    profile = make_profile(ram_bytes=16 * 1024**3, gpu_memory=(3 * 1024**3,))
    short = auto(metadata, profile, context_tokens=2048)
    long = auto(metadata, profile, context_tokens=8192)

    assert short.status == long.status == "fit"
    assert short.kv_cache_bytes < long.kv_cache_bytes
    assert short.gpu_layers >= long.gpu_layers

    with pytest.raises(ValueError, match="context"):
        auto(metadata, profile, context_tokens=16_384)


def test_invalid_kv_cache_type_is_rejected_before_auto_selection():
    with pytest.raises(ValueError, match="kv_cache_type"):
        auto(make_metadata(), make_profile(), kv_cache_type="bogus")
