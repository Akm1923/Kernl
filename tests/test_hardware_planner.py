from pathlib import Path

import pytest

from kernl.core.types import HardwareProfile, Provenance
from kernl.models.gguf import GGUFMetadata
from kernl.models.planner import plan_model_memory


def make_profile(*, ram_bytes=8 * 1024**3, gpu_memory=(4 * 1024**3,)):
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
        backends=Provenance(value=["cpu", "cuda"]),
        scanned_at="now",
    )


def make_metadata(*, file_size=1_000_000_000, layers=20, context=8192, embedding=4096):
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
        attention_head_count_kv=8,
        quantization_version=2,
        file_size_bytes=file_size,
        metadata={},
    )


def test_cpu_only_plan_uses_system_ram_and_does_not_require_gpu():
    plan = plan_model_memory(make_metadata(), make_profile(gpu_memory=()), gpu_layers=0)

    assert plan.gpu_name is None
    assert plan.gpu_fits is None
    assert plan.cpu_fits is True
    assert plan.overall_compatible is True


def test_gpu_offload_checks_selected_gpu_vram_separately_from_system_ram():
    plan = plan_model_memory(
        make_metadata(),
        make_profile(ram_bytes=16 * 1024**3, gpu_memory=(8 * 1024**3,)),
        gpu_layers=10,
    )

    assert plan.gpu_name == "GPU 0"
    assert plan.gpu_required_bytes is not None
    assert plan.gpu_budget_bytes == int(8 * 1024**3 * 0.8)
    assert plan.gpu_fits is True
    assert plan.cpu_fits is True
    assert plan.overall_compatible is True


def test_gpu_overflow_is_incompatible_even_when_system_ram_is_large():
    plan = plan_model_memory(
        make_metadata(file_size=8 * 1024**3),
        make_profile(ram_bytes=64 * 1024**3, gpu_memory=(2 * 1024**3,)),
        gpu_layers=20,
    )

    assert plan.cpu_fits is True
    assert plan.gpu_fits is False
    assert plan.overall_compatible is False


def test_multiple_gpu_memory_is_not_summed_implicitly():
    plan = plan_model_memory(
        make_metadata(file_size=4 * 1024**3),
        make_profile(
            ram_bytes=32 * 1024**3,
            gpu_memory=(2 * 1024**3, 16 * 1024**3),
        ),
        gpu_layers=20,
        gpu_index=0,
    )

    assert plan.gpu_name == "GPU 0"
    assert plan.gpu_fits is False


def test_unknown_vram_never_claims_gpu_compatibility():
    plan = plan_model_memory(
        make_metadata(),
        make_profile(gpu_memory=(None,)),
        gpu_layers=10,
    )

    assert plan.gpu_fits is None
    assert plan.overall_compatible is None
    assert any("VRAM is unknown" in warning for warning in plan.warnings)


def test_gpu_offload_requires_known_layer_count_and_valid_index():
    metadata = make_metadata(layers=None)
    with pytest.raises(ValueError, match="block count"):
        plan_model_memory(metadata, make_profile(), gpu_layers=1)

    with pytest.raises(ValueError, match="GPU index"):
        plan_model_memory(make_metadata(), make_profile(), gpu_layers=1, gpu_index=2)


def test_gpu_layer_count_must_be_within_model_layer_count():
    with pytest.raises(ValueError, match="gpu_layers"):
        plan_model_memory(make_metadata(layers=20), make_profile(), gpu_layers=21)


def test_all_layers_sentinel_resolves_to_model_layer_count():
    plan = plan_model_memory(make_metadata(layers=20), make_profile(), gpu_layers=-1)

    assert plan.gpu_layers == 20


def test_context_cannot_exceed_model_declared_context():
    with pytest.raises(ValueError, match="context"):
        plan_model_memory(
            make_metadata(context=4096),
            make_profile(),
            gpu_layers=0,
            context_tokens=8192,
        )


def test_unknown_kv_metadata_prevents_false_positive_fit():
    metadata = make_metadata(layers=None, embedding=None)
    plan = plan_model_memory(metadata, make_profile(gpu_memory=()), gpu_layers=0)

    assert plan.cpu_required_bytes is not None
    assert plan.cpu_fits is None
    assert plan.overall_compatible is None


def test_gqa_metadata_reduces_kv_cache_estimate():
    metadata = make_metadata(layers=20, embedding=4096)
    plan = plan_model_memory(metadata, make_profile(), gpu_layers=0)
    expected = 2 * 2 * 20 * (4096 * 8 // 32) * 8192

    assert plan.kv_cache_bytes == expected


def test_kv_cache_on_cpu_moves_requirement_from_vram_to_system_ram():
    metadata = make_metadata()
    profile = make_profile(ram_bytes=16 * 1024**3, gpu_memory=(8 * 1024**3,))
    gpu_kv = plan_model_memory(metadata, profile, gpu_layers=10)
    cpu_kv = plan_model_memory(metadata, profile, gpu_layers=10, kv_cache_on_gpu=False)

    assert gpu_kv.gpu_required_bytes > cpu_kv.gpu_required_bytes
    assert gpu_kv.cpu_required_bytes < cpu_kv.cpu_required_bytes


def test_quantized_kv_cache_uses_less_memory_than_fp16():
    metadata = make_metadata()
    profile = make_profile(gpu_memory=())
    fp16 = plan_model_memory(metadata, profile, kv_cache_type="f16")
    q4 = plan_model_memory(metadata, profile, kv_cache_type="q4_0")

    assert q4.kv_cache_bytes < fp16.kv_cache_bytes


def test_missing_kv_metadata_does_not_make_cpu_fit_unknown_when_kv_is_offloaded():
    metadata = make_metadata(layers=20, embedding=None)
    plan = plan_model_memory(
        metadata, make_profile(), gpu_layers=1, kv_cache_on_gpu=True
    )

    assert plan.cpu_fits is True
    assert plan.gpu_fits is None
    assert plan.overall_compatible is None


def test_explicit_context_still_warns_when_model_context_metadata_is_missing():
    metadata = make_metadata(context=None)
    plan = plan_model_memory(
        metadata,
        make_profile(gpu_memory=()),
        gpu_layers=0,
        context_tokens=2048,
    )

    assert any(
        "cannot be checked against the model limit" in warning
        for warning in plan.warnings
    )


def test_negative_profile_memory_values_are_rejected():
    with pytest.raises(ValueError, match="available system RAM"):
        plan_model_memory(
            make_metadata(),
            make_profile(ram_bytes=-1),
            gpu_layers=0,
        )

    with pytest.raises(ValueError, match="GPU memory"):
        plan_model_memory(
            make_metadata(),
            make_profile(gpu_memory=(-1,)),
            gpu_layers=1,
        )


def test_invalid_attention_head_geometry_is_rejected():
    metadata = make_metadata(embedding=4095)
    with pytest.raises(ValueError, match="divisible"):
        plan_model_memory(metadata, make_profile(), gpu_layers=0)
