from pathlib import Path

from kernl.models.memory import Quantization, estimate_memory, parse_quantization
from kernl.runtime.config import RuntimeConfig


def test_quantization_alias():
    assert parse_quantization("q4-k-s") == Quantization.Q4_K_M
    assert parse_quantization("not-real") == Quantization.UNKNOWN


def test_memory_estimate_is_explicit():
    result = estimate_memory(
        parameter_count=7_000_000_000,
        quantization="Q4_K_M",
        context_tokens=2048,
        layers=32,
        embedding_length=4096,
        available_bytes=16_000_000_000,
    )
    assert result.estimated is True
    assert result.total_bytes is not None
    assert result.fits is not None
    assert result.safety_margin_bytes is not None


def test_unknown_memory_does_not_claim_fit():
    result = estimate_memory(parameter_count=None, quantization=None)
    assert result.total_bytes is None
    assert result.fits is None
    assert result.warnings


def test_runtime_arguments_are_deterministic(tmp_path: Path):
    config = RuntimeConfig(
        model_path=tmp_path / "model.gguf",
        batch_size=256,
        microbatch_size=64,
        flash_attention=True,
        threads=8,
    )
    args = config.runtime_arguments()
    assert args[:2] == ["--model", str(tmp_path / "model.gguf")]
    assert "--flash-attn" in args
    assert "--threads" in args


def test_invalid_microbatch_is_rejected(tmp_path: Path):
    try:
        RuntimeConfig(
            model_path=tmp_path / "model.gguf", batch_size=32, microbatch_size=64
        )
    except ValueError as exc:
        assert "microbatch_size" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_partial_memory_estimate_does_not_claim_fit():
    result = estimate_memory(
        parameter_count=7_000_000_000,
        quantization="Q4_K_M",
        context_tokens=2048,
        available_bytes=16_000_000_000,
    )
    assert result.total_bytes is not None
    assert result.kv_cache_bytes is None
    assert result.fits is None


def test_memory_estimate_rejects_non_positive_dimensions():
    for field, value in (
        ("layers", 0),
        ("layers", -1),
        ("embedding_length", 0),
        ("embedding_length", -1),
    ):
        kwargs = {field: value}
        try:
            estimate_memory(
                parameter_count=7_000_000_000, quantization="Q4_K_M", **kwargs
            )
        except ValueError:
            continue
        raise AssertionError(f"expected invalid {field}={value} to be rejected")


def test_memory_estimate_rejects_negative_available_memory():
    try:
        estimate_memory(
            parameter_count=7_000_000_000, quantization="Q4_K_M", available_bytes=-1
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected negative available memory to be rejected")


def test_runtime_config_can_keep_kv_cache_on_cpu(tmp_path: Path):
    config = RuntimeConfig(
        model_path=tmp_path / "model.gguf",
        gpu_layers=12,
        kv_cache_on_gpu=False,
    )

    assert "--no-kv-offload" in config.runtime_arguments()


def test_legacy_runtime_config_defaults_kv_cache_to_gpu(tmp_path: Path):
    import json

    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"model_path": str(tmp_path / "model.gguf")}),
        encoding="utf-8",
    )
    loaded = RuntimeConfig.model_validate_json(path.read_text(encoding="utf-8"))

    assert loaded.kv_cache_on_gpu is True


def test_all_gpu_layers_uses_llama_cpp_all_keyword(tmp_path: Path):
    config = RuntimeConfig(model_path=tmp_path / "model.gguf", gpu_layers=-1)

    args = config.runtime_arguments()
    gpu_layers_index = args.index("--gpu-layers")

    assert args[gpu_layers_index + 1] == "all"
