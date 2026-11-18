from pathlib import Path
from kernl.models.memory import estimate_memory, parse_quantization, Quantization
from kernl.runtime.config import RuntimeConfig


def test_quantization_alias():
    assert parse_quantization("q4-k-s") == Quantization.Q4_K_M
    assert parse_quantization("not-real") == Quantization.UNKNOWN


def test_memory_estimate_is_explicit():
    result = estimate_memory(parameter_count=7_000_000_000, quantization="Q4_K_M", context_tokens=2048, layers=32, embedding_length=4096, available_bytes=16_000_000_000)
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
    config = RuntimeConfig(model_path=tmp_path / "model.gguf", batch_size=256, microbatch_size=64, flash_attention=True, threads=8)
    args = config.runtime_arguments()
    assert args[:2] == ["--model", str(tmp_path / "model.gguf")]
    assert "--flash-attn" in args
    assert "--threads" in args


def test_invalid_microbatch_is_rejected(tmp_path: Path):
    try:
        RuntimeConfig(model_path=tmp_path / "model.gguf", batch_size=32, microbatch_size=64)
    except ValueError as exc:
        assert "microbatch_size" in str(exc)
    else:
        raise AssertionError("expected validation failure")
