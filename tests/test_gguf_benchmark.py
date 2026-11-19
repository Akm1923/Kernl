import struct
import time
from pathlib import Path

import pytest

from kernl.benchmark import BenchmarkError, BenchmarkRunner
from kernl.models.gguf import GGUFError, GGUFReader


def _write_string(handle, value: str):
    data = value.encode()
    handle.write(struct.pack("<Q", len(data)))
    handle.write(data)


def _write_value(handle, value):
    if isinstance(value, str):
        handle.write(struct.pack("<I", 11))
        _write_string(handle, value)
    elif isinstance(value, int):
        handle.write(struct.pack("<I", 4))
        handle.write(struct.pack("<Q", value))
    else:
        raise TypeError(f"unsupported fixture value: {value!r}")


def test_gguf_metadata_reader(tmp_path: Path):
    path = tmp_path / "test.gguf"
    entries = [
        ("general.architecture", "llama"),
        ("llama.context_length", 4096),
        ("llama.embedding_length", 4096),
        ("llama.block_count", 32),
    ]
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", 3, 0, len(entries)))
        for key, value in entries:
            _write_string(handle, key)
            _write_value(handle, value)
    result = GGUFReader().read(path)
    assert result.architecture == "llama"
    assert result.context_length == 4096
    assert result.block_count == 32


def test_gguf_rejects_bad_magic(tmp_path: Path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"nope")
    with pytest.raises(GGUFError):
        GGUFReader().read(path)


def test_benchmark_records_real_callback(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    result = BenchmarkRunner().run(
        runtime="test",
        model=model,
        prompts=["hello", "write code"],
        infer=lambda _: time.sleep(0.001) or 4,
    )
    assert result.median_tokens_per_second > 0
    assert len(result.samples) == 2


def test_benchmark_rejects_invalid_callback(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    with pytest.raises(BenchmarkError):
        BenchmarkRunner().run(
            runtime="test", model=model, prompts=["hello"], infer=lambda _: 0
        )
