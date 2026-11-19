from pathlib import Path

from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.core.types import HardwareProfile, Provenance
from kernl.models.gguf import GGUFMetadata

runner = CliRunner()


def test_model_plan_command_displays_cpu_and_gpu_memory(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    profile = HardwareProfile(
        platform=Provenance(value="Windows 11"),
        architecture=Provenance(value="AMD64"),
        cpu_model=Provenance(value="Test CPU"),
        cpu_cores=Provenance(value=6),
        cpu_threads=Provenance(value=12),
        ram_total_bytes=Provenance(value=16 * 1024**3),
        ram_available_bytes=Provenance(value=12 * 1024**3),
        gpu_names=Provenance(value=["NVIDIA Test GPU"]),
        gpu_driver_versions=Provenance(),
        gpu_memory_bytes=Provenance(value=[8 * 1024**3]),
        storage_free_bytes=Provenance(value=100 * 1024**3),
        backends=Provenance(value=["cpu", "cuda"]),
        scanned_at="now",
    )
    metadata = GGUFMetadata(
        path=model_path,
        version=3,
        tensor_count=10,
        metadata_count=10,
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        embedding_length=4096,
        block_count=32,
        attention_head_count=32,
        attention_head_count_kv=8,
        quantization_version=2,
        file_size_bytes=1_000_000_000,
        metadata={},
    )
    monkeypatch.setattr("kernl.cli.app.scan", lambda: profile)
    monkeypatch.setattr("kernl.cli.app.GGUFReader.read", lambda self, path: metadata)

    result = runner.invoke(
        app,
        ["model", "plan", str(model_path), "--gpu-layers", "16"],
    )

    assert result.exit_code == 0, result.output
    assert "CPU RAM" in result.output
    assert "KV cache estimate" in result.output
    assert "NVIDIA Test GPU" in result.output
    assert "VRAM" in result.output
    assert "Overall compatibility" in result.output
