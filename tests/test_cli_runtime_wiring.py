import json
from pathlib import Path

from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.core.types import HardwareProfile, Provenance
from kernl.models.gguf import GGUFMetadata
from kernl.runtime.llama_cpp import RuntimeErrorBase

runner = CliRunner()


def make_profile(gpu_memory=(4 * 1024**3,)):
    return HardwareProfile(
        platform=Provenance(value="Windows 11"),
        architecture=Provenance(value="AMD64"),
        cpu_model=Provenance(value="Test CPU"),
        cpu_cores=Provenance(value=6),
        cpu_threads=Provenance(value=12),
        ram_total_bytes=Provenance(value=16 * 1024**3),
        ram_available_bytes=Provenance(value=12 * 1024**3),
        gpu_names=Provenance(value=[f"GPU {i}" for i in range(len(gpu_memory))]),
        gpu_driver_versions=Provenance(),
        gpu_memory_bytes=Provenance(value=list(gpu_memory)),
        storage_free_bytes=Provenance(value=100 * 1024**3),
        backends=Provenance(value=["cpu", "cuda"] if gpu_memory else ["cpu"]),
        scanned_at="now",
    )


def make_metadata(path: Path):
    return GGUFMetadata(
        path=path,
        version=3,
        tensor_count=100,
        metadata_count=10,
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=8192,
        embedding_length=4096,
        block_count=20,
        attention_head_count=32,
        attention_head_count_kv=8,
        quantization_version=2,
        file_size_bytes=4 * 1024**3,
        metadata={},
    )


def stub_auto_inputs(monkeypatch, tmp_path: Path, *, gpu_memory=(4 * 1024**3,)):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    monkeypatch.setattr("kernl.cli.app.scan", lambda: make_profile(gpu_memory))
    monkeypatch.setattr(
        "kernl.cli.app.GGUFReader.read", lambda self, path: make_metadata(path)
    )
    monkeypatch.setattr("kernl.cli.app.CONFIG_DIR", tmp_path / "configs")
    return model_path


def test_config_save_auto_stores_concrete_estimated_layers(monkeypatch, tmp_path):
    model_path = stub_auto_inputs(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "config",
            "save",
            "auto-model",
            str(model_path),
            "--gpu-layers",
            "auto",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "configs" / "auto-model.json").read_text())
    assert 0 < saved["gpu_layers"] <= 20
    assert "estimated" in result.output.lower()


def test_config_save_auto_refuses_unknown_memory(monkeypatch, tmp_path):
    model_path = stub_auto_inputs(monkeypatch, tmp_path, gpu_memory=(None,))

    result = runner.invoke(
        app,
        ["config", "save", "auto-model", str(model_path), "--gpu-layers", "auto"],
    )

    assert result.exit_code != 0
    assert "VRAM" in result.output
    assert not (tmp_path / "configs" / "auto-model.json").exists()


def test_config_save_keeps_explicit_layers(monkeypatch, tmp_path):
    model_path = stub_auto_inputs(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["config", "save", "explicit", str(model_path), "--gpu-layers", "7"],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "configs" / "explicit.json").read_text())
    assert saved["gpu_layers"] == 7


def test_runtime_verify_requires_approval_and_a_real_executable(monkeypatch, tmp_path):
    monkeypatch.setattr("kernl.cli.app.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("kernl.cli.app.shutil.which", lambda name: None)

    result = runner.invoke(app, ["runtime", "verify", "demo"])

    assert result.exit_code != 0
    assert "approval" in result.output.lower() or "executable" in result.output.lower()


def test_runtime_verify_runs_real_inference_when_approved(monkeypatch, tmp_path):
    from kernl.runtime.config import RuntimeConfig

    config = RuntimeConfig(model_path=tmp_path / "model.gguf")
    (tmp_path / "demo.json").write_text(config.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("kernl.cli.app.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "kernl.cli.app.shutil.which", lambda name: "C:/fake/llama-cli.exe"
    )
    monkeypatch.setattr(
        "kernl.cli.app.LlamaCppAdapter.infer_once", lambda self, *a, **kw: 5
    )

    result = runner.invoke(app, ["runtime", "verify", "demo", "--approve"])

    assert result.exit_code == 0, result.output
    assert "5 generated tokens" in result.output
    assert "observed" in result.output.lower()


def test_runtime_verify_reports_runtime_errors(monkeypatch, tmp_path):
    from kernl.runtime.config import RuntimeConfig

    config = RuntimeConfig(model_path=tmp_path / "model.gguf")
    (tmp_path / "demo.json").write_text(config.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("kernl.cli.app.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "kernl.cli.app.shutil.which", lambda name: "C:/fake/llama-cli.exe"
    )

    def fail(self, *args, **kwargs):
        raise RuntimeErrorBase("model load failed")

    monkeypatch.setattr("kernl.cli.app.LlamaCppAdapter.infer_once", fail)

    result = runner.invoke(app, ["runtime", "verify", "demo", "--approve"])

    assert result.exit_code != 0
    assert "model load failed" in result.output
