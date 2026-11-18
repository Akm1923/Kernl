from pathlib import Path
import pytest
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.llama_cpp import LlamaCppAdapter


def test_detection_rejects_missing_binary(tmp_path: Path):
    with pytest.raises(RuntimeError):
        LlamaCppAdapter(tmp_path / "missing.exe").detect()


def test_execution_requires_approval(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not-a-real-model")
    config = RuntimeConfig(model_path=model)
    adapter = LlamaCppAdapter(tmp_path / "llama-cli.exe")
    with pytest.raises(PermissionError):
        adapter.launch(config)


def test_runtime_arguments_do_not_use_shell_string(tmp_path: Path):
    config = RuntimeConfig(model_path=tmp_path / "model file.gguf")
    args = config.runtime_arguments()
    assert isinstance(args, list)
    assert "model file.gguf" in args
