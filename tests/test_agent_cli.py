import json
from pathlib import Path

from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.core.types import HardwareProfile, Provenance
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.config_store import ConfigStore

runner = CliRunner()


def prepare(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr("kernl.cli.app.DATA_DIR", tmp_path / "data")
    return tmp_path / "data"


def make_profile() -> HardwareProfile:
    return HardwareProfile(
        platform=Provenance(value="Windows 11"),
        architecture=Provenance(value="AMD64"),
        cpu_model=Provenance(value="Test CPU"),
        cpu_cores=Provenance(value=8),
        cpu_threads=Provenance(value=16),
        ram_total_bytes=Provenance(value=16 * 1024**3),
        ram_available_bytes=Provenance(value=12 * 1024**3),
        gpu_names=Provenance(value=["GPU 0"]),
        gpu_driver_versions=Provenance(),
        gpu_memory_bytes=Provenance(value=[4 * 1024**3]),
        storage_free_bytes=Provenance(value=100 * 1024**3),
        backends=Provenance(value=["cpu", "cuda"]),
        scanned_at="now",
    )


def test_agent_default_view_preserves_existing_command_and_mentions_no_shell(
    monkeypatch, tmp_path
):
    prepare(monkeypatch, tmp_path)

    result = runner.invoke(app, ["agent"])

    assert result.exit_code == 0, result.output
    assert "KERNL" in result.output
    assert "no shell execution" in result.output.lower()


def test_agent_tools_lists_typed_tools_and_approval_gates(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)

    result = runner.invoke(app, ["agent", "tools"])

    assert result.exit_code == 0, result.output
    for tool in ("hardware_profile", "local_model_plan", "runtime_verify"):
        assert tool in result.output
    assert "required" in result.output
    assert "read-only" in result.output


def test_agent_run_hardware_profile_returns_detected_data(monkeypatch, tmp_path):
    data_dir = prepare(monkeypatch, tmp_path)
    from kernl.cli.app import get_store

    get_store().save_profile(make_profile())

    result = runner.invoke(app, ["agent", "run", "hardware_profile"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tool"] == "hardware_profile"
    assert payload["result"]["origin"] == "detected"
    assert payload["result"]["platform"] == "Windows 11"
    assert data_dir.exists()


def test_agent_run_runtime_status_uses_injected_resolver(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "kernl.agent.tools.shutil.which", lambda name: "C:/fake/llama-cli.exe"
    )

    result = runner.invoke(
        app, ["agent", "run", "runtime_status", "--arg", "executable=llama-cli"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result"]["resolved_path"] == "C:/fake/llama-cli.exe"
    assert payload["result"]["origin"] == "detected"


def test_agent_run_denies_unapproved_execution_and_audits(monkeypatch, tmp_path):
    data_dir = prepare(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["agent", "run", "runtime_verify", "--arg", "config_name=demo"],
        env={"COLUMNS": "60"},
    )

    assert result.exit_code != 0
    assert "approval required" in result.output.lower()
    from kernl.cli.app import get_store

    rows = get_store().recent_agent_actions()
    assert rows
    assert rows[0]["tool"] == "runtime_verify"
    assert rows[0]["approved"] is False
    assert data_dir.exists()


def test_agent_run_approved_execution_is_measured_and_audited(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    config_dir = tmp_path / "configs"
    ConfigStore(config_dir).save("demo", RuntimeConfig(model_path=model_path))
    monkeypatch.setattr("kernl.cli.app.CONFIG_DIR", config_dir)
    monkeypatch.setattr(
        "kernl.agent.tools.shutil.which", lambda name: "C:/fake/llama-cli.exe"
    )
    monkeypatch.setattr(
        "kernl.agent.tools.LlamaCppAdapter.infer_once", lambda self, *a, **kw: 7
    )

    result = runner.invoke(
        app,
        ["agent", "run", "runtime_verify", "--arg", "config_name=demo", "--approve"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result"]["generated_tokens"] == 7
    assert payload["result"]["origin"] == "measured"

    audit = runner.invoke(app, ["agent", "audit"])
    assert audit.exit_code == 0, audit.output
    assert "runtime_verify" in audit.output
    assert "measured" in audit.output or '"origin": "measured"' in audit.output


def test_agent_run_rejects_unknown_tool_and_outside_workspace_path(
    monkeypatch, tmp_path
):
    prepare(monkeypatch, tmp_path)

    unknown = runner.invoke(app, ["agent", "run", "shell"])
    assert unknown.exit_code != 0
    assert "unknown agent tool" in unknown.output

    outside = runner.invoke(
        app,
        [
            "agent",
            "run",
            "local_model_inspect",
            "--arg",
            f"path={tmp_path / 'outside.gguf'}",
        ],
    )
    assert outside.exit_code != 0
    assert "workspace" in outside.output.lower()
