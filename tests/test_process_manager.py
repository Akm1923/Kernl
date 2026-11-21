from pathlib import Path

import psutil
import pytest
from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.manager import ProcessError, RuntimeProcessManager
from kernl.runtime.process_store import ProcessStore, RuntimeProcessRecord

runner = CliRunner()


class FakeProcess:
    def __init__(self, pid=4242, name="llama-cli.exe", running=True):
        self.pid = pid
        self._name = name
        self.running = running
        self.terminated = False

    def name(self):
        return self._name

    def cmdline(self):
        return [f"C:/fake/{self._name}", "--model", "model.gguf"]

    def is_running(self):
        return self.running

    def send_signal(self, signal):
        self.terminated = True
        self.running = False

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.terminated = True
        self.running = False

    def wait(self, timeout=None):
        return 0


class FakeAdapter:
    def __init__(self, executable):
        self.executable = executable
        self.calls = []

    def launch(self, config, *, approve_execution, log_path, extra_args=()):
        self.calls.append(
            {
                "approve_execution": approve_execution,
                "log_path": log_path,
                "extra_args": tuple(extra_args),
            }
        )
        return 4242


def make_manager(tmp_path, process=None, adapter=None):
    store = ProcessStore(tmp_path / "processes.json")
    adapters = []

    def adapter_factory(executable):
        instance = adapter or FakeAdapter(executable)
        adapters.append(instance)
        return instance

    manager = RuntimeProcessManager(
        store,
        tmp_path / "logs",
        adapter_factory=adapter_factory,
        process_factory=lambda pid: (
            process if process is not None else FakeProcess(pid=pid)
        ),
    )
    return manager, store, adapters


def make_config(tmp_path):
    return RuntimeConfig(model_path=tmp_path / "model.gguf")


def test_launch_requires_approval(tmp_path):
    manager, _, adapters = make_manager(tmp_path)

    with pytest.raises(PermissionError, match="approval"):
        manager.launch(
            config_name="demo",
            config=make_config(tmp_path),
            executable=Path("C:/fake/llama-cli.exe"),
        )

    assert adapters == []


def test_launch_persists_record_and_log_path(tmp_path):
    manager, store, adapters = make_manager(tmp_path)

    record = manager.launch(
        config_name="demo",
        config=make_config(tmp_path),
        executable=Path("C:/fake/llama-cli.exe"),
        approve_execution=True,
        extra_args=["--prompt", "Hello"],
    )

    assert record.pid == 4242
    assert record.status == "running"
    assert Path(record.log_path).parent == tmp_path / "logs"
    assert store.get(4242) is not None
    assert adapters[0].calls[0]["approve_execution"] is True
    assert adapters[0].calls[0]["extra_args"] == ("--prompt", "Hello")


def test_duplicate_running_configuration_is_rejected(tmp_path):
    manager, _, _ = make_manager(tmp_path)
    manager.launch(
        config_name="demo",
        config=make_config(tmp_path),
        executable=Path("C:/fake/llama-cli.exe"),
        approve_execution=True,
    )

    with pytest.raises(ProcessError, match="already running"):
        manager.launch(
            config_name="demo",
            config=make_config(tmp_path),
            executable=Path("C:/fake/llama-cli.exe"),
            approve_execution=True,
        )


def test_status_marks_missing_process_stopped(tmp_path):
    def missing(pid):
        raise psutil.NoSuchProcess(pid=pid)

    manager, store, _ = make_manager(tmp_path, process=None)
    store.add(
        RuntimeProcessRecord(
            pid=4242,
            config_name="demo",
            executable="C:/fake/llama-cli.exe",
            model_path="model.gguf",
            started_at=1.0,
            log_path=str(tmp_path / "demo.log"),
        )
    )
    manager._process_factory = missing

    status = manager.status(4242)

    assert status.status == "stopped"
    assert store.get(4242).status == "stopped"


def test_status_marks_pid_reuse_foreign(tmp_path):
    manager, store, _ = make_manager(tmp_path, process=FakeProcess(name="other.exe"))
    store.add(
        RuntimeProcessRecord(
            pid=4242,
            config_name="demo",
            executable="C:/fake/llama-cli.exe",
            model_path="model.gguf",
            started_at=1.0,
            log_path=str(tmp_path / "demo.log"),
        )
    )

    status = manager.status(4242)

    assert status.status == "foreign"


def test_stop_terminates_matching_process(tmp_path):
    process = FakeProcess()
    manager, store, _ = make_manager(tmp_path, process=process)
    store.add(
        RuntimeProcessRecord(
            pid=4242,
            config_name="demo",
            executable="C:/fake/llama-cli.exe",
            model_path="model.gguf",
            started_at=1.0,
            log_path=str(tmp_path / "demo.log"),
        )
    )

    record = manager.stop(4242)

    assert record.status == "stopped"
    assert process.terminated is True


def test_stop_refuses_foreign_pid(tmp_path):
    manager, store, _ = make_manager(tmp_path, process=FakeProcess(name="other.exe"))
    store.add(
        RuntimeProcessRecord(
            pid=4242,
            config_name="demo",
            executable="C:/fake/llama-cli.exe",
            model_path="model.gguf",
            started_at=1.0,
            log_path=str(tmp_path / "demo.log"),
        )
    )

    with pytest.raises(ProcessError, match="refusing"):
        manager.stop(4242)


def test_logs_read_persisted_log_file(tmp_path):
    manager, store, _ = make_manager(tmp_path)
    log_path = tmp_path / "demo.log"
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    store.add(
        RuntimeProcessRecord(
            pid=4242,
            config_name="demo",
            executable="C:/fake/llama-cli.exe",
            model_path="model.gguf",
            started_at=1.0,
            log_path=str(log_path),
        )
    )

    assert manager.logs(4242, limit=2) == ["two", "three"]


def test_cli_run_requires_approval():
    result = runner.invoke(app, ["run", "demo"])

    assert result.exit_code != 0
    assert "approval" in result.output.lower()


def test_cli_processes_reports_managed_record(monkeypatch):
    record = RuntimeProcessRecord(
        pid=4242,
        config_name="demo",
        executable="C:/fake/llama-cli.exe",
        model_path="model.gguf",
        started_at=1.0,
        log_path="C:/logs/demo.log",
    )

    class FakeManager:
        def list(self):
            return [record]

    monkeypatch.setattr("kernl.cli.app.get_process_manager", lambda: FakeManager())

    result = runner.invoke(app, ["processes"])

    assert result.exit_code == 0, result.output
    assert "4242" in result.output
    assert "demo" in result.output
    assert "running" in result.output
