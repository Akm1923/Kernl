import io
import subprocess
from pathlib import Path

import pytest

from kernl.core.telemetry import InferenceTelemetry
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.llama_cpp import LlamaCppAdapter, RuntimeErrorBase

LLAMA_STDERR = """
llama_print_timings:        load time =    812.44 ms
llama_print_timings: prompt eval time =    456.78 ms /     32 tokens (   14.27 ms per token,     70.06 tokens per second)
llama_print_timings: eval time =    987.65 ms /     64 tokens (   15.43 ms per token,     64.81 tokens per second)
"""


class FakeProcess:
    def __init__(self, stdout_text="", stderr_text="", returncode=0):
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)
        self.pid = 4242
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def prepare(tmp_path: Path) -> tuple[LlamaCppAdapter, RuntimeConfig]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    executable = tmp_path / "llama-cli.exe"
    executable.write_bytes(b"fake")
    config = RuntimeConfig(model_path=model)
    return LlamaCppAdapter(executable), config


def patch_popen(monkeypatch, process: FakeProcess):
    monkeypatch.setattr(
        "kernl.runtime.llama_cpp.subprocess.Popen",
        lambda *args, **kwargs: process,
    )


def test_infer_telemetry_requires_approval(tmp_path: Path):
    adapter, config = prepare(tmp_path)

    with pytest.raises(PermissionError, match="approval"):
        adapter.infer_once_telemetry(config, "hello")


def test_infer_telemetry_returns_measured_signals(monkeypatch, tmp_path: Path):
    adapter, config = prepare(tmp_path)
    process = FakeProcess(
        stdout_text="one two three four\n",
        stderr_text=LLAMA_STDERR,
    )
    patch_popen(monkeypatch, process)

    class FakePsutilProcess:
        def memory_info(self):
            class Info:
                rss = 7777

            return Info()

    monkeypatch.setattr(
        "kernl.runtime.llama_cpp.psutil.Process",
        lambda pid: FakePsutilProcess(),
    )
    monkeypatch.setattr("kernl.runtime.llama_cpp.shutil.which", lambda name: None)

    telemetry = adapter.infer_once_telemetry(
        config, "hello", approve_execution=True, sample_interval=0.001
    )

    assert isinstance(telemetry, InferenceTelemetry)
    assert telemetry.generated_tokens == 4
    assert telemetry.time_to_first_output_seconds is not None
    assert telemetry.time_to_first_output_seconds >= 0
    assert telemetry.load_time_seconds == pytest.approx(0.81244)
    assert telemetry.prompt_eval_seconds == pytest.approx(0.45678)
    assert telemetry.prompt_tokens == 32
    assert telemetry.prompt_tokens_per_second == pytest.approx(70.06)
    assert telemetry.generation_tokens_per_second == pytest.approx(64.81)
    assert telemetry.peak_rss_bytes == 7777
    assert telemetry.peak_vram_bytes is None
    assert telemetry.wall_seconds > 0

    payload = telemetry.to_dict()
    assert payload["origins"]["load_time_seconds"] == "measured"
    assert payload["origins"]["peak_vram_bytes"] == "unknown"


def test_infer_telemetry_without_timings_marks_unknown(monkeypatch, tmp_path: Path):
    adapter, config = prepare(tmp_path)
    process = FakeProcess(stdout_text="only output\n", stderr_text="")
    patch_popen(monkeypatch, process)
    monkeypatch.setattr(
        "kernl.runtime.llama_cpp.psutil.Process",
        lambda pid: (_ for _ in ()).throw(OSError("gone")),
    )
    monkeypatch.setattr("kernl.runtime.llama_cpp.shutil.which", lambda name: None)

    telemetry = adapter.infer_once_telemetry(
        config, "hello", approve_execution=True, sample_interval=0.001
    )

    assert telemetry.load_time_seconds is None
    assert telemetry.prompt_eval_seconds is None
    assert telemetry.prompt_tokens_per_second is None
    assert telemetry.peak_rss_bytes is None
    assert telemetry.peak_vram_bytes is None
    assert telemetry.to_dict()["origins"]["load_time_seconds"] == "unknown"


def test_infer_telemetry_reports_nonzero_exit(monkeypatch, tmp_path: Path):
    adapter, config = prepare(tmp_path)
    process = FakeProcess(stderr_text="model load failed", returncode=1)
    process.returncode = None

    class FailingProcess(FakeProcess):
        def wait(self, timeout=None):
            return 1

        @property
        def returncode(self):
            return 1

        @returncode.setter
        def returncode(self, value):
            self._returncode = value

    failing = FailingProcess(stderr_text="model load failed")
    patch_popen(monkeypatch, failing)
    monkeypatch.setattr("kernl.runtime.llama_cpp.shutil.which", lambda name: None)

    with pytest.raises(RuntimeErrorBase, match="model load failed"):
        adapter.infer_once_telemetry(
            config, "hello", approve_execution=True, sample_interval=0.001
        )


def test_infer_telemetry_times_out(monkeypatch, tmp_path: Path):
    adapter, config = prepare(tmp_path)

    class HangingProcess(FakeProcess):
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="llama-cli", timeout=timeout)

    hanging = HangingProcess(stdout_text="partial\n")
    patch_popen(monkeypatch, hanging)
    monkeypatch.setattr("kernl.runtime.llama_cpp.shutil.which", lambda name: None)

    with pytest.raises(RuntimeErrorBase, match="timed out"):
        adapter.infer_once_telemetry(
            config,
            "hello",
            approve_execution=True,
            timeout=0.01,
            sample_interval=0.001,
        )
    assert hanging.killed is True


def test_infer_once_returns_generated_tokens(monkeypatch, tmp_path: Path):
    adapter, config = prepare(tmp_path)
    process = FakeProcess(stdout_text="alpha beta gamma\n", stderr_text="")
    patch_popen(monkeypatch, process)
    monkeypatch.setattr("kernl.runtime.llama_cpp.shutil.which", lambda name: None)

    tokens = adapter.infer_once(config, "hello", approve_execution=True)

    assert tokens == 3


def test_infer_telemetry_rejects_empty_prompt(tmp_path: Path):
    adapter, config = prepare(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        adapter.infer_once_telemetry(config, "   ", approve_execution=True)
