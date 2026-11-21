import json
from pathlib import Path

from typer.testing import CliRunner

from kernl.cli.app import app
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.config_store import ConfigStore

runner = CliRunner()


def prepare_config(monkeypatch, tmp_path: Path, name: str = "demo") -> Path:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    config_dir = tmp_path / "configs"
    ConfigStore(config_dir).save(name, RuntimeConfig(model_path=model_path))
    monkeypatch.setattr("kernl.cli.app.CONFIG_DIR", config_dir)
    monkeypatch.setattr("kernl.cli.app.DATA_DIR", tmp_path / "data")
    return model_path


def test_benchmark_run_requires_approval_and_executable(monkeypatch, tmp_path):
    prepare_config(monkeypatch, tmp_path)
    monkeypatch.setattr("kernl.cli.benchmark.shutil.which", lambda name: None)

    result = runner.invoke(app, ["benchmark", "run", "demo", "--prompt", "hello"])

    assert result.exit_code != 0
    assert "approval" in result.output.lower() or "executable" in result.output.lower()


def test_benchmark_run_measures_real_callback_and_stores_results(monkeypatch, tmp_path):
    prepare_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "kernl.cli.benchmark.shutil.which", lambda name: "C:/fake/llama-cli.exe"
    )
    calls: list[str] = []

    def fake_infer(self, config, prompt, *, approve_execution, timeout):
        calls.append(prompt)
        assert approve_execution is True
        from kernl.core.telemetry import InferenceTelemetry

        return InferenceTelemetry(
            generated_tokens=4,
            wall_seconds=1.0,
            time_to_first_output_seconds=0.3,
            load_time_seconds=0.5,
            prompt_eval_seconds=0.2,
            prompt_tokens=8,
            prompt_tokens_per_second=40.0,
            generation_tokens_per_second=4.0,
            peak_rss_bytes=1024,
            peak_vram_bytes=None,
        )

    monkeypatch.setattr(
        "kernl.cli.benchmark.LlamaCppAdapter.infer_once_telemetry", fake_infer
    )
    output = tmp_path / "results" / "run.json"

    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "demo",
            "--prompt",
            "hello",
            "--prompt",
            "write code",
            "--approve",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["hello", "write code"]
    assert "measured" in result.output.lower()
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()
    payload = json.loads(output.read_text())
    assert len(payload["samples"]) == 2
    assert payload["median_tokens_per_second"] > 0

    from kernl.cli.app import get_store

    stored = get_store().latest_benchmark("demo")
    assert stored is not None
    assert stored["runtime"].startswith("llama.cpp")


def test_benchmark_run_reports_runtime_failures(monkeypatch, tmp_path):
    prepare_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "kernl.cli.benchmark.shutil.which", lambda name: "C:/fake/llama-cli.exe"
    )

    def fail(self, config, prompt, *, approve_execution, timeout):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(
        "kernl.cli.benchmark.LlamaCppAdapter.infer_once_telemetry", fail
    )

    result = runner.invoke(
        app, ["benchmark", "run", "demo", "--prompt", "hello", "--approve"]
    )

    assert result.exit_code != 0
    assert "model load failed" in result.output


def test_benchmark_compare_uses_stored_measurements(monkeypatch, tmp_path):
    prepare_config(monkeypatch, tmp_path, "left")
    prepare_config(monkeypatch, tmp_path, "right")
    from kernl.cli.app import get_store

    store = get_store()
    store.save_benchmark(
        "left",
        json.dumps(
            {
                "runtime": "llama.cpp",
                "model": "m",
                "samples": [],
                "median_tokens_per_second": 42.5,
                "p95_latency_seconds": 0.2,
                "measured_at_epoch": 1,
            }
        ),
        1,
    )
    store.save_benchmark(
        "right",
        json.dumps(
            {
                "runtime": "llama.cpp",
                "model": "m",
                "samples": [],
                "median_tokens_per_second": 21.0,
                "p95_latency_seconds": 0.4,
                "measured_at_epoch": 2,
            }
        ),
        2,
    )

    result = runner.invoke(app, ["benchmark", "compare", "left", "right"])

    assert result.exit_code == 0, result.output
    assert "42.50" in result.output
    assert "21.00" in result.output
    assert "measured" in result.output.lower()
