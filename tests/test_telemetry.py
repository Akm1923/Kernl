import json
from pathlib import Path

import pytest

from kernl.benchmark import BenchmarkError, BenchmarkRunner
from kernl.core.telemetry import InferenceTelemetry
from kernl.runtime.llama_cpp import _parse_runtime_timings, _sample_vram_bytes

LLAMA_STDERR = """
llama_print_timings:        load time =    812.44 ms
llama_print_timings: prompt eval time =    456.78 ms /     32 tokens (   14.27 ms per token,     70.06 tokens per second)
llama_print_timings: eval time =    987.65 ms /     64 tokens (   15.43 ms per token,     64.81 tokens per second)
llama_print_timings:       total time =   2256.87 ms
"""


def test_parses_llama_cpp_timings_into_measured_values():
    timings = _parse_runtime_timings(LLAMA_STDERR)

    assert timings["load_time_seconds"] == pytest.approx(0.81244)
    assert timings["prompt_eval_seconds"] == pytest.approx(0.45678)
    assert timings["prompt_tokens"] == 32
    assert timings["prompt_tokens_per_second"] == pytest.approx(70.06)
    assert timings["generation_seconds"] == pytest.approx(0.98765)
    assert timings["generation_tokens"] == 64
    assert timings["generation_tokens_per_second"] == pytest.approx(64.81)


def test_missing_timings_stay_unknown_not_zero():
    timings = _parse_runtime_timings("no timings here")

    assert all(value is None for value in timings.values())


def test_prompt_eval_is_not_mistaken_for_generation_eval():
    timings = _parse_runtime_timings(LLAMA_STDERR)

    # generation eval must come from the "eval time" line, not "prompt eval time"
    assert timings["generation_seconds"] == pytest.approx(0.98765)


def test_vram_sampler_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr("kernl.runtime.llama_cpp.shutil.which", lambda name: None)

    assert _sample_vram_bytes(1234) is None


def test_vram_sampler_reads_matching_pid(monkeypatch):
    monkeypatch.setattr(
        "kernl.runtime.llama_cpp.shutil.which", lambda name: "C:/nvidia-smi.exe"
    )

    class FakeCompleted:
        returncode = 0
        stdout = "999, 100\n1234, 512\n"

    monkeypatch.setattr(
        "kernl.runtime.llama_cpp.subprocess.run", lambda *a, **k: FakeCompleted()
    )

    assert _sample_vram_bytes(1234) == 512 * 1024 * 1024


def make_runner():
    return BenchmarkRunner()


def make_model(tmp_path: Path) -> Path:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    return model


def test_runner_records_full_telemetry(tmp_path: Path):
    telemetry = InferenceTelemetry(
        generated_tokens=8,
        wall_seconds=2.0,
        time_to_first_output_seconds=0.4,
        load_time_seconds=0.6,
        prompt_eval_seconds=0.5,
        prompt_tokens=16,
        prompt_tokens_per_second=32.0,
        generation_tokens_per_second=4.0,
        peak_rss_bytes=4096,
        peak_vram_bytes=None,
    )

    result = make_runner().run(
        runtime="llama.cpp",
        model=make_model(tmp_path),
        prompts=["hello world", "second prompt"],
        infer=lambda prompt: telemetry,
    )

    assert result.median_time_to_first_output_seconds == 0.4
    assert result.median_load_time_seconds == 0.6
    assert result.median_prompt_tokens_per_second == 32.0
    assert result.peak_rss_bytes == 4096
    assert result.peak_vram_bytes is None
    assert result.samples[0].prompt_tokens == 16

    payload = result.to_dict()
    assert payload["origins"]["median_load_time_seconds"] == "measured"
    assert payload["origins"]["peak_vram_bytes"] == "unknown"
    assert (
        payload["samples"][0]["origins"]["time_to_first_output_seconds"] == "measured"
    )


def test_runner_still_accepts_plain_token_count(tmp_path: Path):
    result = make_runner().run(
        runtime="llama.cpp",
        model=make_model(tmp_path),
        prompts=["hello world"],
        infer=lambda prompt: 5,
    )

    assert result.median_tokens_per_second > 0
    assert result.median_load_time_seconds is None
    assert result.peak_rss_bytes is None
    assert result.to_dict()["origins"]["median_load_time_seconds"] == "unknown"


def test_runner_rejects_non_positive_token_count(tmp_path: Path):
    try:
        make_runner().run(
            runtime="llama.cpp",
            model=make_model(tmp_path),
            prompts=["hello"],
            infer=lambda prompt: 0,
        )
    except BenchmarkError as exc:
        assert "positive generated-token count" in str(exc)
    else:
        raise AssertionError("expected BenchmarkError")


def test_runner_rejects_invalid_return_type(tmp_path: Path):
    try:
        make_runner().run(
            runtime="llama.cpp",
            model=make_model(tmp_path),
            prompts=["hello"],
            infer=lambda prompt: "42",
        )
    except BenchmarkError as exc:
        assert "int or InferenceTelemetry" in str(exc)
    else:
        raise AssertionError("expected BenchmarkError")


def test_json_and_csv_export_include_telemetry(tmp_path: Path):
    telemetry = InferenceTelemetry(
        generated_tokens=8,
        wall_seconds=2.0,
        time_to_first_output_seconds=0.4,
        peak_rss_bytes=4096,
    )
    result = make_runner().run(
        runtime="llama.cpp",
        model=make_model(tmp_path),
        prompts=["hello world"],
        infer=lambda prompt: telemetry,
    )

    json_path = tmp_path / "run.json"
    csv_path = tmp_path / "run.csv"
    BenchmarkRunner.write_json(result, json_path)
    BenchmarkRunner.write_csv(result, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["median_time_to_first_output_seconds"] == 0.4
    assert payload["peak_rss_bytes"] == 4096
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "time_to_first_output_seconds" in csv_text
    assert "peak_rss_bytes" in csv_text
