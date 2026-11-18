from __future__ import annotations

import subprocess

from kernl.hardware import scanner


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["probe"], returncode=returncode, stdout=stdout, stderr="")


def test_parse_windows_gpu_json_supports_single_record():
    names, drivers = scanner._parse_windows_gpu_json(
        '{"Name":"NVIDIA RTX 4090","DriverVersion":"31.0.15.5123"}'
    )
    assert names == ["NVIDIA RTX 4090"]
    assert drivers == ["31.0.15.5123"]


def test_parse_windows_gpu_json_supports_multiple_records():
    names, drivers = scanner._parse_windows_gpu_json(
        '[{"Name":"GPU A","DriverVersion":"1"},{"Name":"GPU B"}]'
    )
    assert names == ["GPU A", "GPU B"]
    assert drivers == ["1"]


def test_parse_windows_gpu_json_rejects_malformed_output():
    assert scanner._parse_windows_gpu_json("not json") == ([], [])
    assert scanner._parse_windows_gpu_json("[]") == ([], [])


def test_windows_gpu_probe_handles_command_failure(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")

    def failed_runner(*args, **kwargs):
        return completed(returncode=1)

    assert scanner._windows_gpu_info(failed_runner) == ([], [])


def test_backend_probe_reports_cpu_only_outside_windows(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Linux")
    assert scanner._detect_backends() == ["cpu"]


def test_backend_probe_detects_cuda_and_vulkan(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(scanner.shutil, "which", lambda executable: executable)

    def successful_runner(command, **kwargs):
        return completed("ok")

    assert scanner._detect_backends(successful_runner) == ["cpu", "cuda", "vulkan"]


def test_backend_probe_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(scanner.shutil, "which", lambda executable: None)
    assert scanner._detect_backends() == ["cpu"]
