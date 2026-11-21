from __future__ import annotations

import subprocess

from kernl.hardware import scanner


def completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["probe"], returncode=returncode, stdout=stdout, stderr=""
    )


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
    monkeypatch.setattr(scanner, "_find_powershell", lambda: "powershell.exe")

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
    monkeypatch.setattr(scanner, "_find_nvidia_smi", lambda: None)
    monkeypatch.setattr(scanner.shutil, "which", lambda executable: None)
    assert scanner._detect_backends() == ["cpu"]


def test_parse_windows_gpu_details_reads_adapter_memory_and_rejects_saturated_values():
    output = (
        '[{"Name":"GPU A","DriverVersion":"1","AdapterRAM":2147483648},'
        '{"Name":"GPU B","DriverVersion":"2","AdapterRAM":4294967295}]'
    )
    names, drivers, memory = scanner._parse_windows_gpu_details(output)
    assert names == ["GPU A", "GPU B"]
    assert drivers == ["1", "2"]
    assert memory == [2 * 1024**3, None]


def test_parse_nvidia_smi_memory_converts_mib_to_bytes():
    output = "NVIDIA GeForce GTX 1650, 4096\nNVIDIA RTX A2000, 6144\n"
    assert scanner._parse_nvidia_smi_memory(output) == {
        "nvidia geforce gtx 1650": 4 * 1024**3,
        "nvidia rtx a2000": 6 * 1024**3,
    }


def test_parse_nvidia_smi_memory_ignores_unknown_and_malformed_rows():
    output = "GPU A, [N/A]\nGPU B, Not Supported\nmalformed\n"
    assert scanner._parse_nvidia_smi_memory(output) == {}


def test_resolve_gpu_memory_prefers_nvidia_and_preserves_unknown_adapters():
    resolved = scanner._resolve_gpu_memory(
        ["NVIDIA GeForce GTX 1650", "Microsoft Basic Display Adapter"],
        [None, None],
        {"nvidia geforce gtx 1650": 4 * 1024**3},
    )
    assert resolved == [4 * 1024**3, None]


def test_windows_gpu_details_requests_adapter_ram(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(scanner, "_find_powershell", lambda: "powershell.exe")
    output = '{"Name":"GPU A","DriverVersion":"1","AdapterRAM":1073741824}'
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return completed(output)

    assert scanner._windows_gpu_details(successful_runner) == (
        ["GPU A"],
        ["1"],
        [1024**3],
    )
    assert "AdapterRAM" in calls[0][-1]


def test_nvidia_gpu_memory_probe_uses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(scanner, "_find_nvidia_smi", lambda: "C:/NVIDIA/nvidia-smi.exe")
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return completed("NVIDIA GeForce GTX 1650, 4096")

    assert scanner._nvidia_gpu_memory_info(successful_runner) == {
        "nvidia geforce gtx 1650": 4 * 1024**3
    }
    assert calls[0][1:] == [
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]


def test_nvidia_gpu_memory_probe_returns_unknown_on_command_failure(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(scanner, "_find_nvidia_smi", lambda: "nvidia-smi")
    assert (
        scanner._nvidia_gpu_memory_info(lambda *args, **kwargs: completed(returncode=1))
        == {}
    )


def test_windows_scan_records_vram_with_detected_provenance(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        scanner,
        "_windows_gpu_details",
        lambda: (["NVIDIA GeForce GTX 1650"], ["552.44"], [None]),
    )
    monkeypatch.setattr(
        scanner,
        "_nvidia_gpu_memory_info",
        lambda: {"nvidia geforce gtx 1650": 4 * 1024**3},
    )
    monkeypatch.setattr(scanner, "_detect_backends", lambda: ["cpu", "cuda"])

    profile = scanner.scan()

    assert profile.gpu_memory_bytes.value == [4 * 1024**3]
    assert profile.gpu_memory_bytes.origin.value == "detected"
    assert profile.gpu_memory_bytes.source == "nvidia-smi"
    assert profile.gpu_memory_bytes.warning is None


def test_adapter_ram_rejects_invalid_and_near_limit_values():
    assert scanner._adapter_ram_bytes(True) is None
    assert scanner._adapter_ram_bytes(1.5) is None
    assert scanner._adapter_ram_bytes(4 * 1024**3 - 1024**2) is None
    assert scanner._adapter_ram_bytes(2 * 1024**3) == 2 * 1024**3


def test_windows_scan_keeps_vram_unknown_when_wmi_is_saturated(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        scanner,
        "_windows_gpu_details",
        lambda: (["AMD Radeon GPU"], ["1"], [None]),
    )
    monkeypatch.setattr(scanner, "_nvidia_gpu_memory_info", dict)
    monkeypatch.setattr(scanner, "_detect_backends", lambda: ["cpu"])

    profile = scanner.scan()

    assert profile.gpu_memory_bytes.value is None
    assert profile.gpu_memory_bytes.origin.value == "unknown"
    assert "VRAM is unknown" in " ".join(profile.warnings)


def test_find_powershell_checks_standard_install_path(monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda executable: None)
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.setattr(
        scanner.Path,
        "is_file",
        lambda path: path.as_posix().endswith("WindowsPowerShell/v1.0/powershell.exe"),
    )

    executable = scanner._find_powershell()

    assert executable is not None
    normalized_path = executable.replace("\\", "/")
    assert normalized_path.endswith("WindowsPowerShell/v1.0/powershell.exe")


def test_find_nvidia_smi_checks_system32_path(monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda executable: None)
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.setattr(
        scanner.Path,
        "is_file",
        lambda path: path.as_posix().endswith("Windows/System32/nvidia-smi.exe"),
    )

    executable = scanner._find_nvidia_smi()

    assert executable is not None
    assert executable.endswith("nvidia-smi.exe")


def test_backend_probe_detects_nvidia_smi_outside_path(monkeypatch):
    monkeypatch.setattr(scanner.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        scanner, "_find_nvidia_smi", lambda: "C:/Windows/System32/nvidia-smi.exe"
    )
    monkeypatch.setattr(
        scanner.Path,
        "is_file",
        lambda path: str(path).endswith("nvidia-smi.exe"),
    )
    monkeypatch.setattr(scanner.shutil, "which", lambda executable: None)

    def successful_runner(command, **kwargs):
        assert command[0].endswith("nvidia-smi.exe")
        return completed("NVIDIA GPU")

    assert scanner._detect_backends(successful_runner) == ["cpu", "cuda"]
