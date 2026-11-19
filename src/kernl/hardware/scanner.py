from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import psutil

from kernl.core.types import HardwareProfile, Provenance, ValueOrigin

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_CIM_ADAPTER_RAM_SAFE_MAX = 4 * 1024**3 - 1024**2


def _adapter_ram_bytes(value: Any) -> int | None:
    """Return trustworthy Win32_VideoController.AdapterRAM values.

    AdapterRAM is a 32-bit WMI property. Values close to its 4-GiB ceiling may
    be saturated, so the upper 1-MiB range is deliberately treated as unknown.
    """
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        return None
    try:
        memory_bytes = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if memory_bytes <= 0 or memory_bytes >= _CIM_ADAPTER_RAM_SAFE_MAX:
        return None
    return memory_bytes


def _parse_windows_gpu_details(
    output: str,
) -> tuple[list[str], list[str], list[int | None]]:
    """Parse Win32_VideoController JSON without trusting its shape."""
    try:
        raw: Any = json.loads(output)
    except (TypeError, ValueError):
        return [], [], []

    records = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(record, dict) for record in records):
        return [], [], []

    names: list[str] = []
    drivers: list[str] = []
    memory_bytes: list[int | None] = []
    for record in records:
        name = record.get("Name")
        if not name:
            continue
        names.append(str(name))
        if record.get("DriverVersion"):
            drivers.append(str(record["DriverVersion"]))
        memory_bytes.append(_adapter_ram_bytes(record.get("AdapterRAM")))
    return names, drivers, memory_bytes


def _parse_windows_gpu_json(output: str) -> tuple[list[str], list[str]]:
    """Backward-compatible name and driver parser for Windows adapters."""
    names, drivers, _ = _parse_windows_gpu_details(output)
    return names, drivers


def _find_powershell() -> str | None:
    """Locate Windows PowerShell even when System32 is missing from PATH."""
    for executable in ("powershell", "powershell.exe", "pwsh", "pwsh.exe"):
        on_path = shutil.which(executable)
        if on_path:
            return on_path

    candidates = [
        Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"),
        Path("C:/Windows/SysWOW64/WindowsPowerShell/v1.0/powershell.exe"),
        Path("C:/Program Files/PowerShell/7/pwsh.exe"),
    ]
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.insert(
            0,
            Path(system_root)
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe",
        )
    return next((str(path) for path in candidates if path.is_file()), None)


def _windows_gpu_details(
    runner: CommandRunner = subprocess.run,
) -> tuple[list[str], list[str], list[int | None]]:
    """Read Windows display adapters and WMI-reported memory, if available."""
    if platform.system() != "Windows":
        return [], [], []
    executable = _find_powershell()
    if executable is None:
        return [], [], []
    command = [
        executable,
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress"
        ),
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=8, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return [], [], []
        return _parse_windows_gpu_details(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return [], [], []


def _windows_gpu_info(
    runner: CommandRunner = subprocess.run,
) -> tuple[list[str], list[str]]:
    """Read Windows display adapter names and driver versions."""
    names, drivers, _ = _windows_gpu_details(runner)
    return names, drivers


def _parse_nvidia_smi_memory(output: str) -> dict[str, int]:
    """Parse nvidia-smi memory totals into bytes, keyed by normalized GPU name."""
    memory_by_name: dict[str, int] = {}
    try:
        rows = csv.reader(io.StringIO(output))
        for row in rows:
            if len(row) < 2:
                continue
            name = row[0].strip()
            if not name:
                continue
            try:
                memory_mib = float(row[-1].strip())
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(memory_mib) or memory_mib <= 0:
                continue
            memory_by_name[name.casefold()] = int(memory_mib * 1024**2)
    except (csv.Error, TypeError, ValueError):
        return {}
    return memory_by_name


def _find_nvidia_smi() -> str | None:
    """Locate nvidia-smi from PATH or standard NVIDIA Windows locations."""
    on_path = shutil.which("nvidia-smi")
    if on_path:
        return on_path

    candidates: list[Path] = []
    for variable in ("ProgramW6432", "ProgramFiles"):
        root = os.environ.get(variable)
        if root:
            candidates.append(
                Path(root) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
            )
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(Path(system_root) / "System32" / "nvidia-smi.exe")
    candidates.append(Path("C:/Windows/System32/nvidia-smi.exe"))

    return next((str(path) for path in candidates if path.is_file()), None)


def _nvidia_gpu_memory_info(
    runner: CommandRunner = subprocess.run,
) -> dict[str, int]:
    """Read NVIDIA VRAM using nvidia-smi, which exposes full memory totals."""
    if platform.system() != "Windows":
        return {}
    executable = _find_nvidia_smi()
    if executable is None:
        return {}
    command = [
        executable,
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=8, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        return _parse_nvidia_smi_memory(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return {}


def _resolve_gpu_memory(
    gpu_names: Sequence[str],
    wmi_memory_bytes: Sequence[int | None],
    nvidia_memory_by_name: dict[str, int],
) -> list[int | None]:
    """Prefer exact nvidia-smi values and fall back to trustworthy WMI values."""
    normalized_nvidia = {
        name.strip().casefold(): value for name, value in nvidia_memory_by_name.items()
    }
    resolved: list[int | None] = []
    for index, gpu_name in enumerate(gpu_names):
        nvidia_value = normalized_nvidia.get(gpu_name.strip().casefold())
        wmi_value = wmi_memory_bytes[index] if index < len(wmi_memory_bytes) else None
        resolved.append(nvidia_value if nvidia_value is not None else wmi_value)
    return resolved


def _probe_command(
    executable: str,
    arguments: Sequence[str],
    runner: CommandRunner = subprocess.run,
) -> bool:
    """Return whether an optional capability command executes successfully."""
    resolved_executable = (
        executable if Path(executable).is_file() else shutil.which(executable)
    )
    if resolved_executable is None:
        return False
    try:
        result = runner(
            [resolved_executable, *arguments],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return False


def _detect_backends(runner: CommandRunner = subprocess.run) -> list[str]:
    """Detect available backend signals, not verified inference compatibility."""
    backends = ["cpu"]
    if platform.system() != "Windows":
        return backends
    nvidia_smi = _find_nvidia_smi()
    if nvidia_smi and _probe_command(
        nvidia_smi,
        ["--query-gpu=name", "--format=csv,noheader"],
        runner,
    ):
        backends.append("cuda")
    if _probe_command("vulkaninfo", ["--summary"], runner):
        backends.append("vulkan")
    return backends


def scan() -> HardwareProfile:
    system = platform.system()
    warnings: list[str] = []
    wmi_memory: list[int | None] = []
    nvidia_memory: dict[str, int] = {}
    if system == "Windows":
        gpu_names, drivers, wmi_memory = _windows_gpu_details()
        nvidia_memory = _nvidia_gpu_memory_info()
        gpu_memory = _resolve_gpu_memory(gpu_names, wmi_memory, nvidia_memory)
        warnings.append(
            "Backend probes report available signals; runtime compatibility still requires validation."
        )
    else:
        gpu_names, drivers, gpu_memory = [], [], []
        warnings.append(
            "Running outside Windows; Windows-native GPU probes were skipped."
        )

    known_gpu_memory = any(value is not None for value in gpu_memory)
    normalized_nvidia_memory = {
        name.strip().casefold(): value for name, value in nvidia_memory.items()
    }
    nvidia_used = False
    wmi_used = False
    for index, gpu_name in enumerate(gpu_names):
        normalized_name = gpu_name.strip().casefold()
        if normalized_name in normalized_nvidia_memory:
            nvidia_used = True
        elif index < len(wmi_memory) and wmi_memory[index] is not None:
            wmi_used = True

    memory_sources: list[str] = []
    if nvidia_used:
        memory_sources.append("nvidia-smi")
    if wmi_used:
        memory_sources.append("Win32_VideoController.AdapterRAM")

    if known_gpu_memory:
        gpu_memory_provenance = Provenance[list[int | None]](
            value=gpu_memory,
            origin=ValueOrigin.DETECTED,
            source=" + ".join(memory_sources) or "Windows GPU memory probes",
            warning=(
                "Memory is unknown for one or more adapters."
                if any(value is None for value in gpu_memory)
                else None
            ),
        )
        if any(value is None for value in gpu_memory):
            warnings.append(
                "VRAM could not be detected for one or more display adapters."
            )
    else:
        gpu_memory_provenance = Provenance[list[int | None]](
            value=None,
            origin=ValueOrigin.UNKNOWN,
            source="unavailable",
            warning="VRAM could not be detected; no value is assumed.",
        )
        if system == "Windows" and gpu_names:
            warnings.append(
                "VRAM is unknown. WMI AdapterRAM can saturate at 4 GiB; install or expose nvidia-smi for NVIDIA GPUs."
            )

    root = Path.cwd().anchor or "/"
    memory = psutil.virtual_memory()
    gpu_origin = ValueOrigin.DETECTED if gpu_names else ValueOrigin.UNKNOWN
    backends = _detect_backends()
    return HardwareProfile(
        platform=Provenance(
            value=f"{system} {platform.release()}",
            origin=ValueOrigin.DETECTED,
            source="platform",
        ),
        architecture=Provenance(
            value=platform.machine(),
            origin=ValueOrigin.DETECTED,
            source="platform.machine",
        ),
        cpu_model=Provenance(
            value=platform.processor() or None,
            origin=ValueOrigin.DETECTED
            if platform.processor()
            else ValueOrigin.UNKNOWN,
            source="platform.processor",
        ),
        cpu_cores=Provenance(
            value=psutil.cpu_count(logical=False),
            origin=ValueOrigin.DETECTED,
            source="psutil.cpu_count(logical=False)",
        ),
        cpu_threads=Provenance(
            value=psutil.cpu_count(logical=True),
            origin=ValueOrigin.DETECTED,
            source="psutil.cpu_count(logical=True)",
        ),
        ram_total_bytes=Provenance(
            value=memory.total,
            origin=ValueOrigin.DETECTED,
            source="psutil.virtual_memory",
        ),
        ram_available_bytes=Provenance(
            value=memory.available,
            origin=ValueOrigin.DETECTED,
            source="psutil.virtual_memory",
        ),
        gpu_names=Provenance(
            value=gpu_names or None,
            origin=gpu_origin,
            source="Win32_VideoController" if gpu_names else "unavailable",
        ),
        gpu_driver_versions=Provenance(
            value=drivers or None,
            origin=gpu_origin if drivers else ValueOrigin.UNKNOWN,
            source="Win32_VideoController" if drivers else "unavailable",
        ),
        gpu_memory_bytes=gpu_memory_provenance,
        storage_free_bytes=Provenance(
            value=shutil.disk_usage(root).free,
            origin=ValueOrigin.DETECTED,
            source=f"disk_usage({root})",
        ),
        backends=Provenance(
            value=backends,
            origin=ValueOrigin.DETECTED,
            source="cpu plus Windows capability probes",
        ),
        warnings=warnings,
        scanned_at=dt.datetime.now(dt.UTC).isoformat(),
    )
