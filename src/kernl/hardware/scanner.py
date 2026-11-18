from __future__ import annotations

import datetime as dt
import json
import platform
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import psutil

from kernl.core.types import HardwareProfile, Provenance, ValueOrigin


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _parse_windows_gpu_json(output: str) -> tuple[list[str], list[str]]:
    """Parse Win32_VideoController JSON without trusting its shape."""
    try:
        raw: Any = json.loads(output)
    except (TypeError, ValueError):
        return [], []
    records = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(record, dict) for record in records):
        return [], []
    names = [str(record["Name"]) for record in records if record.get("Name")]
    drivers = [str(record["DriverVersion"]) for record in records if record.get("DriverVersion")]
    return names, drivers


def _windows_gpu_info(runner: CommandRunner = subprocess.run) -> tuple[list[str], list[str]]:
    """Read Windows display adapters through PowerShell CIM, if available."""
    if platform.system() != "Windows":
        return [], []
    command = [
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress",
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=8, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return [], []
        return _parse_windows_gpu_json(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return [], []


def _probe_command(executable: str, arguments: Sequence[str], runner: CommandRunner = subprocess.run) -> bool:
    """Return whether an optional capability command executes successfully."""
    if shutil.which(executable) is None:
        return False
    try:
        result = runner([executable, *arguments], capture_output=True, text=True, timeout=8, check=False)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return False


def _detect_backends(runner: CommandRunner = subprocess.run) -> list[str]:
    """Detect usable backend signals without claiming an actual inference test."""
    backends = ["cpu"]
    if platform.system() != "Windows":
        return backends
    if _probe_command("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"], runner):
        backends.append("cuda")
    if _probe_command("vulkaninfo", ["--summary"], runner):
        backends.append("vulkan")
    return backends


def scan() -> HardwareProfile:
    gpu_names, drivers = _windows_gpu_info()
    warnings: list[str] = []
    system = platform.system()
    if system != "Windows":
        warnings.append("Running outside Windows; Windows-native GPU probes were skipped.")
    else:
        warnings.append("Backend probes report available signals; runtime compatibility still requires validation.")
    root = Path.cwd().anchor or "/"
    memory = psutil.virtual_memory()
    gpu_origin = ValueOrigin.DETECTED if gpu_names else ValueOrigin.UNKNOWN
    backends = _detect_backends()
    return HardwareProfile(
        platform=Provenance(value=f"{system} {platform.release()}", origin=ValueOrigin.DETECTED, source="platform"),
        architecture=Provenance(value=platform.machine(), origin=ValueOrigin.DETECTED, source="platform.machine"),
        cpu_model=Provenance(value=platform.processor() or None, origin=ValueOrigin.DETECTED if platform.processor() else ValueOrigin.UNKNOWN, source="platform.processor"),
        cpu_cores=Provenance(value=psutil.cpu_count(logical=False), origin=ValueOrigin.DETECTED, source="psutil.cpu_count(logical=False)"),
        cpu_threads=Provenance(value=psutil.cpu_count(logical=True), origin=ValueOrigin.DETECTED, source="psutil.cpu_count(logical=True)"),
        ram_total_bytes=Provenance(value=memory.total, origin=ValueOrigin.DETECTED, source="psutil.virtual_memory"),
        ram_available_bytes=Provenance(value=memory.available, origin=ValueOrigin.DETECTED, source="psutil.virtual_memory"),
        gpu_names=Provenance(value=gpu_names or None, origin=gpu_origin, source="Win32_VideoController" if gpu_names else "unavailable"),
        gpu_driver_versions=Provenance(value=drivers or None, origin=gpu_origin, source="Win32_VideoController" if drivers else "unavailable"),
        storage_free_bytes=Provenance(value=shutil.disk_usage(root).free, origin=ValueOrigin.DETECTED, source=f"disk_usage({root})"),
        backends=Provenance(value=backends, origin=ValueOrigin.DETECTED, source="cpu plus Windows capability probes"),
        warnings=warnings,
        scanned_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
