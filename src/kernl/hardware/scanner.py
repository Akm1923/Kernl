from __future__ import annotations
import datetime as dt
import json
import platform
import shutil
import subprocess
from pathlib import Path

import psutil

from kernl.core.types import HardwareProfile, Provenance, ValueOrigin


def _windows_gpu_info() -> tuple[list[str], list[str]]:
    if platform.system() != "Windows":
        return [], []
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return [], []
        raw = json.loads(result.stdout)
        raw = raw if isinstance(raw, list) else [raw]
        return ([str(x["Name"]) for x in raw if x.get("Name")],
                [str(x["DriverVersion"]) for x in raw if x.get("DriverVersion")])
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return [], []


def scan() -> HardwareProfile:
    gpu_names, drivers = _windows_gpu_info()
    warnings: list[str] = []
    if platform.system() != "Windows":
        warnings.append("Running outside Windows; Windows-native GPU probes were skipped.")
    else:
        warnings.append("CUDA/Vulkan capability probing is not enabled in this foundation phase.")
    root = Path.cwd().anchor or "/"
    memory = psutil.virtual_memory()
    gpu_origin = ValueOrigin.DETECTED if gpu_names else ValueOrigin.UNKNOWN
    return HardwareProfile(
        platform=Provenance(value=f"{platform.system()} {platform.release()}", origin=ValueOrigin.DETECTED, source="platform"),
        architecture=Provenance(value=platform.machine(), origin=ValueOrigin.DETECTED, source="platform.machine"),
        cpu_model=Provenance(value=platform.processor() or None, origin=ValueOrigin.DETECTED if platform.processor() else ValueOrigin.UNKNOWN, source="platform.processor"),
        cpu_cores=Provenance(value=psutil.cpu_count(logical=False), origin=ValueOrigin.DETECTED, source="psutil.cpu_count(logical=False)"),
        cpu_threads=Provenance(value=psutil.cpu_count(logical=True), origin=ValueOrigin.DETECTED, source="psutil.cpu_count(logical=True)"),
        ram_total_bytes=Provenance(value=memory.total, origin=ValueOrigin.DETECTED, source="psutil.virtual_memory"),
        ram_available_bytes=Provenance(value=memory.available, origin=ValueOrigin.DETECTED, source="psutil.virtual_memory"),
        gpu_names=Provenance(value=gpu_names or None, origin=gpu_origin, source="Win32_VideoController" if gpu_names else "unavailable"),
        gpu_driver_versions=Provenance(value=drivers or None, origin=gpu_origin, source="Win32_VideoController" if drivers else "unavailable"),
        storage_free_bytes=Provenance(value=shutil.disk_usage(root).free, origin=ValueOrigin.DETECTED, source=f"disk_usage({root})"),
        backends=Provenance(value=None, origin=ValueOrigin.UNKNOWN, source="not probed in foundation phase"),
        warnings=warnings,
        scanned_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
