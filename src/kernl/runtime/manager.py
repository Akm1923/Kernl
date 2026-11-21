from __future__ import annotations

import os
import re
import signal
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import psutil

from kernl.runtime.config import RuntimeConfig
from kernl.runtime.llama_cpp import LlamaCppAdapter
from kernl.runtime.process_store import ProcessStore, RuntimeProcessRecord


class ProcessError(RuntimeError):
    """Raised when a managed runtime process cannot be operated safely."""


class RuntimeProcessManager:
    """Launch, inspect, stop and restart runtime processes across CLI runs."""

    def __init__(
        self,
        store: ProcessStore,
        log_dir: Path,
        *,
        adapter_factory: Callable[[Path], LlamaCppAdapter] = LlamaCppAdapter,
        process_factory: Callable[[int], Any] = psutil.Process,
    ) -> None:
        self.store = store
        self.log_dir = log_dir
        self._adapter_factory = adapter_factory
        self._process_factory = process_factory

    def launch(
        self,
        *,
        config_name: str,
        config: RuntimeConfig,
        executable: Path,
        approve_execution: bool = False,
        extra_args: Sequence[str] = (),
    ) -> RuntimeProcessRecord:
        if not approve_execution:
            raise PermissionError("model execution requires explicit approval")
        if not config_name.strip():
            raise ValueError("config_name cannot be empty")
        for existing in self.list():
            if existing.config_name == config_name and existing.status == "running":
                raise ProcessError(
                    f"configuration already running with PID {existing.pid}"
                )

        log_path = self.log_dir / (
            f"{_safe_name(config_name)}-{int(time.time() * 1000)}.log"
        )
        adapter = self._adapter_factory(executable)
        pid = adapter.launch(
            config,
            approve_execution=True,
            log_path=log_path,
            extra_args=extra_args,
        )
        record = RuntimeProcessRecord(
            pid=pid,
            config_name=config_name,
            executable=str(executable),
            model_path=str(config.model_path),
            started_at=time.time(),
            log_path=str(log_path),
            status="running",
        )
        self.store.add(record)
        return record

    def list(self) -> list[RuntimeProcessRecord]:
        return [self.status(record.pid) for record in self.store.list()]

    def status(self, pid: int) -> RuntimeProcessRecord:
        record = self.store.get(pid)
        if record is None:
            raise KeyError(f"unknown managed process: {pid}")
        if record.status != "running":
            return record

        process = self._find_process(pid)
        if process is None:
            return self.store.update(pid, status="stopped")
        if not self._matches(record, process):
            return self.store.update(pid, status="foreign")
        return record

    def logs(self, pid: int, limit: int = 100) -> list[str]:
        if limit < 1:
            raise ValueError("limit must be positive")
        record = self.status(pid)
        path = Path(record.log_path)
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-limit:]

    def stop(self, pid: int) -> RuntimeProcessRecord:
        record = self.status(pid)
        if record.status == "stopped":
            raise ProcessError(f"process {pid} is already stopped")
        if record.status == "foreign":
            raise ProcessError(
                f"PID {pid} is not a matching managed runtime process; refusing to stop it"
            )

        process = self._find_process(pid)
        if process is None:
            return self.store.update(pid, status="stopped")
        if not self._matches(record, process):
            return self.store.update(pid, status="foreign")
        self._terminate(process)
        return self.store.update(pid, status="stopped")

    def restart(
        self,
        pid: int,
        *,
        config: RuntimeConfig,
        executable: Path,
        approve_execution: bool = False,
        extra_args: Sequence[str] = (),
    ) -> RuntimeProcessRecord:
        record = self.status(pid)
        if record.status == "running":
            self.stop(pid)
        return self.launch(
            config_name=record.config_name,
            config=config,
            executable=executable,
            approve_execution=approve_execution,
            extra_args=extra_args,
        )

    def _find_process(self, pid: int) -> Any | None:
        try:
            return self._process_factory(pid)
        except (psutil.Error, OSError, ValueError):
            return None

    @staticmethod
    def _matches(record: RuntimeProcessRecord, process: Any) -> bool:
        expected = Path(record.executable).name.casefold()
        try:
            names = [process.name().casefold()]
            cmdline = " ".join(process.cmdline()).casefold()
        except (psutil.Error, OSError):
            return False
        return any(expected in name for name in names) or expected in cmdline

    @staticmethod
    def _terminate(process: Any) -> None:
        try:
            if not process.is_running():
                return
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            process.wait(timeout=10)
        except (psutil.Error, OSError):
            pass
        try:
            if process.is_running():
                process.kill()
                process.wait(timeout=5)
        except (psutil.Error, OSError):
            pass


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "config"
