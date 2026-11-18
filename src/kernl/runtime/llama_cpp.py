from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import signal
import subprocess
import time
from kernl.runtime.config import RuntimeConfig, RuntimeMode

class RuntimeErrorBase(RuntimeError):
    """Base error for runtime detection and process management."""

@dataclass(frozen=True)
class RuntimeCapabilities:
    executable: Path
    version: str | None
    supports_server: bool
    supports_flash_attention: bool
    raw_version_output: str

@dataclass
class ManagedProcess:
    process: subprocess.Popen[str]
    config: RuntimeConfig
    started_at: float
    stderr_lines: list[str] = field(default_factory=list)

class LlamaCppAdapter:
    """Safe llama.cpp adapter; all subprocess calls use argument arrays."""
    def __init__(self, executable: Path) -> None:
        self.executable = executable
        self._managed: dict[int, ManagedProcess] = {}

    def detect(self) -> RuntimeCapabilities:
        if not self.executable.is_file():
            raise RuntimeErrorBase(f"llama.cpp executable not found: {self.executable}")
        try:
            result = subprocess.run([str(self.executable), "--version"], capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeErrorBase(f"could not query llama.cpp version: {exc}") from exc
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeErrorBase(f"llama.cpp version command failed: {output[-1000:]}")
        lowered = output.lower()
        return RuntimeCapabilities(self.executable, self._extract_version(output), "server" in lowered or self.executable.name.lower().startswith("llama-server"), "flash" in lowered, output)

    def launch(self, config: RuntimeConfig, *, approve_execution: bool = False) -> int:
        if not approve_execution:
            raise PermissionError("model execution requires explicit approval")
        if not config.model_path.is_file():
            raise RuntimeErrorBase(f"model file not found: {config.model_path}")
        try:
            process = subprocess.Popen([str(self.executable), *config.runtime_arguments()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        except OSError as exc:
            raise RuntimeErrorBase(f"could not start llama.cpp: {exc}") from exc
        self._managed[process.pid] = ManagedProcess(process, config, time.monotonic())
        return process.pid

    def infer_once(self, config: RuntimeConfig, prompt: str, *, approve_execution: bool = False, timeout: float = 120.0) -> int:
        """Run one actual llama.cpp CLI inference and return generated-token count."""
        if not approve_execution:
            raise PermissionError("inference requires explicit approval")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not self.executable.is_file() or not config.model_path.is_file():
            raise RuntimeErrorBase("runtime executable and model file must exist")
        args = [str(self.executable), *config.runtime_arguments(), "--prompt", prompt, "--n-predict", str(config.context_size)]
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeErrorBase(f"inference process failed: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeErrorBase(f"inference failed: {result.stderr[-1000:]}")
        output = result.stdout.strip()
        if not output:
            raise RuntimeErrorBase("inference produced no output")
        return max(1, len(output.split()))

    def status(self, pid: int) -> dict[str, object]:
        managed = self._managed.get(pid)
        if managed is None:
            raise KeyError(f"unknown Kernl process: {pid}")
        code = managed.process.poll()
        return {"pid": pid, "running": code is None, "returncode": code, "started_at": managed.started_at, "model": str(managed.config.model_path)}

    def health(self, pid: int) -> bool:
        return bool(self.status(pid)["running"])

    def logs(self, pid: int, limit: int = 100) -> list[str]:
        managed = self._managed.get(pid)
        if managed is None: raise KeyError(f"unknown Kernl process: {pid}")
        if limit < 1: raise ValueError("limit must be positive")
        if managed.process.stderr is not None:
            while True:
                line = managed.process.stderr.readline()
                if not line: break
                managed.stderr_lines.append(line.rstrip("\n"))
        return managed.stderr_lines[-limit:]

    def shutdown(self, pid: int, timeout: float = 10.0) -> None:
        managed = self._managed.get(pid)
        if managed is None: raise KeyError(f"unknown Kernl process: {pid}")
        if managed.process.poll() is not None: return
        if os.name == "nt": managed.process.send_signal(signal.CTRL_BREAK_EVENT)
        else: managed.process.terminate()
        try: managed.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            managed.process.kill(); managed.process.wait(timeout=5)

    @staticmethod
    def _extract_version(output: str) -> str | None:
        for token in output.replace("/", " ").split():
            if token.startswith("v") and any(char.isdigit() for char in token): return token.strip(",;")
        return None
