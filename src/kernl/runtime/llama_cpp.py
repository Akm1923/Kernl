from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from kernl.core.telemetry import InferenceTelemetry
from kernl.runtime.config import RuntimeConfig


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
    log_path: Path | None = None


class LlamaCppAdapter:
    """Safe llama.cpp adapter; all subprocess calls use argument arrays."""

    def __init__(self, executable: Path) -> None:
        self.executable = executable
        self._managed: dict[int, ManagedProcess] = {}

    def detect(self) -> RuntimeCapabilities:
        if not self.executable.is_file():
            raise RuntimeErrorBase(f"llama.cpp executable not found: {self.executable}")
        try:
            result = subprocess.run(
                [str(self.executable), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeErrorBase(f"could not query llama.cpp version: {exc}") from exc
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeErrorBase(
                f"llama.cpp version command failed: {output[-1000:]}"
            )
        lowered = output.lower()
        return RuntimeCapabilities(
            self.executable,
            self._extract_version(output),
            "server" in lowered
            or self.executable.name.lower().startswith("llama-server"),
            "flash" in lowered,
            output,
        )

    def launch(
        self,
        config: RuntimeConfig,
        *,
        approve_execution: bool = False,
        log_path: Path | None = None,
        extra_args: Sequence[str] = (),
    ) -> int:
        if not approve_execution:
            raise PermissionError("model execution requires explicit approval")
        if not self.executable.is_file():
            raise RuntimeErrorBase(f"llama.cpp executable not found: {self.executable}")
        if not config.model_path.is_file():
            raise RuntimeErrorBase(f"model file not found: {config.model_path}")
        arguments = [str(self.executable), *config.runtime_arguments(), *extra_args]
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            if log_path is None:
                process = subprocess.Popen(
                    arguments,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    creationflags=creation_flags,
                )
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as log_handle:
                    process = subprocess.Popen(
                        arguments,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=creation_flags,
                    )
        except OSError as exc:
            raise RuntimeErrorBase(f"could not start llama.cpp: {exc}") from exc
        self._managed[process.pid] = ManagedProcess(
            process, config, time.monotonic(), log_path=log_path
        )
        return process.pid

    def infer_once(
        self,
        config: RuntimeConfig,
        prompt: str,
        *,
        approve_execution: bool = False,
        timeout: float = 120.0,
    ) -> int:
        """Run one actual llama.cpp CLI inference and return generated-token count."""
        return self.infer_once_telemetry(
            config,
            prompt,
            approve_execution=approve_execution,
            timeout=timeout,
        ).generated_tokens

    def infer_once_telemetry(
        self,
        config: RuntimeConfig,
        prompt: str,
        *,
        approve_execution: bool = False,
        timeout: float = 120.0,
        sample_interval: float = 0.05,
    ) -> InferenceTelemetry:
        """Run one real llama.cpp inference and return observed telemetry.

        Every returned value comes from this run: stdout timing, runtime stderr
        timings, and psutil/nvidia-smi sampling. Signals that cannot be observed
        stay ``None`` and are reported as unknown, never estimated.
        """
        if not approve_execution:
            raise PermissionError("inference requires explicit approval")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        if not self.executable.is_file() or not config.model_path.is_file():
            raise RuntimeErrorBase("runtime executable and model file must exist")
        args = [
            str(self.executable),
            *config.runtime_arguments(),
            "--prompt",
            prompt,
            "--n-predict",
            str(config.context_size),
        ]
        started = time.perf_counter()
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise RuntimeErrorBase(f"inference process failed: {exc}") from exc

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        first_output_at: float | None = None
        peak_rss: int | None = None
        peak_vram: int | None = None
        stop_sampler = threading.Event()

        def read_stdout() -> None:
            nonlocal first_output_at
            assert process.stdout is not None
            for line in process.stdout:
                if first_output_at is None:
                    first_output_at = time.perf_counter() - started
                stdout_lines.append(line)

        def read_stderr() -> None:
            assert process.stderr is not None
            stderr_lines.extend(line for line in process.stderr)

        def sample_resources() -> None:
            nonlocal peak_rss, peak_vram
            tick = 0
            while not stop_sampler.is_set():
                try:
                    rss = psutil.Process(process.pid).memory_info().rss
                    if peak_rss is None or rss > peak_rss:
                        peak_rss = rss
                except (psutil.Error, OSError, ValueError):
                    pass
                if tick % 10 == 0:
                    vram = _sample_vram_bytes(process.pid)
                    if vram is not None and (peak_vram is None or vram > peak_vram):
                        peak_vram = vram
                tick += 1
                stop_sampler.wait(sample_interval)

        threads = [
            threading.Thread(target=read_stdout, daemon=True),
            threading.Thread(target=read_stderr, daemon=True),
            threading.Thread(target=sample_resources, daemon=True),
        ]
        for worker in threads:
            worker.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeErrorBase(
                f"inference timed out after {timeout:g} seconds"
            ) from exc
        finally:
            stop_sampler.set()
            for worker in threads:
                worker.join(timeout=5)
        wall_seconds = time.perf_counter() - started

        if process.returncode != 0:
            raise RuntimeErrorBase(f"inference failed: {''.join(stderr_lines)[-1000:]}")
        output = "".join(stdout_lines).strip()
        if not output:
            raise RuntimeErrorBase("inference produced no output")
        generated = max(1, len(output.split()))
        timings = _parse_runtime_timings("".join(stderr_lines))
        prompt_tokens_per_second = timings["prompt_tokens_per_second"]
        if (
            prompt_tokens_per_second is None
            and timings["prompt_eval_seconds"]
            and timings["prompt_tokens"]
        ):
            prompt_tokens_per_second = (
                timings["prompt_tokens"] / timings["prompt_eval_seconds"]
                if timings["prompt_eval_seconds"] > 0
                else None
            )
        generation_tokens_per_second = timings["generation_tokens_per_second"]
        if (
            generation_tokens_per_second is None
            and timings["generation_seconds"]
            and timings["generation_tokens"]
        ):
            generation_tokens_per_second = (
                timings["generation_tokens"] / timings["generation_seconds"]
                if timings["generation_seconds"] > 0
                else None
            )
        return InferenceTelemetry(
            generated_tokens=generated,
            wall_seconds=wall_seconds,
            time_to_first_output_seconds=first_output_at,
            load_time_seconds=timings["load_time_seconds"],
            prompt_eval_seconds=timings["prompt_eval_seconds"],
            prompt_tokens=timings["prompt_tokens"],
            prompt_tokens_per_second=prompt_tokens_per_second,
            generation_tokens_per_second=generation_tokens_per_second,
            peak_rss_bytes=peak_rss,
            peak_vram_bytes=peak_vram,
        )

    def status(self, pid: int) -> dict[str, object]:
        managed = self._managed.get(pid)
        if managed is None:
            raise KeyError(f"unknown Kernl process: {pid}")
        code = managed.process.poll()
        return {
            "pid": pid,
            "running": code is None,
            "returncode": code,
            "started_at": managed.started_at,
            "model": str(managed.config.model_path),
        }

    def health(self, pid: int) -> bool:
        return bool(self.status(pid)["running"])

    def logs(self, pid: int, limit: int = 100) -> list[str]:
        managed = self._managed.get(pid)
        if managed is None:
            raise KeyError(f"unknown Kernl process: {pid}")
        if limit < 1:
            raise ValueError("limit must be positive")
        if managed.log_path is not None and managed.log_path.is_file():
            lines = managed.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            return lines[-limit:]
        if managed.process.stderr is not None:
            while True:
                line = managed.process.stderr.readline()
                if not line:
                    break
                managed.stderr_lines.append(line.rstrip("\n"))
        return managed.stderr_lines[-limit:]

    def shutdown(self, pid: int, timeout: float = 10.0) -> None:
        managed = self._managed.get(pid)
        if managed is None:
            raise KeyError(f"unknown Kernl process: {pid}")
        if managed.process.poll() is not None:
            return
        if os.name == "nt":
            managed.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            managed.process.terminate()
        try:
            managed.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            managed.process.kill()
            managed.process.wait(timeout=5)

    @staticmethod
    def _extract_version(output: str) -> str | None:
        for token in output.replace("/", " ").split():
            if token.startswith("v") and any(char.isdigit() for char in token):
                return token.strip(",;")
        return None


_LOAD_TIME_RE = re.compile(r"load time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
_PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"
    r"(?:\s*/\s*([0-9]+)\s*tokens?)?"
    r"(?:[^()\r\n]*\(\s*[^,)]*,\s*([0-9]+(?:\.[0-9]+)?)\s*tokens per second\))?",
    re.IGNORECASE,
)
_EVAL_TIME_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"
    r"(?:\s*/\s*([0-9]+)\s*tokens?)?"
    r"(?:[^()\r\n]*\(\s*[^,)]*,\s*([0-9]+(?:\.[0-9]+)?)\s*tokens per second\))?",
    re.IGNORECASE,
)


def _parse_runtime_timings(stderr_text: str) -> dict[str, float | int | None]:
    """Parse llama.cpp's own timing report; missing signals stay ``None``."""
    timings: dict[str, float | int | None] = {
        "load_time_seconds": None,
        "prompt_eval_seconds": None,
        "prompt_tokens": None,
        "prompt_tokens_per_second": None,
        "generation_seconds": None,
        "generation_tokens": None,
        "generation_tokens_per_second": None,
    }
    if match := _LOAD_TIME_RE.search(stderr_text):
        timings["load_time_seconds"] = float(match.group(1)) / 1000.0
    if match := _PROMPT_EVAL_RE.search(stderr_text):
        timings["prompt_eval_seconds"] = float(match.group(1)) / 1000.0
        if match.group(2) is not None:
            timings["prompt_tokens"] = int(match.group(2))
        if match.group(3) is not None:
            timings["prompt_tokens_per_second"] = float(match.group(3))
    if match := _EVAL_TIME_RE.search(stderr_text):
        timings["generation_seconds"] = float(match.group(1)) / 1000.0
        if match.group(2) is not None:
            timings["generation_tokens"] = int(match.group(2))
        if match.group(3) is not None:
            timings["generation_tokens_per_second"] = float(match.group(3))
    return timings


def _sample_vram_bytes(pid: int) -> int | None:
    """Sample this process's VRAM through nvidia-smi; ``None`` when unavailable."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        result = subprocess.run(
            [
                smi,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    total_mib = 0.0
    matched = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        if int(parts[0]) != pid:
            continue
        try:
            total_mib += float(parts[1])
        except ValueError:
            continue
        matched = True
    if not matched:
        return None
    return int(total_mib * 1024 * 1024)
