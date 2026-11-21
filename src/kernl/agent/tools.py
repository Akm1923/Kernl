from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from kernl.core.types import ValueOrigin
from kernl.hardware.scanner import scan
from kernl.models.gguf import GGUFReader
from kernl.models.planner import plan_model_memory
from kernl.runtime.config_store import ConfigStore
from kernl.runtime.llama_cpp import LlamaCppAdapter, RuntimeErrorBase
from kernl.storage.db import Store


def _origin(value: ValueOrigin) -> str:
    return value.value


@dataclass
class AgentContext:
    """Dependencies available to deterministic agent tools."""

    store: Store
    config_dir: Path
    workspace_root: Path
    executable_resolver: Callable[[str], str | None] | None = None


def _resolve_executable(ctx: AgentContext, executable: str) -> str | None:
    resolver = ctx.executable_resolver or shutil.which
    return resolver(executable)


class HardwareProfileInput(BaseModel):
    pass


class RuntimeStatusInput(BaseModel):
    executable: str = "llama-cli"

    @field_validator("executable")
    @classmethod
    def bare_command_name(cls, value: str) -> str:
        if not value.strip() or Path(value).name != value:
            raise ValueError("executable must be a bare command name")
        return value


class LocalModelInput(BaseModel):
    path: str = Field(min_length=1)


class LocalModelPlanInput(LocalModelInput):
    gpu_layers: int = Field(default=0, ge=-1, le=10_000)
    context_tokens: int | None = Field(default=None, ge=1)
    kv_cache_on_cpu: bool = False


class BenchmarkHistoryInput(BaseModel):
    config_name: str = Field(min_length=1)


class ScanInput(BaseModel):
    pass


class RuntimeVerifyInput(BaseModel):
    config_name: str = Field(min_length=1)
    executable: str = "llama-cli"
    prompt: str = "Hello"
    timeout: float = Field(default=120.0, gt=0)

    @field_validator("executable")
    @classmethod
    def bare_command_name(cls, value: str) -> str:
        if not value.strip() or Path(value).name != value:
            raise ValueError("executable must be a bare command name")
        return value


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    input_model: type[BaseModel]
    requires_approval: bool
    handler: Callable[[BaseModel, AgentContext], dict[str, object]]


def _workspace_path(raw: str, root: Path) -> Path:
    candidate = Path(raw).expanduser()
    workspace = root.resolve()
    resolved = (
        candidate if candidate.is_absolute() else workspace / candidate
    ).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("path must stay inside the current workspace")
    return resolved


def _hardware_profile(_: BaseModel, ctx: AgentContext) -> dict[str, object]:
    profile = ctx.store.latest_profile()
    if profile is None:
        return {
            "origin": _origin(ValueOrigin.UNKNOWN),
            "status": "unknown",
            "reason": "No saved hardware profile; run the scan tool or kernl scan.",
        }
    return {
        "origin": _origin(ValueOrigin.DETECTED),
        "platform": profile.platform.value,
        "ram_available_bytes": profile.ram_available_bytes.value,
        "gpu_names": profile.gpu_names.value,
        "gpu_memory_bytes": profile.gpu_memory_bytes.value,
    }


def _runtime_status(args: BaseModel, ctx: AgentContext) -> dict[str, object]:
    assert isinstance(args, RuntimeStatusInput)
    found = _resolve_executable(ctx, args.executable)
    return {
        "origin": _origin(ValueOrigin.DETECTED if found else ValueOrigin.UNKNOWN),
        "executable": args.executable,
        "resolved_path": found,
    }


def _local_model_inspect(args: BaseModel, ctx: AgentContext) -> dict[str, object]:
    assert isinstance(args, LocalModelInput)
    path = _workspace_path(args.path, ctx.workspace_root)
    if not path.is_file():
        raise ValueError(f"model file not found: {path}")
    metadata = GGUFReader().read(path)
    return {
        "origin": _origin(ValueOrigin.DETECTED),
        "path": str(path),
        "architecture": metadata.architecture,
        "parameter_count": metadata.parameter_count,
        "context_length": metadata.context_length,
        "block_count": metadata.block_count,
        "file_size_bytes": metadata.file_size_bytes,
    }


def _local_model_plan(args: BaseModel, ctx: AgentContext) -> dict[str, object]:
    assert isinstance(args, LocalModelPlanInput)
    path = _workspace_path(args.path, ctx.workspace_root)
    metadata = GGUFReader().read(path)
    profile = ctx.store.latest_profile()
    if profile is None:
        return {
            "origin": _origin(ValueOrigin.UNKNOWN),
            "status": "unknown",
            "reason": "No saved hardware profile; run the scan tool or kernl scan.",
        }
    plan = plan_model_memory(
        metadata,
        profile,
        gpu_layers=args.gpu_layers,
        context_tokens=args.context_tokens,
        kv_cache_on_gpu=not args.kv_cache_on_cpu,
    )
    return {
        "origin": _origin(ValueOrigin.ESTIMATED),
        "cpu_required_bytes": plan.cpu_required_bytes,
        "cpu_fits": plan.cpu_fits,
        "gpu_name": plan.gpu_name,
        "gpu_required_bytes": plan.gpu_required_bytes,
        "gpu_fits": plan.gpu_fits,
        "overall_compatible": plan.overall_compatible,
        "warnings": list(plan.warnings),
    }


def _benchmark_history(args: BaseModel, ctx: AgentContext) -> dict[str, object]:
    assert isinstance(args, BenchmarkHistoryInput)
    record = ctx.store.latest_benchmark(args.config_name)
    if record is None:
        return {
            "origin": _origin(ValueOrigin.UNKNOWN),
            "status": "unknown",
            "reason": "No stored benchmark for this configuration.",
        }
    return {"origin": _origin(ValueOrigin.MEASURED), **record}


def _scan(_: BaseModel, ctx: AgentContext) -> dict[str, object]:
    profile = scan()
    ctx.store.save_profile(profile)
    return {
        "origin": _origin(ValueOrigin.DETECTED),
        "platform": profile.platform.value,
        "gpu_names": profile.gpu_names.value,
        "warnings": list(profile.warnings),
    }


def _runtime_verify(args: BaseModel, ctx: AgentContext) -> dict[str, object]:
    assert isinstance(args, RuntimeVerifyInput)
    config = ConfigStore(ctx.config_dir).load(args.config_name)
    resolved = _resolve_executable(ctx, args.executable)
    if not resolved:
        raise RuntimeErrorBase(f"runtime executable not found: {args.executable}")
    tokens = LlamaCppAdapter(Path(resolved)).infer_once(
        config, args.prompt, approve_execution=True, timeout=args.timeout
    )
    return {
        "origin": _origin(ValueOrigin.MEASURED),
        "generated_tokens": tokens,
        "source": "observed llama.cpp inference",
    }


TOOLS: dict[str, AgentTool] = {
    "hardware_profile": AgentTool(
        "hardware_profile",
        "Show the most recently saved hardware profile (read-only).",
        HardwareProfileInput,
        False,
        _hardware_profile,
    ),
    "runtime_status": AgentTool(
        "runtime_status",
        "Check whether a named llama.cpp executable is on PATH (read-only).",
        RuntimeStatusInput,
        False,
        _runtime_status,
    ),
    "local_model_inspect": AgentTool(
        "local_model_inspect",
        "Inspect GGUF metadata for a file inside the workspace (read-only).",
        LocalModelInput,
        False,
        _local_model_inspect,
    ),
    "local_model_plan": AgentTool(
        "local_model_plan",
        "Estimate model memory from a workspace GGUF file (estimated, read-only).",
        LocalModelPlanInput,
        False,
        _local_model_plan,
    ),
    "benchmark_history": AgentTool(
        "benchmark_history",
        "Show the latest stored measurement for a configuration (read-only).",
        BenchmarkHistoryInput,
        False,
        _benchmark_history,
    ),
    "scan": AgentTool(
        "scan",
        "Run a real hardware scan and save the resulting profile.",
        ScanInput,
        True,
        _scan,
    ),
    "runtime_verify": AgentTool(
        "runtime_verify",
        "Run one approved llama.cpp inference against a saved configuration.",
        RuntimeVerifyInput,
        True,
        _runtime_verify,
    ),
}
