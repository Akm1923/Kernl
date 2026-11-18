from __future__ import annotations
from enum import StrEnum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class Backend(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    VULKAN = "vulkan"
    METAL = "metal"


class RuntimeMode(StrEnum):
    CLI = "cli"
    SERVER = "server"


class RuntimeConfig(BaseModel):
    """Single validated configuration object used for display/export/execution."""

    model_path: Path
    backend: Backend = Backend.AUTO
    mode: RuntimeMode = RuntimeMode.CLI
    context_size: int = Field(default=4096, ge=128, le=1_048_576)
    batch_size: int = Field(default=512, ge=1, le=65_536)
    microbatch_size: int = Field(default=128, ge=1, le=65_536)
    threads: int | None = Field(default=None, ge=1, le=1024)
    gpu_layers: int = Field(default=0, ge=-1, le=10_000)
    flash_attention: bool = False
    kv_cache_type: str = "f16"
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int = 0

    @field_validator("model_path")
    @classmethod
    def model_path_must_be_file(cls, value: Path) -> Path:
        # Path existence is checked at execution time so configs can be exported before download.
        if value.name in {"", ".", ".."}:
            raise ValueError("model_path must point to a model file")
        return value

    @field_validator("microbatch_size")
    @classmethod
    def microbatch_not_larger_than_batch(cls, value: int, info):
        batch = info.data.get("batch_size", 512)
        if value > batch:
            raise ValueError("microbatch_size cannot exceed batch_size")
        return value

    @field_validator("kv_cache_type")
    @classmethod
    def supported_kv_cache(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"f16", "f32", "q8_0", "q4_0"}:
            raise ValueError("kv_cache_type must be one of f16, f32, q8_0, q4_0")
        return normalized

    def runtime_arguments(self) -> list[str]:
        """Return deterministic llama.cpp-style arguments; no shell interpolation."""
        args = ["--model", str(self.model_path), "--ctx-size", str(self.context_size), "--batch-size", str(self.batch_size), "--ubatch-size", str(self.microbatch_size), "--gpu-layers", str(self.gpu_layers), "--cache-type-k", self.kv_cache_type, "--cache-type-v", self.kv_cache_type, "--temp", str(self.temperature), "--top-p", str(self.top_p), "--seed", str(self.seed)]
        if self.threads is not None:
            args.extend(["--threads", str(self.threads)])
        if self.flash_attention:
            args.extend(["--flash-attn", "on"])
        if self.backend != Backend.AUTO:
            args.extend(["--split-mode", "none"] if self.backend == Backend.CPU else [])
        if self.mode == RuntimeMode.SERVER:
            args.extend(["--host", "127.0.0.1"])
        return args

    def export(self) -> dict[str, object]:
        return self.model_dump(mode="json")
