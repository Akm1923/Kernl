from enum import StrEnum
from typing import Generic, TypeVar
from pydantic import BaseModel, Field

class ValueOrigin(StrEnum):
    DETECTED = "detected"
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"

T = TypeVar("T")

class Provenance(BaseModel, Generic[T]):
    value: T | None = None
    origin: ValueOrigin = ValueOrigin.UNKNOWN
    source: str = ""
    warning: str | None = None

class HardwareProfile(BaseModel):
    platform: Provenance[str]
    architecture: Provenance[str]
    cpu_model: Provenance[str]
    cpu_cores: Provenance[int]
    cpu_threads: Provenance[int]
    ram_total_bytes: Provenance[int]
    ram_available_bytes: Provenance[int]
    gpu_names: Provenance[list[str]]
    gpu_driver_versions: Provenance[list[str]]
    storage_free_bytes: Provenance[int]
    backends: Provenance[list[str]]
    warnings: list[str] = Field(default_factory=list)
    scanned_at: str
