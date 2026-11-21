from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class InferenceTelemetry:
    """Observed telemetry from one real inference run.

    A value of ``None`` means the signal was not observed during this run.
    Nothing in this structure is estimated or simulated.
    """

    generated_tokens: int
    wall_seconds: float
    time_to_first_output_seconds: float | None = None
    load_time_seconds: float | None = None
    prompt_eval_seconds: float | None = None
    prompt_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    peak_rss_bytes: int | None = None
    peak_vram_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            f.name: getattr(self, f.name) for f in fields(self)
        }
        payload["origins"] = {
            f.name: ("measured" if getattr(self, f.name) is not None else "unknown")
            for f in fields(self)
            if f.name not in {"generated_tokens", "wall_seconds"}
        }
        return payload
