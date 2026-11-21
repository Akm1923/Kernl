from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from pathlib import Path

from kernl.core.telemetry import InferenceTelemetry


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce a valid measurement."""


@dataclass(frozen=True)
class BenchmarkSample:
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    time_to_first_output_seconds: float | None = None
    load_time_seconds: float | None = None
    prompt_eval_seconds: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    peak_rss_bytes: int | None = None
    peak_vram_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["origins"] = {
            f.name: ("measured" if getattr(self, f.name) is not None else "unknown")
            for f in fields(self)
            if f.name not in {"prompt_tokens", "generated_tokens"}
        }
        return payload


def _median(values: Iterable[float]) -> float | None:
    collected = list(values)
    return statistics.median(collected) if collected else None


@dataclass(frozen=True)
class BenchmarkResult:
    runtime: str
    model: str
    samples: tuple[BenchmarkSample, ...]
    median_tokens_per_second: float | None
    p95_latency_seconds: float | None
    measured_at_epoch: float
    median_time_to_first_output_seconds: float | None = None
    median_load_time_seconds: float | None = None
    median_prompt_tokens_per_second: float | None = None
    peak_rss_bytes: int | None = None
    peak_vram_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "model": self.model,
            "samples": [sample.to_dict() for sample in self.samples],
            "median_tokens_per_second": self.median_tokens_per_second,
            "p95_latency_seconds": self.p95_latency_seconds,
            "measured_at_epoch": self.measured_at_epoch,
            "median_time_to_first_output_seconds": (
                self.median_time_to_first_output_seconds
            ),
            "median_load_time_seconds": self.median_load_time_seconds,
            "median_prompt_tokens_per_second": self.median_prompt_tokens_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "origins": {
                "median_tokens_per_second": _origin(self.median_tokens_per_second),
                "p95_latency_seconds": _origin(self.p95_latency_seconds),
                "median_time_to_first_output_seconds": _origin(
                    self.median_time_to_first_output_seconds
                ),
                "median_load_time_seconds": _origin(self.median_load_time_seconds),
                "median_prompt_tokens_per_second": _origin(
                    self.median_prompt_tokens_per_second
                ),
                "peak_rss_bytes": _origin(self.peak_rss_bytes),
                "peak_vram_bytes": _origin(self.peak_vram_bytes),
            },
        }


def _origin(value: object) -> str:
    return "measured" if value is not None else "unknown"


class BenchmarkRunner:
    """Records timings from a supplied real inference function.

    The callback must perform actual inference and return a generated-token
    count (``int``) or a full :class:`InferenceTelemetry` observation. No
    synthetic throughput values are accepted.
    """

    def run(
        self,
        *,
        runtime: str,
        model: Path,
        prompts: Iterable[str],
        infer: Callable[[str], int | InferenceTelemetry],
    ) -> BenchmarkResult:
        if not model.is_file():
            raise BenchmarkError(f"model file not found: {model}")
        samples: list[BenchmarkSample] = []
        for prompt in prompts:
            started = time.perf_counter()
            outcome = infer(prompt)
            elapsed = time.perf_counter() - started
            if isinstance(outcome, InferenceTelemetry):
                telemetry = outcome
                if not isinstance(telemetry.generated_tokens, int) or (
                    telemetry.generated_tokens <= 0
                ):
                    raise BenchmarkError(
                        "inference callback must return a positive generated-token count"
                    )
            elif isinstance(outcome, int) and not isinstance(outcome, bool):
                if outcome <= 0:
                    raise BenchmarkError(
                        "inference callback must return a positive generated-token count"
                    )
                telemetry = InferenceTelemetry(
                    generated_tokens=outcome, wall_seconds=elapsed
                )
            else:
                raise BenchmarkError(
                    "inference callback must return an int or InferenceTelemetry"
                )
            if elapsed <= 0:
                raise BenchmarkError("inference elapsed time was not measurable")
            prompt_tokens = telemetry.prompt_tokens or max(1, len(prompt.split()))
            samples.append(
                BenchmarkSample(
                    prompt_tokens=prompt_tokens,
                    generated_tokens=telemetry.generated_tokens,
                    elapsed_seconds=elapsed,
                    tokens_per_second=telemetry.generated_tokens / elapsed,
                    time_to_first_output_seconds=telemetry.time_to_first_output_seconds,
                    load_time_seconds=telemetry.load_time_seconds,
                    prompt_eval_seconds=telemetry.prompt_eval_seconds,
                    prompt_tokens_per_second=telemetry.prompt_tokens_per_second,
                    generation_tokens_per_second=(
                        telemetry.generation_tokens_per_second
                    ),
                    peak_rss_bytes=telemetry.peak_rss_bytes,
                    peak_vram_bytes=telemetry.peak_vram_bytes,
                )
            )
        if not samples:
            raise BenchmarkError("at least one benchmark prompt is required")
        throughputs = [sample.tokens_per_second for sample in samples]
        latencies = sorted(sample.elapsed_seconds for sample in samples)
        p95_index = min(len(latencies) - 1, max(0, int(len(latencies) * 0.95) - 1))
        peak_rss = [s.peak_rss_bytes for s in samples if s.peak_rss_bytes is not None]
        peak_vram = [
            s.peak_vram_bytes for s in samples if s.peak_vram_bytes is not None
        ]
        return BenchmarkResult(
            runtime=runtime,
            model=str(model),
            samples=tuple(samples),
            median_tokens_per_second=statistics.median(throughputs),
            p95_latency_seconds=latencies[p95_index],
            measured_at_epoch=time.time(),
            median_time_to_first_output_seconds=_median(
                s.time_to_first_output_seconds
                for s in samples
                if s.time_to_first_output_seconds is not None
            ),
            median_load_time_seconds=_median(
                s.load_time_seconds for s in samples if s.load_time_seconds is not None
            ),
            median_prompt_tokens_per_second=_median(
                s.prompt_tokens_per_second
                for s in samples
                if s.prompt_tokens_per_second is not None
            ),
            peak_rss_bytes=max(peak_rss) if peak_rss else None,
            peak_vram_bytes=max(peak_vram) if peak_vram else None,
        )

    @staticmethod
    def write_json(result: BenchmarkResult, path: Path) -> None:
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def write_csv(result: BenchmarkResult, path: Path) -> None:
        fieldnames = [f.name for f in fields(BenchmarkSample)]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample in result.samples:
                writer.writerow({name: getattr(sample, name) for name in fieldnames})
