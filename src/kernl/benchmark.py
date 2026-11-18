from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import statistics
import time
from typing import Callable, Iterable


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce a valid measurement."""


@dataclass(frozen=True)
class BenchmarkSample:
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class BenchmarkResult:
    runtime: str
    model: str
    samples: tuple[BenchmarkSample, ...]
    median_tokens_per_second: float | None
    p95_latency_seconds: float | None
    measured_at_epoch: float

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "model": self.model,
            "samples": [sample.__dict__ for sample in self.samples],
            "median_tokens_per_second": self.median_tokens_per_second,
            "p95_latency_seconds": self.p95_latency_seconds,
            "measured_at_epoch": self.measured_at_epoch,
        }


class BenchmarkRunner:
    """Records timings from a supplied real inference function.

    The callback must perform actual inference and return generated token count.
    No synthetic throughput values are accepted.
    """

    def run(
        self,
        *,
        runtime: str,
        model: Path,
        prompts: Iterable[str],
        infer: Callable[[str], int],
    ) -> BenchmarkResult:
        if not model.is_file():
            raise BenchmarkError(f"model file not found: {model}")
        samples: list[BenchmarkSample] = []
        for prompt in prompts:
            prompt_tokens = max(1, len(prompt.split()))
            started = time.perf_counter()
            generated = infer(prompt)
            elapsed = time.perf_counter() - started
            if not isinstance(generated, int) or generated <= 0:
                raise BenchmarkError("inference callback must return a positive generated-token count")
            if elapsed <= 0:
                raise BenchmarkError("inference elapsed time was not measurable")
            samples.append(BenchmarkSample(prompt_tokens, generated, elapsed, generated / elapsed))
        if not samples:
            raise BenchmarkError("at least one benchmark prompt is required")
        throughputs = [sample.tokens_per_second for sample in samples]
        latencies = sorted(sample.elapsed_seconds for sample in samples)
        p95_index = min(len(latencies) - 1, max(0, int(len(latencies) * 0.95) - 1))
        return BenchmarkResult(runtime, str(model), tuple(samples), statistics.median(throughputs), latencies[p95_index], time.time())

    @staticmethod
    def write_json(result: BenchmarkResult, path: Path) -> None:
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def write_csv(result: BenchmarkResult, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["prompt_tokens", "generated_tokens", "elapsed_seconds", "tokens_per_second"])
            writer.writeheader()
            writer.writerows(sample.__dict__ for sample in result.samples)
