from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from kernl.benchmark import BenchmarkError, BenchmarkRunner
from kernl.core.types import ValueOrigin
from kernl.runtime.config_store import ConfigStore
from kernl.runtime.llama_cpp import LlamaCppAdapter

benchmark_app = typer.Typer(help="Measure real runtime performance")


def register_benchmark(app: typer.Typer) -> None:
    app.add_typer(benchmark_app, name="benchmark")


def _format_metric(value: object, suffix: str = "") -> str:
    if value is None:
        return "UNKNOWN"
    try:
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


@benchmark_app.command("run")
def benchmark_run(
    config_name: str,
    prompt: Annotated[list[str], typer.Option("--prompt")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    approve: Annotated[bool, typer.Option("--approve")] = False,
    executable: Annotated[str, typer.Option("--executable")] = "llama-cli",
    timeout: Annotated[float, typer.Option("--timeout", min=1.0)] = 120.0,
) -> None:
    """Run observed llama.cpp inferences, store raw samples, and report measured values."""
    from kernl.cli.app import CONFIG_DIR, console, get_store

    if not approve:
        raise typer.BadParameter(
            "benchmark execution requires explicit approval via --approve"
        )
    resolved = shutil.which(executable)
    if not resolved:
        raise typer.BadParameter(f"llama.cpp executable not found: {executable}")

    try:
        config = ConfigStore(CONFIG_DIR).load(config_name)
        adapter = LlamaCppAdapter(Path(resolved))
        result = BenchmarkRunner().run(
            runtime=f"llama.cpp:{Path(resolved).name}",
            model=config.model_path,
            prompts=prompt,
            infer=lambda text: adapter.infer_once(
                config, text, approve_execution=True, timeout=timeout
            ),
        )
    except (
        BenchmarkError,
        RuntimeError,
        FileNotFoundError,
        PermissionError,
        ValueError,
    ) as exc:
        console.print(f"[red]Benchmark failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        BenchmarkRunner.write_json(result, output)
        BenchmarkRunner.write_csv(result, output.with_suffix(".csv"))
    get_store().save_benchmark(
        config_name, json.dumps(result.to_dict()), result.measured_at_epoch
    )

    console.print(
        f"[bold]Runtime:[/] {result.runtime}  [bold]Model:[/] {result.model}  "
        f"[bold]Origin:[/] {ValueOrigin.MEASURED.value}"
    )
    console.print(f"[bold]Prompts measured:[/] {len(result.samples)}")
    console.print(
        f"[bold]Median throughput:[/] "
        f"{_format_metric(result.median_tokens_per_second, ' tokens/s')}"
    )
    console.print(
        f"[bold]P95 request latency:[/] {_format_metric(result.p95_latency_seconds, ' s')}"
    )
    if output is not None:
        console.print(f"[green]Raw JSON:[/] {output}")
        console.print(f"[green]Raw CSV:[/] {output.with_suffix('.csv')}")
    console.print(
        "[yellow]Unmeasured:[/] load time, prompt-processing/eval split, peak RAM, "
        "and peak VRAM require a richer runtime telemetry source and are not invented here."
    )


@benchmark_app.command("compare")
def benchmark_compare(left: str, right: str) -> None:
    """Compare the most recent stored measurements for two named configurations."""
    from kernl.cli.app import console, get_store

    store = get_store()
    records = (
        ("left", left, store.latest_benchmark(left)),
        ("right", right, store.latest_benchmark(right)),
    )
    missing = [name for _, name, record in records if record is None]
    if missing:
        raise typer.BadParameter("No stored benchmark for: " + ", ".join(missing))

    table = Table(title="Benchmark comparison", border_style="magenta")
    for column in ("Side", "Config", "Median", "P95 latency", "Origin"):
        table.add_column(column)
    for side, name, record in records:
        assert record is not None
        table.add_row(
            side,
            name,
            _format_metric(record.get("median_tokens_per_second"), " tokens/s"),
            _format_metric(record.get("p95_latency_seconds"), " s"),
            ValueOrigin.MEASURED.value,
        )
    console.print(table)
    console.print(
        "[dim]Only values produced by observed llama.cpp inference are compared; "
        "no estimated throughput is mixed into this table.[/]"
    )
