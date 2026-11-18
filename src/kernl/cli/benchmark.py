from __future__ import annotations
from pathlib import Path
import json
import typer
from kernl.benchmark import BenchmarkError, BenchmarkRunner
from kernl.runtime.config_store import ConfigStore

benchmark_app = typer.Typer(help="Measure real runtime performance")


def register_benchmark(app: typer.Typer) -> None:
    app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("run")
def benchmark_run(config_name: str, prompt: list[str] = typer.Option(..., "--prompt"), output: Path | None = typer.Option(None, "--output"), approve: bool = typer.Option(False, "--approve")) -> None:
    """Run a supplied runtime callback only after execution approval.

    The llama.cpp inference callback is intentionally injected later; this command
    refuses to fabricate throughput until a live adapter is connected.
    """
    if not approve:
        raise typer.BadParameter("benchmark execution requires --approve")
    raise typer.Exit(code=2)
