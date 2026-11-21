from __future__ import annotations

import typer
from rich.table import Table

from kernl.models.discovery import DiscoveryError, HuggingFaceDiscovery
from kernl.models.recommendation import recommend

models_app = typer.Typer(help="Discover models from Hugging Face")


def register_models(app: typer.Typer) -> None:
    app.add_typer(models_app, name="models")


@models_app.command("search")
def search(
    query: str = typer.Argument(""),
    task: str | None = typer.Option(None, "--task"),
    limit: int = typer.Option(10, min=1, max=100),
) -> None:
    """Search live Hugging Face metadata; no fake catalog is used."""
    try:
        models = HuggingFaceDiscovery().search(query=query, task=task, limit=limit)
    except DiscoveryError as exc:
        raise typer.Exit(code=1) from exc
    table = Table(title="Hugging Face models", border_style="magenta")
    for column in ("Repository", "Task", "Downloads", "Likes", "Gated"):
        table.add_column(column)
    for model in models:
        table.add_row(
            model.repo_id,
            model.pipeline_tag or "unknown",
            str(model.downloads),
            str(model.likes),
            str(model.gated if model.gated is not None else "unknown"),
        )
    from kernl.cli.app import console

    console.print(table)


@models_app.command("inspect")
def inspect_repo(
    repo_id: str = typer.Argument(...),
    revision: str | None = typer.Option(None, "--revision"),
) -> None:
    """Inspect a Hugging Face repo's GGUF variants without downloading weights."""
    try:
        inspection = HuggingFaceDiscovery().inspect(repo_id, revision=revision)
    except DiscoveryError as exc:
        raise typer.Exit(code=1) from exc

    from kernl.cli.app import console

    console.print(f"[bold]Repository:[/] {inspection.repo_id}")
    console.print(
        f"[bold]Revision:[/] {inspection.revision or 'UNKNOWN'}  "
        f"[bold]License:[/] {inspection.license or 'UNKNOWN'}  "
        f"[bold]Gated:[/] {inspection.gated if inspection.gated is not None else 'UNKNOWN'}"
    )
    table = Table(title="GGUF variants", border_style="magenta")
    for column in ("File", "Quantization", "Size"):
        table.add_column(column)
    for variant in inspection.variants:
        size = (
            f"{variant.size_bytes / 1024**3:.2f} GiB"
            if variant.size_bytes is not None
            else "UNKNOWN"
        )
        table.add_row(
            variant.filename,
            variant.quantization or "UNKNOWN",
            size,
        )
    if inspection.variants:
        console.print(table)
    else:
        console.print("[yellow]No GGUF variants were listed for this revision.[/]")


@models_app.command("recommend")
def recommend_models(
    query: str = typer.Argument(""),
    limit: int = typer.Option(5, min=1, max=20),
    available_gb: float | None = typer.Option(None, "--available-gb", min=0),
) -> None:
    """Rank discovered models using a tri-state, file-size-only estimate."""
    from kernl.cli.app import console, get_store

    if available_gb is not None:
        available = int(available_gb * 1024**3)
    else:
        profile = get_store().latest_profile()
        available = profile.ram_available_bytes.value if profile else None
        if available is None:
            raise typer.BadParameter(
                "No saved available RAM; run kernl scan or pass --available-gb"
            )

    discovery = HuggingFaceDiscovery()
    try:
        candidates = discovery.search(query=query, task="text-generation", limit=limit)
    except DiscoveryError as exc:
        raise typer.Exit(code=1) from exc

    sizes: dict[str, int] = {}
    for candidate in candidates:
        try:
            inspection = discovery.inspect(candidate.repo_id)
        except DiscoveryError:
            continue
        known_sizes = [
            variant.size_bytes
            for variant in inspection.variants
            if variant.size_bytes is not None
        ]
        if known_sizes:
            sizes[candidate.repo_id] = min(known_sizes)

    results = recommend(
        candidates, available_ram_bytes=available, required_ram_bytes_by_repo=sizes
    )
    table = Table(title="Model recommendations", border_style="magenta")
    for column in ("Repository", "Estimate", "Reason"):
        table.add_column(column)
    for result in results:
        label = {True: "YES", False: "NO", None: "UNKNOWN"}[
            result.compatibility.compatible
        ]
        table.add_row(
            result.model.repo_id,
            label,
            "; ".join(result.compatibility.reasons) or "no constraints detected",
        )
    if results:
        console.print(table)
    else:
        console.print("No models found.")
    console.print(
        "[yellow]Note:[/] GGUF file size only; KV cache, runtime overhead, and GPU "
        "VRAM are not included. Run [bold]kernl model plan --gpu-layers auto[/] on a "
        "downloaded file for a full estimate."
    )
