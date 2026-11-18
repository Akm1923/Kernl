from __future__ import annotations
import typer
from rich.table import Table
from kernl.models.discovery import HuggingFaceDiscovery, DiscoveryError

models_app = typer.Typer(help="Discover models from Hugging Face")

def register_models(app: typer.Typer) -> None:
    app.add_typer(models_app, name="models")

@models_app.command("search")
def search(query: str = typer.Argument(""), task: str | None = typer.Option(None, "--task"), limit: int = typer.Option(10, min=1, max=100)) -> None:
    """Search live Hugging Face metadata; no fake catalog is used."""
    try:
        models = HuggingFaceDiscovery().search(query=query, task=task, limit=limit)
    except DiscoveryError as exc:
        raise typer.Exit(code=1) from exc
    table = Table(title="Hugging Face models", border_style="magenta")
    for column in ("Repository", "Task", "Downloads", "Likes", "Gated"):
        table.add_column(column)
    for model in models:
        table.add_row(model.repo_id, model.pipeline_tag or "unknown", str(model.downloads), str(model.likes), str(model.gated if model.gated is not None else "unknown"))
    from kernl.cli.app import console
    console.print(table)
