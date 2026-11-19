from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kernl import __version__
from kernl.cli.banner import render_banner
from kernl.cli.benchmark import register_benchmark
from kernl.cli.models import register_models
from kernl.hardware.scanner import scan
from kernl.models.gguf import GGUFError, GGUFReader
from kernl.models.memory import estimate_memory
from kernl.models.planner import plan_model_memory
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.config_store import ConfigStore
from kernl.runtime.installer import InstallationError, RuntimeArtifact, RuntimeInstaller
from kernl.security import has_secret, set_secret
from kernl.storage.db import Store

app = typer.Typer(help="Kernl — local AI compute platform", no_args_is_help=False)
console = Console()
DATA_DIR = Path.home() / ".kernl"
CONFIG_DIR = DATA_DIR / "configs"
register_models(app)
register_benchmark(app)


def get_store() -> Store:
    return Store(DATA_DIR / "kernl.sqlite3")


def display_profile(profile) -> None:
    table = Table(title="Hardware profile", border_style="magenta")
    for column in ("Capability", "Value", "Origin", "Source"):
        table.add_column(column)
    profile_fields = (
        "platform",
        "architecture",
        "cpu_model",
        "cpu_cores",
        "cpu_threads",
        "ram_total_bytes",
        "ram_available_bytes",
        "gpu_names",
        "gpu_driver_versions",
        "gpu_memory_bytes",
        "storage_free_bytes",
        "backends",
    )
    for name in profile_fields:
        value = getattr(profile, name)
        display_value = value.value
        if name == "gpu_memory_bytes" and display_value is not None:
            display_value = [
                f"{memory / 1024**3:.2f} GiB" if memory is not None else "UNKNOWN"
                for memory in display_value
            ]
        table.add_row(
            name,
            str(display_value if display_value is not None else "UNKNOWN"),
            value.origin.value,
            value.source,
        )
    console.print(table)
    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/] {warning}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        render_banner(console, __version__)
        console.print(
            "[magenta]>[/] Welcome to [bold bright_magenta]Kernl[/]. Run [bold]kernl init[/] to begin."
        )


@app.command()
def init() -> None:
    store = get_store()
    render_banner(console, __version__)
    provider = (
        typer.prompt("Provider (local/gemini/groq)", default="local").strip().lower()
    )
    if provider not in {"local", "gemini", "groq"}:
        raise typer.BadParameter("Choose local, gemini, or groq")
    store.set("provider", provider)
    if provider != "local":
        secret = typer.prompt(
            f"{provider} API key", hide_input=True, confirmation_prompt=True
        )
        set_secret(provider, secret)
        console.print(
            "[green]Credential stored in the OS keyring; it was not written to SQLite.[/]"
        )
    profile = scan()
    store.save_profile(profile)
    store.set("onboarding_complete", "1")
    console.print("[green]Onboarding complete. Hardware profile saved.[/]")
    display_profile(profile)


@app.command("scan")
def scan_command() -> None:
    profile = scan()
    get_store().save_profile(profile)
    display_profile(profile)


@app.command("profile")
def profile_command(action: str = typer.Argument("show")) -> None:
    if action != "show":
        raise typer.BadParameter("Only 'show' is supported")
    profile = get_store().latest_profile()
    if profile is None:
        raise typer.BadParameter("No profile found. Run kernl init or kernl scan.")
    display_profile(profile)


provider_app = typer.Typer(help="Manage AI providers")
app.add_typer(provider_app, name="provider")


@provider_app.command("list")
def provider_list() -> None:
    selected = get_store().get("provider") or "not configured"
    for provider in ("local", "gemini", "groq"):
        marker = "[bold bright_magenta]>[/] " if provider == selected else "  "
        console.print(f"{marker}{provider}")


@provider_app.command("add")
def provider_add(provider: str = typer.Argument(...)) -> None:
    if provider not in {"gemini", "groq"}:
        raise typer.BadParameter("Only gemini or groq accepts an API key")
    secret = typer.prompt("API key", hide_input=True, confirmation_prompt=True)
    set_secret(provider, secret)
    get_store().set("provider", provider)
    console.print("[green]Provider credential stored securely.[/]")


@provider_app.command("test")
def provider_test() -> None:
    provider = get_store().get("provider") or "local"
    if provider == "local":
        console.print(
            "[green]Local-only provider is ready; no network request is required.[/]"
        )
    elif has_secret(provider):
        console.print(
            f"[yellow]{provider} credential exists. Live provider validation is deferred until the provider adapter is implemented.[/]"
        )
    else:
        console.print(f"[red]No credential stored for {provider}.[/]")


models_local_app = typer.Typer(help="Inspect local model files")
app.add_typer(models_local_app, name="model")


@models_local_app.command("inspect")
def model_inspect(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    try:
        metadata = GGUFReader().read(path)
    except GGUFError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for key, value in (
        ("Architecture", metadata.architecture),
        ("Parameters", metadata.parameter_count),
        ("Context", metadata.context_length),
        ("Embedding", metadata.embedding_length),
        ("Blocks", metadata.block_count),
        ("File bytes", metadata.file_size_bytes),
    ):
        console.print(f"[bold]{key}:[/] {value if value is not None else 'UNKNOWN'}")


@models_local_app.command("plan")
def model_plan(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    gpu_layers: int = typer.Option(0, "--gpu-layers", min=-1),
    gpu_index: int = typer.Option(0, "--gpu-index", min=0),
    context_tokens: int | None = typer.Option(None, "--context", min=1),
    kv_cache_type: str = typer.Option("f16", "--kv-cache-type"),
    kv_cache_on_cpu: bool = typer.Option(False, "--kv-cache-on-cpu"),
) -> None:
    """Estimate CPU RAM and selected GPU VRAM needs for a local GGUF model."""
    try:
        metadata = GGUFReader().read(path)
        plan = plan_model_memory(
            metadata,
            scan(),
            gpu_layers=gpu_layers,
            gpu_index=gpu_index,
            context_tokens=context_tokens,
            kv_cache_type=kv_cache_type,
            kv_cache_on_gpu=not kv_cache_on_cpu,
        )
    except (GGUFError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    def format_bytes(value: int | None) -> str:
        return f"{value / 1024**3:.2f} GiB" if value is not None else "UNKNOWN"

    def fit_label(value: bool | None) -> str:
        return {True: "YES", False: "NO", None: "UNKNOWN"}[value]

    console.print(f"[bold]Model:[/] {path}")
    console.print(
        f"[bold]Model weights (GGUF file size):[/] {format_bytes(plan.model_bytes)}"
    )
    console.print(f"[bold]KV cache estimate:[/] {format_bytes(plan.kv_cache_bytes)}")
    cpu_requirement_label = (
        "known minimum"
        if plan.kv_cache_bytes is None and (plan.gpu_layers == 0 or kv_cache_on_cpu)
        else "estimated"
    )
    gpu_requirement_label = (
        "known minimum"
        if plan.kv_cache_bytes is None and plan.gpu_layers > 0 and not kv_cache_on_cpu
        else "estimated"
    )
    console.print(
        f"[bold]CPU RAM:[/] {format_bytes(plan.cpu_required_bytes)} {cpu_requirement_label} / "
        f"{format_bytes(plan.cpu_available_bytes)} available — {fit_label(plan.cpu_fits)}"
    )
    if plan.gpu_name is not None:
        console.print(
            f"[bold]{plan.gpu_name} VRAM:[/] {format_bytes(plan.gpu_required_bytes)} {gpu_requirement_label} / "
            f"{format_bytes(plan.gpu_budget_bytes)} budget — {fit_label(plan.gpu_fits)}"
        )
        console.print(f"[bold]GPU layers:[/] {plan.gpu_layers}")
    else:
        console.print("[bold]GPU offload:[/] disabled")
    console.print(
        f"[bold]Overall compatibility:[/] {fit_label(plan.overall_compatible)}"
    )
    for warning in plan.warnings:
        console.print(f"[yellow]Warning:[/] {warning}")


@models_local_app.command("estimate")
def model_estimate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    quantization: str = typer.Option("", "--quantization"),
    available_gb: float | None = typer.Option(None, "--available-gb"),
) -> None:
    try:
        metadata = GGUFReader().read(path)
    except GGUFError as exc:
        raise typer.BadParameter(str(exc)) from exc
    available = int(available_gb * 1024**3) if available_gb is not None else None
    estimate = estimate_memory(
        parameter_count=metadata.parameter_count,
        quantization=quantization or None,
        context_tokens=metadata.context_length or 4096,
        layers=metadata.block_count,
        embedding_length=metadata.embedding_length,
        available_bytes=available,
    )
    console.print(
        f"Estimated total bytes: {estimate.total_bytes if estimate.total_bytes is not None else 'UNKNOWN'}"
    )
    console.print(
        f"Fits available memory: {estimate.fits if estimate.fits is not None else 'UNKNOWN'}"
    )
    for warning in estimate.warnings:
        console.print(f"[yellow]Warning:[/] {warning}")


runtime_app = typer.Typer(help="Manage local inference runtime")
app.add_typer(runtime_app, name="runtime")


@runtime_app.command("install")
def runtime_install(
    url: str,
    sha256: str,
    filename: str,
    approve: bool = typer.Option(False, "--approve"),
) -> None:
    try:
        target = RuntimeInstaller().install(
            RuntimeArtifact(url, sha256, filename),
            DATA_DIR / "runtime",
            approve_install=approve,
        )
    except (InstallationError, PermissionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Installed verified runtime:[/] {target}")


@runtime_app.command("status")
def runtime_status(executable: str = typer.Option("llama-cli", "--executable")) -> None:
    found = shutil.which(executable)
    console.print(
        f"[green]Found:[/] {found}" if found else f"[yellow]Not found:[/] {executable}"
    )


config_app = typer.Typer(help="Manage runtime configurations")
app.add_typer(config_app, name="config")


@config_app.command("save")
def config_save(
    name: str,
    model_path: Path,
    context_size: int = 4096,
    gpu_layers: int = 0,
    kv_cache_on_cpu: bool = typer.Option(False, "--kv-cache-on-cpu"),
) -> None:
    path = ConfigStore(CONFIG_DIR).save(
        name,
        RuntimeConfig(
            model_path=model_path,
            context_size=context_size,
            gpu_layers=gpu_layers,
            kv_cache_on_gpu=not kv_cache_on_cpu,
        ),
    )
    console.print(f"[green]Saved:[/] {path}")


@config_app.command("show")
def config_show(name: str) -> None:
    console.print_json(data=ConfigStore(CONFIG_DIR).load(name).export())


@app.command()
def doctor() -> None:
    console.print(f"[green]OK[/] Kernl version: {__version__}")
    console.print(f"[green]OK[/] Data directory: {DATA_DIR}")
    console.print("[green]OK[/] SQLite store: available")
    console.print(
        "[yellow]Deferred:[/] provider API validation, runtime execution, and end-to-end benchmarks require local credentials/binaries."
    )


@app.command()
def agent() -> None:
    render_banner(console, __version__)
    console.print("[bold]Restricted agent workspace[/]")
    console.print(
        "Available actions: scan, profile show, provider list, models search, model inspect/estimate/plan, runtime status, benchmark run."
    )
    console.print("No unrestricted shell execution is exposed.")


if __name__ == "__main__":
    app()
