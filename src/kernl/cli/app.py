from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from kernl import __version__
from kernl.cli.banner import render_banner
from kernl.hardware.scanner import scan
from kernl.security import has_secret, set_secret
from kernl.storage.db import Store

app = typer.Typer(help="Kernl — local AI compute platform", no_args_is_help=False)
console = Console()
DATA_DIR = Path.home() / ".kernl"


def get_store() -> Store:
    return Store(DATA_DIR / "kernl.sqlite3")


def display_profile(profile) -> None:
    table = Table(title="Hardware profile", border_style="magenta")
    for column in ("Capability", "Value", "Origin", "Source"):
        table.add_column(column)
    for name in ("platform", "architecture", "cpu_model", "cpu_cores", "cpu_threads", "ram_total_bytes", "ram_available_bytes", "gpu_names", "gpu_driver_versions", "storage_free_bytes", "backends"):
        value = getattr(profile, name)
        table.add_row(name, str(value.value if value.value is not None else "UNKNOWN"), value.origin.value, value.source)
    console.print(table)
    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/] {warning}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        render_banner(console, __version__)
        console.print("[magenta]>[/] Welcome to [bold bright_magenta]Kernl[/]. Run [bold]kernl init[/] to begin.")


@app.command()
def init() -> None:
    """Run first-launch onboarding and save a real hardware profile."""
    store = get_store()
    render_banner(console, __version__)
    provider = typer.prompt("Provider (local/gemini/groq)", default="local").strip().lower()
    if provider not in {"local", "gemini", "groq"}:
        raise typer.BadParameter("Choose local, gemini, or groq")
    store.set("provider", provider)
    if provider != "local":
        secret = typer.prompt(f"{provider} API key", hide_input=True, confirmation_prompt=True)
        set_secret(provider, secret)
        console.print("[green]Credential stored in the OS keyring; it was not written to SQLite.[/]")
    profile = scan()
    store.save_profile(profile)
    store.set("onboarding_complete", "1")
    console.print("[green]Onboarding complete. Hardware profile saved.[/]")
    display_profile(profile)


@app.command("scan")
def scan_command() -> None:
    """Scan the actual host and save a hardware profile."""
    profile = scan()
    get_store().save_profile(profile)
    display_profile(profile)


@app.command("profile")
def profile_command(action: str = typer.Argument("show")) -> None:
    """Show the latest saved hardware profile."""
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
        marker = "[bold bright_magenta]→[/] " if provider == selected else "  "
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
        console.print("[green]Local-only provider is ready; no network request is required.[/]")
    elif has_secret(provider):
        console.print(f"[yellow]{provider} credential exists. Live provider validation is deferred until the provider adapter is implemented.[/]")
    else:
        console.print(f"[red]No credential stored for {provider}.[/]")


@app.command()
def doctor() -> None:
    """Check foundation prerequisites without claiming deferred features work."""
    console.print(f"[green]OK[/] Kernl version: {__version__}")
    console.print(f"[green]OK[/] Data directory: {DATA_DIR}")
    console.print(f"[green]OK[/] SQLite store: available")
    console.print("[yellow]Deferred:[/] model discovery, llama.cpp installation/execution, and benchmarks.")


@app.command()
def agent() -> None:
    """Open the restricted foundation workspace."""
    render_banner(console, __version__)
    console.print("[bold]Restricted agent workspace[/]")
    console.print("Available foundation actions: scan, profile show, provider list.")
    console.print("No unrestricted shell execution is exposed.")


if __name__ == "__main__":
    app()
