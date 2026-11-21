from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kernl import __version__
from kernl.cli.agent import register_agent
from kernl.cli.banner import render_banner
from kernl.cli.benchmark import register_benchmark
from kernl.cli.models import register_models
from kernl.hardware.scanner import scan
from kernl.models.discovery import DiscoveryError, HuggingFaceDiscovery
from kernl.models.download import DownloadError, ModelDownloader
from kernl.models.gguf import GGUFError, GGUFReader
from kernl.models.memory import estimate_memory
from kernl.models.planner import plan_model_memory, recommend_gpu_layers
from kernl.providers import (
    ALL_PROVIDERS,
    REMOTE_PROVIDERS,
    redact_secret,
    validate_provider,
)
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.config_store import ConfigStore
from kernl.runtime.installer import InstallationError, RuntimeArtifact, RuntimeInstaller
from kernl.runtime.llama_cpp import LlamaCppAdapter, RuntimeErrorBase
from kernl.runtime.manager import ProcessError, RuntimeProcessManager
from kernl.runtime.process_store import ProcessStore
from kernl.security import get_secret, set_secret
from kernl.storage.db import Store

app = typer.Typer(help="Kernl — local AI compute platform", no_args_is_help=False)
console = Console()
DATA_DIR = Path.home() / ".kernl"
CONFIG_DIR = DATA_DIR / "configs"
PROCESS_STORE_PATH = DATA_DIR / "processes.json"
LOG_DIR = DATA_DIR / "logs"
register_models(app)
register_benchmark(app)
register_agent(app)


def get_store() -> Store:
    return Store(DATA_DIR / "kernl.sqlite3")


def get_process_manager() -> RuntimeProcessManager:
    return RuntimeProcessManager(ProcessStore(PROCESS_STORE_PATH), LOG_DIR)


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
    if provider not in ALL_PROVIDERS:
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
    for provider in ALL_PROVIDERS:
        marker = "[bold bright_magenta]>[/] " if provider == selected else "  "
        console.print(f"{marker}{provider}")


@provider_app.command("add")
def provider_add(provider: str = typer.Argument(...)) -> None:
    if provider not in REMOTE_PROVIDERS:
        raise typer.BadParameter("Only gemini or groq accepts an API key")
    secret = typer.prompt("API key", hide_input=True, confirmation_prompt=True)
    set_secret(provider, secret)
    get_store().set("provider", provider)
    console.print("[green]Provider credential stored securely.[/]")


@provider_app.command("test")
def provider_test(
    provider_name: str | None = typer.Option(
        None, "--provider", help="Provider to test; defaults to the selected provider"
    ),
    timeout: float = typer.Option(10.0, "--timeout", min=1.0),
) -> None:
    """Validate a stored provider credential with one live read-only request."""
    selected = get_store().get("provider") or "local"
    provider = (provider_name or selected).strip().lower()
    if provider not in ALL_PROVIDERS:
        raise typer.BadParameter(f"Unsupported provider: {provider}")
    if provider == "local":
        console.print(
            "[green]Local-only provider is ready; no network request is required.[/]"
        )
        return

    secret = get_secret(provider)
    if not secret:
        console.print(f"[red]No credential stored for {provider}.[/]")
        raise typer.Exit(code=1)

    result = validate_provider(provider, secret, timeout=timeout)
    if result.ok:
        http_status = (
            result.http_status if result.http_status is not None else "UNKNOWN"
        )
        console.print(
            f"[green]Verified:[/] {provider} credential accepted (HTTP {http_status})."
        )
        console.print(f"[dim]{redact_secret(result.detail, secret)}[/]")
        console.print(
            "[dim]Source: live provider API response; the key was sent in a header and was not printed.[/]"
        )
        return

    labels = {
        "invalid": "Rejected",
        "unreachable": "Unreachable",
        "unexpected": "Unverified",
    }
    console.print(
        f"[red]{labels[result.status]}:[/] {redact_secret(result.detail, secret)}"
    )
    http_status = result.http_status if result.http_status is not None else "UNKNOWN"
    console.print(f"[dim]HTTP status: {http_status}[/]")
    raise typer.Exit(code=1)


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


def _parse_gpu_layers_option(value: str) -> int | str:
    """Accept an explicit layer count, -1 (all), or 'auto'."""
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    message = "--gpu-layers must be an integer >= -1 or 'auto'"
    try:
        layers = int(normalized)
    except ValueError as exc:
        raise typer.BadParameter(message) from exc
    if layers < -1:
        raise typer.BadParameter(message)
    return layers


@models_local_app.command("plan")
def model_plan(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    gpu_layers: str = typer.Option(
        "0",
        "--gpu-layers",
        help="Layer count, -1 for all, or auto for the estimated-safe maximum",
    ),
    gpu_index: int = typer.Option(0, "--gpu-index", min=0),
    context_tokens: int | None = typer.Option(None, "--context", min=1),
    kv_cache_type: str = typer.Option("f16", "--kv-cache-type"),
    kv_cache_on_cpu: bool = typer.Option(False, "--kv-cache-on-cpu"),
) -> None:
    """Estimate CPU RAM and selected GPU VRAM needs for a local GGUF model."""
    try:
        layers_option = _parse_gpu_layers_option(gpu_layers)
        metadata = GGUFReader().read(path)
        profile = scan()
        recommendation = None
        if layers_option == "auto":
            recommendation = recommend_gpu_layers(
                metadata,
                profile,
                gpu_index=gpu_index,
                context_tokens=context_tokens,
                kv_cache_type=kv_cache_type,
                kv_cache_on_gpu=not kv_cache_on_cpu,
            )
            resolved_layers = (
                recommendation.gpu_layers if recommendation.status == "fit" else None
            )
        else:
            resolved_layers = layers_option
        plan = plan_model_memory(
            metadata,
            profile,
            gpu_layers=resolved_layers if resolved_layers is not None else 0,
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

    if recommendation is not None:
        statuses = {
            "fit": "estimated maximum fit; not the fastest configuration",
            "unknown": "UNKNOWN — required information is missing",
            "infeasible": "NONE — no estimated-safe layer count",
        }
        selected = (
            str(recommendation.gpu_layers)
            if recommendation.status == "fit"
            else "UNKNOWN"
        )
        console.print(f"[bold]Estimated-safe GPU layers:[/] {selected}")
        console.print(f"[bold]Selection status:[/] {statuses[recommendation.status]}")
        console.print(f"[bold]Recommendation basis:[/] {recommendation.reason}")
        for warning in recommendation.warnings:
            if warning not in plan.warnings:
                console.print(f"[yellow]Warning:[/] {warning}")

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
        layer_label = (
            "GPU layers used for estimate"
            if recommendation is not None
            else "GPU layers"
        )
        console.print(f"[bold]{layer_label}:[/] {plan.gpu_layers}")
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


@app.command("download")
def download_command(
    repo_id: str,
    filename: Annotated[
        str | None,
        typer.Option(
            "--filename",
            help="Repository file path; defaults to the smallest listed GGUF artifact",
        ),
    ] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    destination: Annotated[
        Path | None,
        typer.Option("--destination", help="Local download directory", file_okay=False),
    ] = None,
    approve: Annotated[
        bool,
        typer.Option("--approve", help="Explicitly allow model artifact download"),
    ] = False,
) -> None:
    """Download one verified Hugging Face model artifact after approval."""
    if not approve:
        raise typer.BadParameter(
            "model download requires explicit approval via --approve"
        )
    try:
        selected_filename = filename
        if selected_filename is None:
            inspection = HuggingFaceDiscovery().inspect(repo_id, revision=revision)
            candidates = [
                variant
                for variant in inspection.variants
                if variant.filename.lower().endswith(".gguf")
            ]
            if not candidates:
                raise typer.BadParameter(
                    "No GGUF artifact listed; pass --filename explicitly"
                )
            known = [
                variant for variant in candidates if variant.size_bytes is not None
            ]
            selected = min(
                known or candidates,
                key=lambda variant: (
                    variant.size_bytes is None,
                    variant.size_bytes if variant.size_bytes is not None else 0,
                    variant.filename,
                ),
            )
            selected_filename = selected.filename
            console.print(f"[bold]Selected artifact:[/] {selected_filename}")
        result = ModelDownloader().download(
            repo_id,
            selected_filename,
            revision=revision,
            destination=destination,
            approve_download=True,
        )
    except (DiscoveryError, DownloadError, PermissionError, ValueError) as exc:
        console.print(f"[red]Download failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Downloaded:[/] {result.path}")
    console.print(
        f"[bold]Size:[/] {result.size_bytes} bytes  [bold]SHA-256:[/] {result.sha256}"
    )
    if result.verified:
        console.print("[green]Verified:[/] SHA-256 matches Hugging Face metadata.")
    else:
        console.print(
            "[yellow]Unverified:[/] repository metadata did not provide a SHA-256 LFS oid."
        )


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


@runtime_app.command("verify")
def runtime_verify(
    name: str = typer.Argument(...),
    approve: bool = typer.Option(
        False, "--approve", help="Explicitly allow model execution"
    ),
    executable: str = typer.Option("llama-cli", "--executable"),
    prompt: str = typer.Option("Hello", "--prompt"),
    timeout: float = typer.Option(120.0, "--timeout", min=1.0),
) -> None:
    """Run one real llama.cpp inference against a saved configuration."""
    if not approve:
        raise typer.BadParameter(
            "runtime verification requires explicit approval via --approve"
        )
    resolved = shutil.which(executable)
    if not resolved:
        raise typer.BadParameter(f"llama.cpp executable not found: {executable}")
    try:
        config = ConfigStore(CONFIG_DIR).load(name)
        tokens = LlamaCppAdapter(Path(resolved)).infer_once(
            config, prompt, approve_execution=True, timeout=timeout
        )
    except (FileNotFoundError, PermissionError, RuntimeErrorBase, ValueError) as exc:
        console.print(f"[red]Verification failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Observed:[/] {tokens} generated tokens")
    console.print(
        "[dim]Source: observed llama.cpp inference; no simulated throughput is reported.[/]"
    )


def _resolve_runtime_executable(executable: str) -> Path:
    resolved = shutil.which(executable)
    if not resolved:
        raise typer.BadParameter(f"llama.cpp executable not found: {executable}")
    return Path(resolved)


def _runtime_extra_args(executable: Path, prompt: str, n_predict: int) -> list[str]:
    if "server" in executable.name.lower():
        return []
    return ["--prompt", prompt, "--n-predict", str(n_predict)]


@app.command("run")
def run_command(
    name: str,
    executable: str = typer.Option("llama-cli", "--executable"),
    prompt: str = typer.Option("Hello", "--prompt"),
    n_predict: int = typer.Option(128, "--n-predict", min=1, max=1_048_576),
    approve: bool = typer.Option(
        False, "--approve", help="Explicitly allow model execution"
    ),
) -> None:
    """Launch a saved runtime configuration as a managed process."""
    if not approve:
        raise typer.BadParameter(
            "runtime launch requires explicit approval via --approve"
        )
    resolved = _resolve_runtime_executable(executable)
    try:
        config = ConfigStore(CONFIG_DIR).load(name)
        record = get_process_manager().launch(
            config_name=name,
            config=config,
            executable=resolved,
            approve_execution=True,
            extra_args=_runtime_extra_args(resolved, prompt, n_predict),
        )
    except (
        FileNotFoundError,
        PermissionError,
        ProcessError,
        RuntimeErrorBase,
        ValueError,
    ) as exc:
        console.print(f"[red]Launch failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Started:[/] PID {record.pid}")
    console.print(f"[bold]Configuration:[/] {record.config_name}")
    console.print(f"[bold]Log file:[/] {record.log_path}")
    console.print(
        "[dim]Use kernl processes, kernl logs, kernl stop, and kernl restart.[/]"
    )


@app.command("stop")
def stop_command(pid: int) -> None:
    """Stop one managed runtime process."""
    try:
        record = get_process_manager().stop(pid)
    except (KeyError, ProcessError) as exc:
        console.print(f"[red]Stop failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Stopped:[/] PID {record.pid} ({record.config_name})")


@app.command("restart")
def restart_command(
    pid: int,
    executable: str = typer.Option("llama-cli", "--executable"),
    prompt: str = typer.Option("Hello", "--prompt"),
    n_predict: int = typer.Option(128, "--n-predict", min=1, max=1_048_576),
    approve: bool = typer.Option(
        False, "--approve", help="Explicitly allow model execution"
    ),
) -> None:
    """Stop and relaunch a managed runtime process."""
    if not approve:
        raise typer.BadParameter(
            "runtime restart requires explicit approval via --approve"
        )
    manager = get_process_manager()
    try:
        existing = manager.status(pid)
        config = ConfigStore(CONFIG_DIR).load(existing.config_name)
        resolved = _resolve_runtime_executable(executable)
        record = manager.restart(
            pid,
            config=config,
            executable=resolved,
            approve_execution=True,
            extra_args=_runtime_extra_args(resolved, prompt, n_predict),
        )
    except (
        FileNotFoundError,
        KeyError,
        PermissionError,
        ProcessError,
        RuntimeErrorBase,
        ValueError,
    ) as exc:
        console.print(f"[red]Restart failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Restarted:[/] PID {record.pid} ({record.config_name})")


@app.command("processes")
def processes_command() -> None:
    """List managed runtime processes and their observed status."""
    records = get_process_manager().list()
    if not records:
        console.print("No managed runtime processes.")
        return
    table = Table(title="Managed runtime processes", border_style="magenta")
    for column in ("PID", "Config", "Status", "Started", "Log"):
        table.add_column(column)
    for record in records:
        started = dt.datetime.fromtimestamp(record.started_at, tz=dt.UTC).isoformat(
            timespec="seconds"
        )
        table.add_row(
            str(record.pid),
            record.config_name,
            record.status,
            started,
            record.log_path,
        )
    console.print(table)


@app.command("logs")
def logs_command(
    pid: int,
    limit: int = typer.Option(100, "--limit", min=1, max=10_000),
) -> None:
    """Show the tail of a managed runtime process log."""
    try:
        lines = get_process_manager().logs(pid, limit)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]Logs failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    if not lines:
        console.print("[yellow]No log lines available.[/]")
        return
    for line in lines:
        console.print(line)


config_app = typer.Typer(help="Manage runtime configurations")
app.add_typer(config_app, name="config")


@config_app.command("save")
def config_save(
    name: str,
    model_path: Path,
    context_size: int = typer.Option(4096, "--context-size", min=128, max=1_048_576),
    gpu_layers: str = typer.Option(
        "0",
        "--gpu-layers",
        help="Layer count, -1 for all, or auto for the estimated-safe count",
    ),
    kv_cache_on_cpu: bool = typer.Option(False, "--kv-cache-on-cpu"),
) -> None:
    try:
        layers_option = _parse_gpu_layers_option(gpu_layers)
        resolved_layers: int
        if layers_option == "auto":
            metadata = GGUFReader().read(model_path)
            profile = scan()
            recommendation = recommend_gpu_layers(
                metadata,
                profile,
                gpu_index=0,
                context_tokens=context_size,
                kv_cache_type="f16",
                kv_cache_on_gpu=not kv_cache_on_cpu,
            )
            if recommendation.status != "fit" or recommendation.gpu_layers is None:
                raise typer.BadParameter(
                    f"Cannot save auto GPU layers: {recommendation.reason}"
                )
            resolved_layers = recommendation.gpu_layers
            estimated = True
        else:
            resolved_layers = layers_option
            estimated = False
        path = ConfigStore(CONFIG_DIR).save(
            name,
            RuntimeConfig(
                model_path=model_path,
                context_size=context_size,
                gpu_layers=resolved_layers,
                kv_cache_on_gpu=not kv_cache_on_cpu,
            ),
        )
    except (GGUFError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Saved:[/] {path}")
    if estimated:
        console.print(
            f"[yellow]Estimated-safe GPU layers:[/] {resolved_layers} "
            "(planning estimate, not a measured runtime result)."
        )


@config_app.command("show")
def config_show(name: str) -> None:
    console.print_json(data=ConfigStore(CONFIG_DIR).load(name).export())


@app.command()
def doctor() -> None:
    console.print(f"[green]OK[/] Kernl version: {__version__}")
    console.print(f"[green]OK[/] Data directory: {DATA_DIR}")
    console.print("[green]OK[/] SQLite store: available")
    console.print(
        "[yellow]Requires local assets:[/] runtime verification and benchmarks need a "
        "llama.cpp executable, a GGUF model, and explicit approval."
    )
    console.print(
        "[green]OK[/] Provider validation: live Gemini/Groq checks via kernl provider test"
    )
    console.print(
        "[yellow]Deferred:[/] native telemetry sampling for load time, prompt/eval split, "
        "peak RAM, and peak VRAM."
    )


if __name__ == "__main__":
    app()
