from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.table import Table

from kernl import __version__
from kernl.agent.tools import TOOLS, AgentContext
from kernl.cli.banner import render_banner

agent_app = typer.Typer(help="Restricted deterministic agent tools")


def register_agent(app: typer.Typer) -> None:
    app.add_typer(agent_app, name="agent")


def _context() -> AgentContext:
    from kernl.cli.app import CONFIG_DIR, get_store

    return AgentContext(
        store=get_store(),
        config_dir=CONFIG_DIR,
        workspace_root=Path.cwd(),
    )


def _parse_arguments(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--arg must use key=value form: {item}")
        key, value = item.split("=", 1)
        if not key.strip():
            raise typer.BadParameter("--arg keys cannot be empty")
        parsed[key.strip()] = value
    return parsed


def _summary(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, default=str)
    return rendered if len(rendered) <= 500 else rendered[:497] + "..."


@agent_app.callback(invoke_without_command=True)
def agent_workspace(ctx: typer.Context) -> None:
    """Show the restricted agent workspace when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    from kernl.cli.app import console

    render_banner(console, __version__)
    console.print("[bold]Restricted agent workspace[/]")
    console.print(
        "Available actions: hardware_profile, runtime_status, local_model_inspect, "
        "local_model_plan, benchmark_history, scan, runtime_verify."
    )
    console.print(
        "Tools are typed and approved explicitly; no shell execution is exposed."
    )
    console.print("Use [bold]kernl agent tools[/] and [bold]kernl agent run <tool>[/].")


@agent_app.command("tools")
def agent_tools() -> None:
    """List the typed tools the agent is allowed to call."""
    from kernl.cli.app import console

    table = Table(title="Allowed agent tools", border_style="magenta")
    for column in ("Tool", "Approval", "Description"):
        table.add_column(column)
    for tool in TOOLS.values():
        table.add_row(
            tool.name,
            "required" if tool.requires_approval else "read-only",
            tool.description,
        )
    console.print(table)


@agent_app.command("run")
def agent_run(
    tool: str,
    arg: Annotated[
        list[str] | None, typer.Option("--arg", help="key=value argument")
    ] = None,
    approve: Annotated[bool, typer.Option("--approve")] = False,
) -> None:
    """Run one typed tool with explicit inputs and auditable output."""
    from kernl.cli.app import console

    arguments_input = arg or []
    definition = TOOLS.get(tool)
    if definition is None:
        raise typer.BadParameter(f"unknown agent tool: {tool}")
    arguments = _parse_arguments(arguments_input)
    try:
        inputs = definition.input_model.model_validate(arguments)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    ctx = _context()
    if definition.requires_approval and not approve:
        ctx.store.record_agent_action(
            tool, arguments, False, "denied: explicit approval required"
        )
        raise typer.BadParameter("Approval required: --approve")

    try:
        result = definition.handler(inputs, ctx)
    except Exception as exc:
        ctx.store.record_agent_action(tool, arguments, approve, f"error: {exc}")
        console.print(f"[red]Agent tool failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    ctx.store.record_agent_action(tool, arguments, approve, _summary(result))
    console.print_json(data={"tool": tool, "result": result})


@agent_app.command("audit")
def agent_audit(limit: int = typer.Option(20, min=1, max=100)) -> None:
    """Show the most recent audited agent actions."""
    from kernl.cli.app import console

    rows = _context().store.recent_agent_actions(limit)
    table = Table(title="Agent audit log", border_style="magenta")
    for column in ("ID", "Tool", "Approved", "Result"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["id"]),
            str(row["tool"]),
            "yes" if row["approved"] else "no",
            str(row["result_summary"]),
        )
    if rows:
        console.print(table)
    else:
        console.print("No agent actions recorded.")
