from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from kernl import __version__

VERSION = __version__

_ART = """      /\\        _  __
  ___/  \\___   | |/ /___ _ __ _ __
 /  / /\\ \\  \\  | ' // _ \\ '__| '_ \\
    /_/  \\_\\  | . \\  __/ |  | | | |
                |_|\\_\\___|_|  |_| |_|"""


def render_banner(console: Console, version: str = VERSION) -> None:
    """Render Kernl identity locally with a compact terminal fallback."""
    if console.width < 88:
        art = Text("⚡ KERNL", style="bold bright_magenta")
        nav = Text("CHAT · AGENTS · MODELS · RUNTIME · BENCHMARK", style="magenta")
    else:
        art = Text(_ART, style="bold bright_magenta")
        nav = Text(
            "  CHAT    AGENTS    MODELS    RUNTIME    BENCHMARK",
            style="bold bright_magenta",
        )
    body = Group(
        art, Text("[ LOCAL AI COMPUTE PLATFORM ]", style="white"), Text(""), nav
    )
    console.print(
        Panel(
            body,
            title="[bold white]⚡ KERNL[/]",
            title_align="left",
            subtitle=f"[dim]v{version}[/]",
            border_style="magenta",
            expand=True,
        )
    )
