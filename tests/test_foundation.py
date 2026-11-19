from importlib.metadata import version as package_version
from types import SimpleNamespace

from rich.console import Console

from kernl import __version__
from kernl.cli.app import display_profile
from kernl.cli.banner import render_banner
from kernl.core.types import HardwareProfile, Provenance, ValueOrigin
from kernl.hardware.scanner import scan
from kernl.storage.db import Store


def test_unknown_provenance_is_explicit():
    value = Provenance[str]()
    assert value.value is None
    assert value.origin == ValueOrigin.UNKNOWN


def test_banner_renders():
    console = Console(record=True, width=120)
    render_banner(console)
    assert "KERNL" in console.export_text()


def test_scan_returns_real_values():
    profile = scan()
    assert profile.cpu_threads.value is not None
    assert profile.ram_total_bytes.value is not None
    assert profile.storage_free_bytes.value is not None
    assert profile.platform.origin == ValueOrigin.DETECTED


def test_profile_round_trip(tmp_path):
    store = Store(tmp_path / "kernl.sqlite3")
    profile = HardwareProfile(
        platform=Provenance(value="test", origin=ValueOrigin.DETECTED),
        architecture=Provenance(),
        cpu_model=Provenance(),
        cpu_cores=Provenance(),
        cpu_threads=Provenance(),
        ram_total_bytes=Provenance(),
        ram_available_bytes=Provenance(),
        gpu_names=Provenance(),
        gpu_driver_versions=Provenance(),
        storage_free_bytes=Provenance(),
        backends=Provenance(),
        scanned_at="now",
    )
    store.save_profile(profile)
    restored = store.latest_profile()
    assert restored.platform.value == "test"
    assert restored.gpu_memory_bytes.value is None
    assert restored.gpu_memory_bytes.origin == ValueOrigin.UNKNOWN


def test_display_profile_formats_vram_in_gib(monkeypatch):
    field_names = (
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
    profile = SimpleNamespace(
        **{name: Provenance() for name in field_names},
        warnings=[],
    )
    profile.gpu_memory_bytes = Provenance(
        value=[4 * 1024**3, 512 * 1024**2],
        origin=ValueOrigin.DETECTED,
        source="nvidia-smi + WMI",
    )
    console = Console(record=True, width=180)
    monkeypatch.setattr("kernl.cli.app.console", console)

    display_profile(profile)

    output = console.export_text()
    assert "4.00 GiB" in output
    assert "0.50 GiB" in output


def test_package_version_matches_project_metadata():
    assert __version__ == package_version("kernl")
