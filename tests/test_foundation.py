import pytest
from rich.console import Console
from kernl.cli.banner import render_banner
from kernl.core.types import Provenance, ValueOrigin
from kernl.hardware.scanner import scan
from kernl.storage.db import Store
from kernl.core.types import HardwareProfile

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
        architecture=Provenance(), cpu_model=Provenance(), cpu_cores=Provenance(), cpu_threads=Provenance(),
        ram_total_bytes=Provenance(), ram_available_bytes=Provenance(), gpu_names=Provenance(),
        gpu_driver_versions=Provenance(), storage_free_bytes=Provenance(), backends=Provenance(), scanned_at="now",
    )
    store.save_profile(profile)
    assert store.latest_profile().platform.value == "test"
