# Kernl

Kernl is a Windows-first, CLI-first local AI compute platform.

## Current foundation

The `feature/foundation` branch contains the first real foundation implementation:

- Rich startup banner with compact terminal fallback
- Typer CLI with onboarding, scan, profile, provider, doctor, and restricted agent commands
- Real host CPU, RAM, storage, platform, and Windows video-controller probing where available
- Provenance labels for detected, measured, estimated, and unknown values
- SQLite persistence for settings and hardware profiles
- OS keyring storage for provider credentials
- Pytest coverage for core foundation behavior

## Truthful status

Model discovery, runtime installation/execution, downloads, compatibility scoring, and benchmarks are not claimed as complete yet. They will be added only with real implementations and tests.

## Development

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -e ".[test]"
pytest
kernl init
kernl scan
kernl profile
```
