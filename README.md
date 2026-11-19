# Kernl

Kernl is a Windows-first, CLI-first local AI compute platform. It profiles a
machine, inspects local GGUF models, estimates memory requirements, and helps
prepare a local llama.cpp runtime.

## Implemented capabilities

- Rich CLI startup, provider onboarding, and secure API-key storage through the
  operating-system keyring.
- Hardware profile for CPU, RAM, storage, Windows display adapters, and driver
  versions.
- GPU memory reporting with explicit provenance:
  - NVIDIA adapters use nvidia-smi memory totals.
  - Other adapters use Windows CIM AdapterRAM only below the 4-GiB saturation
    boundary.
  - Unsupported or ambiguous values remain UNKNOWN; Kernl does not substitute
    zero or guess.
- GGUF metadata inspection and model-memory estimation.
- Hugging Face model discovery, compatibility assessment, and recommendation.
- Verified runtime artifact installation and local runtime configuration.
- SQLite persistence for settings and hardware profiles.

Backend detection is a capability signal, not proof that a particular model can
run on a particular GPU. Runtime compatibility and actual inference performance
must still be validated with the selected llama.cpp build and model.

## Development

Python 3.11 or newer is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
kernl init
kernl scan
kernl profile
```

The CI matrix runs the test suite on Windows and Ubuntu with Python 3.11 and 3.13.
