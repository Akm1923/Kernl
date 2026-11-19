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
- GGUF metadata inspection and model-memory estimation, including GQA-aware
  KV-cache estimates when attention-head metadata is available.
- Hardware-aware `kernl model plan` estimates for CPU RAM and a selected GPU's
  VRAM. It accounts for GPU layer offload, KV-cache placement/type, per-device
  VRAM (never summed implicitly), and conservative memory reserves.
- Runtime configs can keep the KV cache on CPU with `--kv-cache-on-cpu`; the
  generated llama.cpp arguments include `--no-kv-offload`.
- Hugging Face model discovery and model recommendations.
  Compatibility is tri-state: known fit, known constraint failure, or unknown
  until sufficient memory metadata is available. Missing model size is never
  treated as proof of compatibility.
- Verified runtime artifact installation and local runtime configuration.
- SQLite persistence for settings and hardware profiles.

Backend detection is a capability signal, not proof that a particular model can
run on a particular GPU. Runtime compatibility and actual inference performance
must still be validated with the selected llama.cpp build and model.

`kernl model plan` is an estimate, not a guarantee. It uses the GGUF file size as a
weight-size proxy, approximates GPU layer placement with uniform block sizes, and
keeps a CPU-resident allowance for non-block tensors. KV-cache sizing uses the
model's KV attention-head count when available; otherwise it falls back to the
full embedding width. The planner reserves 20% of detected total VRAM because
current hardware scanning does not measure free VRAM. Review its warnings and
use a real llama.cpp load/benchmark before relying on a configuration.

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
kernl model inspect .\\models\\your-model.gguf
kernl model plan .\\models\\your-model.gguf --gpu-layers 20
```

The CI matrix runs the test suite on Windows and Ubuntu with Python 3.11 and 3.13.
