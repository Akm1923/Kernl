from __future__ import annotations
import json
from pathlib import Path
from kernl.runtime.config import RuntimeConfig

class ConfigStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, config: RuntimeConfig) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("configuration name must be a file name without directories")
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(config.export(), indent=2), encoding="utf-8")
        return path

    def load(self, name: str) -> RuntimeConfig:
        if not name or Path(name).name != name:
            raise ValueError("configuration name must be a file name without directories")
        path = self.root / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return RuntimeConfig.model_validate_json(path.read_text(encoding="utf-8"))
