from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class RuntimeProcessRecord:
    pid: int
    config_name: str
    executable: str
    model_path: str
    started_at: float
    log_path: str
    status: str = "running"


class ProcessStore:
    """Persist managed runtime processes across CLI invocations."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self) -> list[RuntimeProcessRecord]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        records: list[RuntimeProcessRecord] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                records.append(RuntimeProcessRecord(**item))
            except TypeError:
                continue
        return sorted(records, key=lambda record: record.started_at)

    def get(self, pid: int) -> RuntimeProcessRecord | None:
        return next((record for record in self.list() if record.pid == pid), None)

    def add(self, record: RuntimeProcessRecord) -> None:
        records = [existing for existing in self.list() if existing.pid != record.pid]
        records.append(record)
        self._write(records)

    def update(self, pid: int, **changes: object) -> RuntimeProcessRecord:
        records = self.list()
        updated: RuntimeProcessRecord | None = None
        for index, record in enumerate(records):
            if record.pid == pid:
                updated = replace(record, **changes)  # type: ignore[arg-type]
                records[index] = updated
                break
        if updated is None:
            raise KeyError(f"unknown managed process: {pid}")
        self._write(records)
        return updated

    def remove(self, pid: int) -> None:
        records = [record for record in self.list() if record.pid != pid]
        self._write(records)

    def _write(self, records: list[RuntimeProcessRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(
            [asdict(record) for record in records], indent=2, sort_keys=True
        )
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)
