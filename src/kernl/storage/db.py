from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from kernl.core.types import HardwareProfile


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS hardware_profiles (id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS benchmark_results (id INTEGER PRIMARY KEY AUTOINCREMENT, config_name TEXT NOT NULL, measured_at REAL NOT NULL, payload TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_actions (id INTEGER PRIMARY KEY AUTOINCREMENT, tool TEXT NOT NULL, arguments TEXT NOT NULL, approved INTEGER NOT NULL, result_summary TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def save_profile(self, profile: HardwareProfile) -> None:
        self.conn.execute(
            "INSERT INTO hardware_profiles(scanned_at,payload) VALUES(?,?)",
            (profile.scanned_at, profile.model_dump_json()),
        )
        self.conn.commit()

    def latest_profile(self) -> HardwareProfile | None:
        row = self.conn.execute(
            "SELECT payload FROM hardware_profiles ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return HardwareProfile.model_validate_json(row[0]) if row else None

    def save_benchmark(self, config_name: str, payload: str, measured_at: float) -> int:
        cursor = self.conn.execute(
            "INSERT INTO benchmark_results(config_name,measured_at,payload) VALUES(?,?,?)",
            (config_name, measured_at, payload),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def latest_benchmark(self, config_name: str) -> dict[str, object] | None:
        row = self.conn.execute(
            "SELECT payload FROM benchmark_results WHERE config_name=? ORDER BY id DESC LIMIT 1",
            (config_name,),
        ).fetchone()
        if not row:
            return None
        payload = json.loads(row[0])
        return payload if isinstance(payload, dict) else None

    def record_agent_action(
        self,
        tool: str,
        arguments: dict[str, object],
        approved: bool,
        result_summary: str,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO agent_actions(tool,arguments,approved,result_summary,created_at) "
            "VALUES(?,?,?,?,?)",
            (
                tool,
                json.dumps(arguments, sort_keys=True),
                int(approved),
                result_summary,
                time.time(),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def recent_agent_actions(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT id,tool,arguments,approved,result_summary,created_at "
            "FROM agent_actions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "tool": row[1],
                "arguments": json.loads(row[2]),
                "approved": bool(row[3]),
                "result_summary": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]
