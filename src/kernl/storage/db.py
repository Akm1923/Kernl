from __future__ import annotations
import sqlite3
from pathlib import Path
from kernl.core.types import HardwareProfile

class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS hardware_profiles (id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT NOT NULL, payload TEXT NOT NULL)")
        self.conn.commit()
    def get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else None
    def set(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value)); self.conn.commit()
    def save_profile(self, profile: HardwareProfile) -> None:
        self.conn.execute("INSERT INTO hardware_profiles(scanned_at,payload) VALUES(?,?)", (profile.scanned_at, profile.model_dump_json())); self.conn.commit()
    def latest_profile(self) -> HardwareProfile | None:
        row = self.conn.execute("SELECT payload FROM hardware_profiles ORDER BY id DESC LIMIT 1").fetchone()
        return HardwareProfile.model_validate_json(row[0]) if row else None
