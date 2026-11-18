from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Cache:
    """Thread-safe sqlite key/value cache. path=None keeps it in memory; enabled=False disables it."""

    def __init__(self, path: Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            self._conn.commit()

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        data = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, data))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
