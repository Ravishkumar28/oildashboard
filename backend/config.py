"""Loads configuration from a project-root .env file into os.environ.

A tiny zero-dependency loader — no python-dotenv needed. Values already set
in the real environment win over the .env file."""
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env() -> None:
    if not _ENV_PATH.exists():
        return
    for raw in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

TWELVE_DATA_API_KEY: str = os.environ.get("TWELVE_DATA_API_KEY", "")
EIA_API_KEY: str = os.environ.get("EIA_API_KEY", "")
AIS_API_KEY: str = os.environ.get("AIS_API_KEY", "")
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")
