"""Utilities for saving and loading EQ presets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PRESET_VERSION = 1


def save_preset(path: str | Path, bands: List[Dict[str, Any]]) -> None:
    """Write a preset JSON file."""
    data = {"version": PRESET_VERSION, "bands": bands}
    Path(path).write_text(json.dumps(data, indent=2))


def load_preset(path: str | Path) -> List[Dict[str, Any]]:
    obj = json.loads(Path(path).read_text())
    return obj.get("bands", []) 