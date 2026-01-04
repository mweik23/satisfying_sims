from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class LoadedPreset:
    preset_path: Path
    resolved: Dict[str, Any]
    loaded_files: Tuple[Path, ...]  # includes + preset itself


def _deep_merge(base: Any, override: Any) -> Any:
    """
    Merge override into base and return merged value.

    Rules:
      - dict + dict: recursive merge
      - list: override replaces base (no concatenation)
      - scalars: override replaces base
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    if isinstance(override, list):
        return list(override)

    return override


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def load_preset(preset_path: str | Path) -> LoadedPreset:
    """
    Load a preset YAML that may contain:

      include:
        - other.yaml
        - path/to/more.yaml

    Returns a fully merged dict plus the list of files that were loaded.
    """
    preset_path = Path(preset_path).expanduser().resolve()
    preset_dir = preset_path.parent

    preset_data = _load_yaml(preset_path)

    include_list = preset_data.get("include", [])
    if include_list is None:
        include_list = []
    if not isinstance(include_list, list):
        raise ValueError(f"'include' must be a list in {preset_path}")

    loaded: List[Path] = []
    merged: Dict[str, Any] = {}

    # 1) Load includes first
    for rel in include_list:
        if not isinstance(rel, str):
            raise ValueError(f"include entries must be strings. Got {type(rel)} in {preset_path}")
        inc_path = (preset_dir / rel).expanduser().resolve()
        inc_data = _load_yaml(inc_path)
        merged = _deep_merge(merged, inc_data)
        loaded.append(inc_path)

    # 2) Apply preset keys last (excluding 'include')
    preset_overrides = dict(preset_data)
    preset_overrides.pop("include", None)
    merged = _deep_merge(merged, preset_overrides)

    # Track preset itself too
    loaded.append(preset_path)

    return LoadedPreset(
        preset_path=preset_path,
        resolved=merged,
        loaded_files=tuple(loaded),
    )
