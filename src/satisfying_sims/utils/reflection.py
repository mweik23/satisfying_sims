from __future__ import annotations

from typing import Any

_MISSING = object()

def get_path(obj: Any, path: str, default: Any = None) -> Any:
    """
    Resolve dotted attribute paths like "a_kind.type" against an object.

    Returns `default` if any attribute along the path is missing.
    """
    cur: Any = obj
    for part in path.split("."):
        cur = getattr(cur, part, _MISSING)
        if cur is _MISSING:
            return default
    return cur

def get_class(name: str, module):
    target = name.lower()
    for key, obj in module.__dict__.items():
        if isinstance(obj, type) and key.lower() == target:
            return obj
    raise ValueError(f"Class '{name}' not found in module {module.__name__}")
