import json
from pathlib import Path
from typing import Any


def resolve_config_path(
    config_path: str | None,
    default_relative_path: str,
    caller_file: str,
) -> Path:
    project_root = Path(caller_file).resolve().parents[1]
    if config_path:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
    else:
        candidate = project_root / default_relative_path
    return candidate.resolve()


def load_json_config(
    config_path: str | None,
    default_relative_path: str,
    caller_file: str,
) -> dict[str, Any]:
    path = resolve_config_path(config_path, default_relative_path, caller_file)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

