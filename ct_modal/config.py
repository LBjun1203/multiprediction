"""Small YAML configuration loader with dataclass validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ct_preparation.manifest import CTInputSpec, resolve_ct_input_spec


@dataclass(frozen=True)
class ExperimentConfig:
    ct: CTInputSpec
    values: dict[str, Any]
    source: Path

    def resolved_dict(self) -> dict[str, Any]:
        return {**self.values, "ct": self.ct.to_dict()}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    result: dict[str, Any] = {}
    active_mapping: dict[str, Any] | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip())
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML at {path}:{line_number}")
        key, raw_value = (part.strip() for part in stripped.split(":", 1))
        if not key:
            raise ValueError(f"Empty YAML key at {path}:{line_number}")
        if indent == 0:
            if raw_value == "":
                nested: dict[str, Any] = {}
                result[key] = nested
                active_mapping = nested
            else:
                result[key] = _parse_scalar(raw_value)
                active_mapping = None
        elif active_mapping is not None:
            active_mapping[key] = _parse_scalar(raw_value)
        else:
            raise ValueError(f"Unexpected YAML indentation at {path}:{line_number}")
    return result


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip("'\"")


def load_ct_config(path: str | Path, *, require_trainable: bool = True) -> CTInputSpec:
    source = Path(path)
    value = _read_yaml(source)
    if "ct" in value:
        value = value["ct"]
    if not isinstance(value, Mapping):
        raise ValueError(f"ct configuration must be a mapping: {source}")
    return resolve_ct_input_spec(value, require_trainable=require_trainable)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    values = _read_yaml(source)
    ct_reference = values.pop("ct_config", None)
    inline = values.pop("ct", None)
    if ct_reference is not None and inline is not None:
        raise ValueError(
            "Experiment config must use either ct_config or inline ct, not both"
        )
    if ct_reference is not None:
        ct_path = Path(str(ct_reference))
        if not ct_path.is_absolute():
            ct_path = source.parent / ct_path
        ct = load_ct_config(ct_path)
    else:
        ct = resolve_ct_input_spec(inline)
    return ExperimentConfig(ct=ct, values=values, source=source.resolve())


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config.resolved_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
