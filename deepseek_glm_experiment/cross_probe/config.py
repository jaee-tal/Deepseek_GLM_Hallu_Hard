"""Experiment configuration (single YAML), model/dataset registries and run-directory layout.

Every run is identified by (model, dataset, tag).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .vendor.io_utils import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
CLOUD_CONFIG = PROJECT_ROOT / "configs" / "cloud_experiment.yaml"
_ENV = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str) and (m := _ENV.match(value)):
        return os.environ.get(m.group(1), m.group(2) or "")
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"configuration inheritance cycle at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        raw = _merge(_read_config(parent_path, seen), raw)
    seen.remove(path)
    return raw


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self.raw.get(k, default)

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    # ---------------------------------------------------------------- registries
    def model_name(self, name: str | None = None) -> str:
        return name or self.raw.get("default_model") or "qwen3_8b"

    def dataset_name(self, name: str | None = None) -> str:
        return name or self.raw.get("default_dataset") or "pop_qa"

    def model(self, name: str | None = None) -> dict[str, Any]:
        """Return a copy of the requested subject-model configuration."""
        n = self.model_name(name)
        try:
            return copy.deepcopy(self.raw["models"][n])
        except KeyError:
            raise SystemExit(f"unknown model {n!r}; known: {sorted(self.raw.get('models', {}))}")

    def dataset(self, name: str | None = None) -> dict[str, Any]:
        n = self.dataset_name(name)
        try:
            return copy.deepcopy(self.raw["datasets"][n])
        except KeyError:
            raise SystemExit(f"unknown dataset {n!r}; known: {sorted(self.raw.get('datasets', {}))}")

    def generation(self, model: str | None = None, dataset: str | None = None) -> dict[str, Any]:
        """Resolve base generation settings, dataset overrides, then model/style overrides."""
        d = self.dataset(dataset)
        gen = dict(self.raw["generation"])
        gen.update(d.get("generation_overrides") or {})
        mo = (self.model(model).get("generation_overrides") or {}).get(d["prompt_style"], {})
        gen.update(mo)
        return gen

    # ---------------------------------------------------------------- paths
    def data_dir(self, dataset: str | None = None) -> Path:
        base = PROJECT_ROOT / self.raw["paths"]["data_dir"]
        return base if dataset is None else base / self.dataset_name(dataset)

    def run_dir(self, tag: str, model: str | None = None, dataset: str | None = None) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["runs_dir"] / self.model_name(model) / self.dataset_name(dataset) / tag

    def summary_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["runs_dir"] / "summary"

    def hf_home(self) -> Path:
        p = Path(self.raw["paths"]["hf_home"])
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path | str | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    raw = _expand_env(_read_config(p))
    return Config(raw=raw, path=p)


def questions_path(cfg: Config, dataset: str | None = None) -> Path:
    return cfg.data_dir(dataset) / "questions.jsonl"


def split_path(cfg: Config, dataset: str | None = None) -> Path:
    return cfg.data_dir(dataset) / f"split_seed{cfg['split']['seed']}.json"
