"""Centralised configuration loading.

The configuration lives in ``configs/config.yaml``. Relative paths inside it are
resolved against the *project root*, defined as the parent directory of the
folder containing the YAML file. This keeps every path relative and portable
between Windows and Ubuntu.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV_VAR = "XANTHOMONAS_ML_CONFIG"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


@dataclass
class Config:
    """Parsed configuration plus the project root used to resolve paths.

    Attributes:
        root: Project root directory; all relative paths are resolved from it.
        data: The raw nested dictionary loaded from YAML.
        source: Path of the YAML file the configuration came from.
    """

    root: Path
    data: dict[str, Any]
    source: Path | None = None
    _required_sections: tuple[str, ...] = field(
        default=("paths", "dataset", "labels", "preprocessing", "cv", "models"), repr=False
    )

    def __post_init__(self) -> None:
        # Fail at load time, not later in the pipeline, if a required section is absent.
        missing =[s for s in self._required_sections if s not in self.data]
        if missing:
            raise ConfigError(f"Configuration {self.source} is missing sections: {missing}")

    # --- convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def seed(self) -> int:
        return int(self.data.get("seed", 42))

    def path(self, key: str) -> Path:
        """Return the absolute path for an entry of the ``paths`` section."""
        try:
            value = self.data["paths"][key]
        except KeyError as exc:
            raise ConfigError(f"Unknown path key '{key}' in configuration") from exc
        p = Path(value)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def raw_dir(self) -> Path:
        return self.path("raw_data")

    @property
    def processed_dir(self) -> Path:
        return self.path("processed_data")

    @property
    def models_dir(self) -> Path:
        return self.path("models")

    @property
    def outputs_dir(self) -> Path:
        return self.path("outputs")

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    @property
    def reports_dir(self) -> Path:
        return self.path("reports")

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with nested keys overridden (used by tests and CLI flags)."""
        merged = copy.deepcopy(self.data)
        _deep_update(merged, overrides)
        return Config(root=self.root, data=merged, source=self.source)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    # Merge nested dicts key by key; any other value replaces the old one.
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate the configuration file.

    Search order: explicit argument, ``$XANTHOMONAS_ML_CONFIG``,
    ``./configs/config.yaml`` in the current directory, then the repository
    root relative to this source file (works for editable installs).

    Raises:
        ConfigError: if no configuration file can be found.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(CONFIG_ENV_VAR):
        candidates.append(Path(os.environ[CONFIG_ENV_VAR]))
    candidates.append(Path.cwd() / DEFAULT_CONFIG_RELATIVE)
    # src/xanthomonas_ml/config.py -> parents[2] is the repository root.
    candidates.append(Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_RELATIVE)

    # Use the first existing candidate. An explicit path that is missing is an error,
    # not a reason to silently fall back to another config.
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if explicit and candidate == Path(explicit):
            raise ConfigError(f"Configuration file not found: {candidate}")
    raise ConfigError(
        "Could not find configs/config.yaml. Run commands from the project root or pass --config PATH."
    )


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate the YAML configuration.

    Args:
        path: Optional explicit path to a YAML file.

    Returns:
        A :class:`Config` whose root is the parent of the config file's folder.

    Raises:
        ConfigError: if the file is missing or not a YAML mapping.
    """
    config_file = find_config_file(path)
    try:
        with config_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{config_file} must contain a YAML mapping")
    return Config(root=config_file.parent.parent, data=data, source=config_file)
