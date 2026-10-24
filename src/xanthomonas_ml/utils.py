"""Small shared helpers: logging, hashing, JSON I/O and directory handling."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "xanthomonas_ml"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the package logger."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


class _CurrentStderrHandler(logging.StreamHandler):
    """Stream handler that always writes to the *current* ``sys.stderr``.

    A plain StreamHandler keeps the stream it was created with; if stderr is
    later replaced (e.g. by a test harness or an embedding application) that
    stale stream may be closed.
    """

    @property
    def stream(self):  # type: ignore[override]
        import sys

        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # StreamHandler.__init__ assigns it; ignore
        pass


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging for the package (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, _CurrentStderrHandler) for h in logger.handlers):
        handler = _CurrentStderrHandler()
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        logger.addHandler(handler)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Current UTC time in ISO-8601 format (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(obj: Any) -> Any:
    # numpy scalars/arrays and paths are not JSON-serialisable by default.
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _replace_nan(obj: Any) -> Any:
    """Recursively turn float NaN into None so the JSON stays standard-compliant."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _replace_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_replace_nan(v) for v in obj]
    return obj


def write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON atomically (temp file + rename)."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_replace_nan(payload), handle, indent=2, default=_json_default, allow_nan=False)
        handle.write("\n")
    # Rename in one step so an interrupted run never leaves a half-written file behind.
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file written by :func:`write_json`."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs. scikit-learn estimators get ``random_state`` explicitly."""
    random.seed(seed)
    np.random.seed(seed)


def relpath(path: Path, start: Path) -> str:
    """POSIX-style relative path for reports (falls back to absolute)."""
    try:
        return Path(os.path.relpath(path, start)).as_posix()
    except ValueError:  # different drives on Windows
        return path.as_posix()
