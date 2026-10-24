"""Loading of the raw input tables with explicit, documented parsing rules.

Parsing decisions:
* Files are read with ``utf-8-sig`` because the published CSVs start with a
  UTF-8 byte-order mark; otherwise the first column name would contain it.
* The metadata table uses the literal string ``"NA"`` (R's export convention)
  for missing values. We read everything as text and convert only ``"NA"`` and
  empty cells to missing, so values such as "unclassified" are never lost.
* The domain matrix is read as text first and converted afterwards, so that
  non-numeric cells are *detected* by validation instead of crashing pandas or
  being coerced silently.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xanthomonas_ml.config import Config
from xanthomonas_ml.utils import get_logger

log = get_logger("loader")

NA_STRINGS = {"NA", ""}


@dataclass
class RawDataset:
    """Raw tables exactly as published (no rows or columns removed).

    Attributes:
        features: Domain matrix with the sample id column as a regular column.
        metadata: Genome metadata with the sample id column as a regular column.
        descriptions: Optional Pfam accession -> description table.
    """

    features: pd.DataFrame
    metadata: pd.DataFrame
    descriptions: pd.DataFrame | None


def discover_files(raw_dir: Path) -> list[Path]:
    """List candidate data files (CSV/TSV) in ``raw_dir`` sorted by name."""
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".tsv"})


def _read_text_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV with every cell as a string (no implicit NA or type inference)."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load the domain presence/absence matrix as text.

    Returns:
        DataFrame whose first column is the sample identifier and remaining
        columns are features. Values are strings; see :func:`to_numeric_matrix`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the table has no feature columns.
    """
    df = _read_text_table(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path} must contain an identifier column and at least one feature column")
    # Strip stray spaces so that " PF00001" and "PF00001" are treated as the same column.
    df.columns = [c.strip() for c in df.columns]
    # pandas silently renames repeated headers ("PF1", "PF1.1"), so duplicates
    # are detected on the raw header line and kept for the validation report.
    df.attrs["duplicate_columns"] = find_duplicate_header_names(path)
    return df


def find_duplicate_header_names(path: Path) -> list[str]:
    """Return column names that occur more than once in the header of ``path``."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=sep), [])
    counts = Counter(name.strip() for name in header)
    return sorted(name for name, n in counts.items() if n > 1)


def load_metadata(path: Path) -> pd.DataFrame:
    """Load genome metadata; ``"NA"``/empty cells become missing values."""
    df = _read_text_table(path)
    df.columns = [c.strip() for c in df.columns]
    # Trim each cell, then replace "NA" and empty cells with real missing values (NaN).
    return df.apply(lambda col: col.str.strip().where(~col.str.strip().isin(NA_STRINGS), np.nan))


def load_descriptions(path: Path | None) -> pd.DataFrame | None:
    """Load optional Pfam descriptions (columns ``accession``, ``description``)."""
    if path is None or not path.is_file():
        return None
    df = _read_text_table(path)
    if not {"accession", "description"} <= set(df.columns):
        log.warning("%s lacks 'accession'/'description' columns; descriptions ignored", path.name)
        return None
    return df[["accession", "description"]]


def to_numeric_matrix(features: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Convert a text feature table to a numeric matrix indexed by sample id.

    Non-numeric cells become NaN here; validation reports them explicitly
    before any analysis is allowed to continue.
    """
    if id_column not in features.columns:
        raise KeyError(f"Identifier column '{id_column}' not found in feature table")
    numeric = features.set_index(id_column).apply(pd.to_numeric, errors="coerce")
    # Trim ids so they match the metadata keys exactly when the tables are joined.
    numeric.index = numeric.index.astype(str).str.strip()
    numeric.index.name = id_column
    return numeric


def load_raw_dataset(paths: dict[str, Path]) -> RawDataset:
    """Load the verified raw files returned by ``require_valid_files``."""
    return RawDataset(
        features=load_feature_matrix(paths["domain_matrix"]),
        metadata=load_metadata(paths["genome_information"]),
        descriptions=load_descriptions(paths.get("domain_descriptions")),
    )


def load_from_config(cfg: Config, allow_checksum_mismatch: bool = False) -> RawDataset:
    """Verify the local files configured in ``cfg`` and load them."""
    from xanthomonas_ml.dataset_download import require_valid_files

    return load_raw_dataset(require_valid_files(cfg, allow_checksum_mismatch=allow_checksum_mismatch))
