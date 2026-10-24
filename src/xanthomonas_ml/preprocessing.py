"""Construction of the analysis dataset (samples x features, binary target).

Only *label-free, sample-independent* operations happen here, before any
cross-validation:

1. Samples are aligned by identifier (inner join of features and metadata).
2. Samples are excluded, with a recorded reason, when they lack a label, carry
   a label outside {pathogenic, non-pathogenic} (e.g. 'unknown'), or belong to
   another genus. Labels are never re-mapped between categories.
3. Features that are constant across *all* retained strains are dropped. This
   cannot leak information: a feature constant in the full set is constant in
   every training fold and would be removed by the in-fold prevalence filter
   anyway; dropping it early only saves memory.

All data-dependent filtering that could use information from validation
samples (prevalence and correlation filters, hyper-parameter tuning) is done
inside each training fold; see :mod:`xanthomonas_ml.feature_engineering`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from xanthomonas_ml.config import Config
from xanthomonas_ml.data_loader import RawDataset, to_numeric_matrix
from xanthomonas_ml.utils import ensure_dir, get_logger, read_json, sha256_file, utc_now_iso, write_json

log = get_logger("preprocessing")

PROCESSED_FILES = {
    "features": "features.csv.gz",
    "labels": "labels.csv",
    "samples": "sample_metadata.csv",
    "removed_samples": "removed_samples.csv",
    "removed_features": "removed_features.csv",
    "metadata": "preparation_metadata.json",
}
METADATA_COLUMNS = ["strain", "alias_of", "genus", "species", "subspecies", "hostGenus", "hostSpecies", "source"]


class PreparationError(RuntimeError):
    """Raised when a scientifically valid analysis set cannot be built."""


@dataclass
class PreparedData:
    """Aligned analysis dataset.

    Attributes:
        X: Binary feature matrix (int8), index = sample id, columns = Pfam ids.
        y: Target vector (1 = pathogenic, 0 = non-pathogenic), same index as X.
        samples: Per-sample metadata (strain, species, host, original label ...).
        removed_samples: Excluded samples with the reason for exclusion.
        removed_features: Dropped features with the reason for removal.
        info: Provenance metadata (source checksums, counts, settings).
    """

    X: pd.DataFrame
    y: pd.Series
    samples: pd.DataFrame
    removed_samples: pd.DataFrame
    removed_features: pd.DataFrame
    info: dict[str, Any]

    @property
    def groups(self) -> pd.Series:
        """Species label per sample, used for species-grouped cross-validation.

        Strains without a species assignment ('unclassified') are treated as a
        single group, which is conservative (they are always held out together).
        """
        species = self.samples["species"].fillna("unclassified")
        return (self.samples["genus"].fillna("") + " " + species).str.strip()


def label_mapping(cfg: Config) -> dict[str, int]:
    """Source label -> integer class. Only the two configured labels are mapped."""
    return {cfg["labels"]["positive"]: 1, cfg["labels"]["negative"]: 0}


def encode_labels(labels: pd.Series, mapping: dict[str, int]) -> tuple[pd.Series, pd.Series]:
    """Encode labels; return (encoded, rejected) where rejected holds unmapped values."""
    # Anything not in the mapping (e.g. 'unknown' or a missing value) becomes NaN.
    encoded = labels.map(mapping)
    rejected = labels[encoded.isna()]
    return encoded.dropna().astype(int), rejected


def class_balance(y: pd.Series) -> dict[str, Any]:
    """Counts and fractions per class of an encoded target vector."""
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    return {
        "n_samples": total,
        "counts": {int(k): int(v) for k, v in counts.items()},
        "fractions": {int(k): round(float(v) / total, 4) for k, v in counts.items()} if total else {},
        "minority_fraction": round(float(counts.min()) / total, 4) if total else None,
    }


def build_analysis_dataset(raw: RawDataset, cfg: Config) -> PreparedData:
    """Align, filter and encode the raw tables into an analysis dataset.

    Raises:
        PreparationError: if identifiers are duplicated or a class becomes empty.
    """
    id_col = cfg["dataset"]["sample_id_column"]
    label_col = cfg["dataset"]["label_column"]
    genus = cfg["labels"].get("required_genus")
    mapping = label_mapping(cfg)

    X_all = to_numeric_matrix(raw.features, id_col)
    meta = raw.metadata.copy()
    meta[id_col] = meta[id_col].astype(str).str.strip()
    # Duplicate ids would make the join ambiguous, so stop instead of guessing which row is right.
    if X_all.index.duplicated().any() or meta[id_col].duplicated().any():
        raise PreparationError("duplicate sample identifiers; run validate-data for details")
    if X_all.isna().any().any():
        raise PreparationError("feature matrix contains missing/non-numeric values; run validate-data")
    meta = meta.set_index(id_col)

    # Every exclusion is recorded with its reason so it can be written to removed_samples.csv.
    removed: list[dict[str, str]] = []

    def drop(ids: list[str], reason: str) -> None:
        for sample_id in ids:
            removed.append({"sample_id": sample_id, "reason": reason})

    # 1) explicit identifier alignment
    drop(sorted(set(X_all.index) - set(meta.index)), "feature profile has no metadata/label")
    drop(sorted(set(meta.index) - set(X_all.index)), "metadata row has no feature profile")
    common = [i for i in X_all.index if i in meta.index]
    meta = meta.loc[common]

    # 2) genus restriction (the study concerns Xanthomonas only)
    if genus and "genus" in meta.columns:
        for sample_id in meta.index[meta["genus"] != genus]:
            row = meta.loc[sample_id]
            drop([sample_id], f"genus is {row['genus']}, not {genus} (source label '{row[label_col]}')")
        meta = meta.loc[meta["genus"] == genus]

    # 3) labels: keep only the two well-defined phenotype classes
    y, rejected = encode_labels(meta[label_col], mapping)
    for sample_id, value in rejected.items():
        reason = "missing phenotype label" if pd.isna(value) else f"phenotype label '{value}' is not a binary class"
        drop([sample_id], reason)
    meta = meta.loc[y.index]
    X = X_all.loc[y.index].astype("int8")

    # With one class left, accuracy, ROC-AUC and the other metrics are meaningless.
    if y.nunique() < 2:
        raise PreparationError(f"analysis set contains a single class: {y.value_counts().to_dict()}")

    # 4) globally constant features (zero information; see module docstring)
    prevalence = X.mean()
    constant = prevalence[(prevalence == 0) | (prevalence == 1)]
    removed_features = pd.DataFrame(
        {
            "feature": constant.index,
            "reason": [
                "absent from all analysed strains" if p == 0 else "present in all analysed strains"
                for p in constant.values
            ],
        }
    )
    X = X.drop(columns=constant.index)

    samples = meta[[c for c in METADATA_COLUMNS if c in meta.columns]].copy()
    samples["source_label"] = meta[label_col]
    samples["target"] = y
    samples.index.name = id_col
    X.index.name = id_col
    y.index.name = id_col
    y.name = "target"

    info = {
        "created_utc": utc_now_iso(),
        "label_mapping": mapping,
        "positive_class": cfg["labels"]["positive"],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_features_in_source": int(X_all.shape[1]),
        "n_removed_samples": len(removed),
        "n_removed_features": int(len(removed_features)),
        "class_balance": class_balance(y),
        "n_species_groups": None,
    }
    prepared = PreparedData(
        X=X,
        y=y,
        samples=samples,
        removed_samples=pd.DataFrame(removed, columns=["sample_id", "reason"]),
        removed_features=removed_features,
        info=info,
    )
    info["n_species_groups"] = int(prepared.groups.nunique())
    log.info(
        "Analysis set: %d strains (%s), %d features; %d samples and %d constant features removed",
        X.shape[0], info["class_balance"]["counts"], X.shape[1], len(removed), len(removed_features),
    )
    return prepared


def _settings_fingerprint(cfg: Config) -> str:
    """Hash of the configuration parts that influence preparation (for caching)."""
    # Only dataset and label settings are hashed, so changing e.g. CV settings does not force a rebuild.
    relevant = {"dataset": cfg["dataset"], "labels": cfg["labels"]}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def _source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {key: sha256_file(path) for key, path in sorted(paths.items())}


def save_prepared(prepared: PreparedData, out_dir: Path, source_paths: dict[str, Path], cfg: Config) -> Path:
    """Persist the analysis dataset and its provenance to ``out_dir``."""
    ensure_dir(out_dir)
    prepared.X.to_csv(out_dir / PROCESSED_FILES["features"], compression="gzip")
    prepared.y.to_csv(out_dir / PROCESSED_FILES["labels"])
    prepared.samples.to_csv(out_dir / PROCESSED_FILES["samples"])
    prepared.removed_samples.to_csv(out_dir / PROCESSED_FILES["removed_samples"], index=False)
    prepared.removed_features.to_csv(out_dir / PROCESSED_FILES["removed_features"], index=False)
    info = dict(prepared.info)
    info["source_sha256"] = _source_hashes(source_paths)
    info["settings_fingerprint"] = _settings_fingerprint(cfg)
    write_json(out_dir / PROCESSED_FILES["metadata"], info)
    return out_dir


def load_prepared(out_dir: Path, cfg: Config) -> PreparedData:
    """Load a previously saved analysis dataset.

    Raises:
        FileNotFoundError: if any processed file is missing.
    """
    missing = [name for name in PROCESSED_FILES.values() if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Processed data incomplete in {out_dir}: missing {missing}. Run prepare-data.")
    id_col = cfg["dataset"]["sample_id_column"]
    # Ids are read as text so that identifiers keep their exact spelling when compared.
    X = pd.read_csv(out_dir / PROCESSED_FILES["features"], index_col=id_col, dtype={id_col: str}).astype("int8")
    y = pd.read_csv(out_dir / PROCESSED_FILES["labels"], index_col=id_col, dtype={id_col: str})["target"].astype(int)
    samples = pd.read_csv(out_dir / PROCESSED_FILES["samples"], index_col=id_col, dtype=str, keep_default_na=True)
    samples["target"] = samples["target"].astype(int)
    if not (X.index.equals(y.index) and X.index.equals(samples.index)):
        raise PreparationError("processed files are not aligned by sample identifier; re-run prepare-data --force")
    return PreparedData(
        X=X,
        y=y,
        samples=samples,
        removed_samples=pd.read_csv(out_dir / PROCESSED_FILES["removed_samples"], dtype=str),
        removed_features=pd.read_csv(out_dir / PROCESSED_FILES["removed_features"], dtype=str),
        info=read_json(out_dir / PROCESSED_FILES["metadata"]),
    )


def processed_is_current(out_dir: Path, source_paths: dict[str, Path], cfg: Config) -> bool:
    """True if saved processed data were built from the same source files and settings."""
    meta_path = out_dir / PROCESSED_FILES["metadata"]
    if not meta_path.is_file():
        return False
    try:
        info = read_json(meta_path)
    except (OSError, ValueError):
        return False
    return (
        info.get("source_sha256") == _source_hashes(source_paths)
        and info.get("settings_fingerprint") == _settings_fingerprint(cfg)
        and all((out_dir / name).is_file() for name in PROCESSED_FILES.values())
    )


def prepare_or_load(cfg: Config, force: bool = False, allow_checksum_mismatch: bool = False) -> PreparedData:
    """Return the analysis dataset, rebuilding it only when sources/settings changed.

    Validation is run before every rebuild; any ERROR stops the analysis.
    """
    from xanthomonas_ml.data_loader import load_raw_dataset
    from xanthomonas_ml.dataset_download import check_local_files, require_valid_files
    from xanthomonas_ml.validation import validate_raw_dataset

    paths = require_valid_files(cfg, allow_checksum_mismatch=allow_checksum_mismatch)
    out_dir = cfg.processed_dir
    # Cached data was validated when it was built, so validation only runs on a rebuild.
    if not force and processed_is_current(out_dir, paths, cfg):
        log.info("Reusing processed data in %s (sources unchanged)", out_dir.as_posix())
        return load_prepared(out_dir, cfg)

    raw = load_raw_dataset(paths)
    report = validate_raw_dataset(raw, cfg, check_local_files(cfg))
    report.save(cfg.reports_dir)
    if not report.ok:
        raise PreparationError(
            "Data validation failed; see reports/data_validation.md:\n  - "
            + "\n  - ".join(f"{i.code}: {i.message}" for i in report.errors)
        )
    prepared = build_analysis_dataset(raw, cfg)
    save_prepared(prepared, out_dir, paths, cfg)
    log.info("Processed data saved to %s", out_dir.as_posix())
    return prepared
