"""Prediction for new strains from Pfam presence/absence profiles.

Input contract (documented in README and docs/METHODOLOGY.md):

* A CSV/TSV with one row per strain, an identifier column (default ``gca``;
  configurable with ``--id-column``) and one column per Pfam accession
  (e.g. ``PF13855``) holding 1 (domain present) or 0 (absent).
* The domains must be annotated the same way as the training genomes
  (Prodigal gene calls + InterProScan/Pfam via SAPP in the reference study).
  ``build_feature_table`` converts InterProScan TSV output to this format.
* Raw DNA sequences are NOT accepted: the model only understands domain
  profiles, and no sequence-to-domain pipeline is bundled here.

Handling rules:
* Required features = the domains the trained pipeline actually uses after
  its filters. Missing required features are an error, unless the caller
  explicitly opts in to treating them as absent (0) - this is risky because
  'not annotated' is not the same as 'absent', and is reported per sample.
* Extra columns (other Pfam domains) are ignored and counted.
* Values must be 0/1 (``true``/``false`` and ``1.0``/``0.0`` are accepted);
  anything else, including blanks, is an error naming the offending cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xanthomonas_ml import RESEARCH_DISCLAIMER
from xanthomonas_ml.utils import ensure_dir, get_logger

log = get_logger("predict")

TRUE_STRINGS = {"1", "1.0", "true", "yes", "present"}
FALSE_STRINGS = {"0", "0.0", "false", "no", "absent"}


class PredictionInputError(ValueError):
    """Raised when an input feature file does not satisfy the model's schema."""


def read_feature_file(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV as text (no type inference), stripping a BOM if present."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _encode_binary(column: pd.Series) -> pd.Series:
    """Map textual 0/1 encodings to integers; invalid cells become NaN."""
    # Only the listed spellings count as valid; anything else stays NaN and is reported as an error.
    lowered = column.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=column.index, dtype=float)
    out[lowered.isin(TRUE_STRINGS)] = 1
    out[lowered.isin(FALSE_STRINGS)] = 0
    return out


def validate_input(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    id_column: str = "gca",
    fill_missing_as_absent: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and align an input table to the model's feature schema.

    Args:
        df: Text table from :func:`read_feature_file`.
        bundle: Loaded model bundle.
        id_column: Name of the sample identifier column.
        fill_missing_as_absent: Treat missing *required* columns as 0.

    Returns:
        (X, report) where X has exactly the model's input schema (index = id)
        and report lists ignored/missing/filled columns.

    Raises:
        PredictionInputError: on any schema or encoding problem.
    """
    if df.empty:
        raise PredictionInputError("input file contains no samples")
    if id_column not in df.columns:
        raise PredictionInputError(
            f"identifier column '{id_column}' not found; columns start with {list(df.columns[:5])}. "
            "Use --id-column to name it."
        )
    # Identifiers become the sample_id column of the output, so they must be non-empty and unique.
    ids = df[id_column].astype(str).str.strip()
    if (ids == "").any():
        raise PredictionInputError(f"{int((ids == '').sum())} rows have an empty identifier")
    dup = sorted(ids[ids.duplicated()].unique())
    if dup:
        raise PredictionInputError(f"duplicate sample identifiers: {dup[:10]}")

    schema: list[str] = bundle["feature_schema"]
    required: list[str] = bundle["selected_features"]
    present = set(df.columns) - {id_column}
    # A missing required domain is an error unless the caller explicitly accepts it as absent.
    missing_required = [f for f in required if f not in present]
    extra = sorted(present - set(schema))
    if missing_required and not fill_missing_as_absent:
        raise PredictionInputError(
            f"{len(missing_required)} feature(s) required by the model are missing, e.g. {missing_required[:10]}. "
            "Provide them (0/1) or re-run with --fill-missing-as-absent if you are sure that a missing column "
            "means the domain is absent from the genome."
        )

    features = df.drop(columns=[id_column])
    encoded = features.apply(_encode_binary)
    used_cols = [c for c in encoded.columns if c in schema]
    # Only columns the model uses are checked; bad values in ignored columns do not block prediction.
    bad = encoded[used_cols].isna()
    if bad.values.any():
        rows, cols = np.nonzero(bad.to_numpy())
        examples = [f"{ids.iloc[r]}/{used_cols[c]}={features[used_cols[c]].iloc[r]!r}" for r, c in zip(rows[:10], cols[:10])]
        raise PredictionInputError(
            f"{int(bad.values.sum())} feature cells are not valid presence/absence values (use 0 or 1): {examples}"
        )
    # Schema columns not used by the fitted model are irrelevant to the
    # prediction; they are set to 0 only to satisfy the pipeline's input width.
    X = pd.DataFrame(0, index=ids.to_numpy(), columns=schema, dtype="int8")
    for col in used_cols:
        X[col] = encoded[col].astype("int8").to_numpy()
    X.index.name = id_column
    report = {
        "n_samples": int(len(X)),
        "n_required_features": len(required),
        "missing_required_filled_as_absent": missing_required if fill_missing_as_absent else [],
        "n_extra_columns_ignored": len(extra),
        "extra_columns_ignored_examples": extra[:20],
    }
    if missing_required:
        log.warning("%d required features were missing and set to 0 (absent)", len(missing_required))
    if extra:
        log.info("%d input columns are not in the model schema and were ignored", len(extra))
    return X, report


def predict_dataframe(
    bundle: dict[str, Any], X: pd.DataFrame, report: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Predict class and probabilities for an aligned feature matrix."""
    pipeline = bundle["pipeline"]
    classes = list(pipeline.classes_)
    proba = pipeline.predict_proba(X)
    # Map class numbers (1/0) back to the original label strings for the output table.
    inverse = {v: k for k, v in bundle["label_mapping"].items()}
    pred = pipeline.predict(X)
    out = pd.DataFrame(
        {
            "sample_id": X.index,
            "predicted_class": [inverse[int(p)] for p in pred],
            "probability_pathogenic": proba[:, classes.index(1)],
            "probability_non_pathogenic": proba[:, classes.index(0)],
            "model": bundle["model_name"],
        }
    )
    filled = (report or {}).get("missing_required_filled_as_absent", [])
    out["n_required_features_filled_as_absent"] = len(filled)
    out["note"] = "research estimate only - not a diagnosis"
    return out


def predict_file(
    model_path: Path,
    input_path: Path,
    output_path: Path | None = None,
    id_column: str = "gca",
    fill_missing_as_absent: bool = False,
) -> pd.DataFrame:
    """Load a model, validate an input file, predict, and optionally save a CSV.

    Raises:
        FileNotFoundError: model or input file missing.
        PredictionInputError: input does not match the model's schema.
    """
    from xanthomonas_ml.train import load_model_bundle

    bundle = load_model_bundle(model_path)
    df = read_feature_file(input_path)
    X, report = validate_input(df, bundle, id_column=id_column, fill_missing_as_absent=fill_missing_as_absent)
    result = predict_dataframe(bundle, X, report)
    if output_path is not None:
        ensure_dir(output_path.parent)
        result.to_csv(output_path, index=False)
        log.info("Predictions written to %s", output_path.as_posix())
    # Shown on every run, because the outputs are research estimates and not diagnoses.
    log.warning(RESEARCH_DISCLAIMER)
    return result


def build_feature_table(interproscan_files: list[Path], sample_ids: list[str] | None = None,
                        id_column: str = "gca", schema: list[str] | None = None) -> pd.DataFrame:
    """Convert InterProScan TSV output (one file per genome) to a presence/absence table.

    InterProScan TSV columns (no header): 1 protein accession, 2 MD5, 3 length,
    4 analysis (e.g. 'Pfam'), 5 signature accession (e.g. 'PF13855'), ...
    A domain is 'present' (1) if at least one Pfam hit is reported anywhere in
    the genome's proteome.

    Caveat: the reference data were produced with Prodigal + InterProScan
    5.44-79.0 (via SAPP). Other gene callers or Pfam releases can change which
    domains are detected, so profiles from a different pipeline are not
    guaranteed to be comparable.

    Args:
        interproscan_files: One TSV per genome.
        sample_ids: Identifiers to use (default: file name without extension).
        id_column: Name of the identifier column in the output.
        schema: Optional model feature schema. When given, every schema domain
            gets a column (0 if not detected). Because InterProScan scanned the
            whole proteome, a domain without a hit is genuinely 'not detected',
            so this explicit 0 is justified here (unlike a missing CSV column).

    Returns:
        DataFrame with ``id_column`` and one 0/1 column per Pfam accession.
    """
    if sample_ids is not None and len(sample_ids) != len(interproscan_files):
        raise ValueError("sample_ids must have one entry per InterProScan file")
    rows: dict[str, set[str]] = {}
    for i, path in enumerate(interproscan_files):
        if not path.is_file():
            raise FileNotFoundError(f"InterProScan file not found: {path}")
        # Default id is the file name up to the first dot (genomeA.tsv -> genomeA).
        sid = sample_ids[i] if sample_ids else path.name.split(".")[0]
        if sid in rows:
            raise ValueError(f"duplicate sample identifier '{sid}'")
        table = pd.read_csv(path, sep="\t", header=None, dtype=str, usecols=range(5), comment=None,
                            on_bad_lines="error")
        # Keep only Pfam hits whose accession has the PFxxxxx form (column 4 is the signature accession).
        pfam = table[(table[3].str.lower() == "pfam") & table[4].str.match(r"^PF\d{5}$", na=False)][4]
        rows[sid] = set(pfam)
        log.info("%s: %d distinct Pfam domains", sid, len(rows[sid]))
    # Union of domains seen in any genome plus the model schema, so every table has the same columns.
    accessions = sorted(set().union(*rows.values(), schema or [])) if rows else []
    out = pd.DataFrame([[int(a in doms) for a in accessions] for doms in rows.values()],
                       columns=accessions, index=list(rows.keys()))
    out.index.name = id_column
    return out.reset_index()
