"""Data-quality validation of the raw published tables.

Every check produces an :class:`Issue` with a severity:

* ``ERROR``   - the scientific analysis must not proceed (e.g. duplicate sample
  identifiers, non-binary feature values, no overlap between tables).
* ``WARNING`` - the analysis can proceed, but samples are excluded or a
  scientific caveat applies (e.g. 'unknown' phenotype labels).
* ``INFO``    - descriptive facts recorded for provenance.

Labels and features are aligned *only* through the explicit sample identifier
(GenBank assembly accession, column ``gca``), never by row order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xanthomonas_ml.config import Config
from xanthomonas_ml.data_loader import RawDataset, to_numeric_matrix
from xanthomonas_ml.utils import ensure_dir, utc_now_iso, write_json

ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"


@dataclass
class Issue:
    """One validation finding."""

    level: str
    code: str
    message: str
    details: Any = None


@dataclass
class ValidationReport:
    """Collection of validation findings plus summary statistics."""

    created_utc: str = field(default_factory=utc_now_iso)
    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, details: Any = None) -> None:
        self.issues.append(Issue(level, code, message, details))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_utc": self.created_utc,
            "passed": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "summary": self.summary,
            "issues": [asdict(i) for i in self.issues],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Data validation report",
            "",
            f"Generated: {self.created_utc} (UTC)",
            "",
            f"**Result: {'PASSED' if self.ok else 'FAILED'}** "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)",
            "",
            "## Summary",
            "",
        ]
        for key, value in self.summary.items():
            lines.append(f"- **{key}**: {value}")
        lines += ["", "## Findings", "", "| Level | Check | Message |", "|---|---|---|"]
        for issue in self.issues:
            message = issue.message.replace("|", "\\|")
            lines.append(f"| {issue.level} | `{issue.code}` | {message} |")
        detailed = [i for i in self.issues if i.details]
        if detailed:
            lines += ["", "## Details", ""]
            for issue in detailed:
                lines += [f"### {issue.code}", "", "```", _format_details(issue.details), "```", ""]
        return "\n".join(lines) + "\n"

    def save(self, directory: Path, stem: str = "data_validation") -> tuple[Path, Path]:
        """Write JSON and Markdown versions of the report into ``directory``."""
        ensure_dir(directory)
        json_path = write_json(directory / f"{stem}.json", self.to_dict())
        md_path = directory / f"{stem}.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path


def _format_details(details: Any, limit: int = 60) -> str:
    if isinstance(details, (list, tuple)):
        shown = [str(d) for d in details[:limit]]
        extra = f"\n... ({len(details) - limit} more)" if len(details) > limit else ""
        return "\n".join(shown) + extra
    if isinstance(details, dict):
        return "\n".join(f"{k}: {v}" for k, v in list(details.items())[:limit])
    return str(details)


def check_identifiers(ids: pd.Series, table: str, report: ValidationReport) -> None:
    """Report empty and duplicated sample identifiers in one table."""
    stripped = ids.astype(str).str.strip()
    # Blank cells and the literal 'NA' count as missing identifiers, matching the loader's rule.
    empty =int((ids.isna() | (stripped == "") | (stripped.str.upper() == "NA")).sum())
    if empty:
        report.add(ERROR, f"{table}_empty_ids", f"{empty} rows of the {table} table have no sample identifier")
    dupes = sorted(stripped[stripped.duplicated(keep=False)].unique())
    if dupes:
        report.add(
            ERROR,
            f"{table}_duplicate_ids",
            f"{len(dupes)} sample identifiers occur more than once in the {table} table",
            dupes,
        )


def check_feature_values(matrix_text: pd.DataFrame, id_column: str, report: ValidationReport) -> pd.DataFrame:
    """Check that every feature cell is a valid binary presence/absence value.

    Returns:
        The numeric matrix (indexed by sample id) for further checks.
    """
    numeric = to_numeric_matrix(matrix_text, id_column)
    raw = matrix_text.set_index(id_column)
    blank = raw.apply(lambda c: c.str.strip().isin(["", "NA", "NaN", "nan"]))
    n_missing = int(blank.values.sum())
    if n_missing:
        report.add(ERROR, "missing_feature_values", f"{n_missing} feature cells are empty/NA",
                   _cells(blank, limit=50))
    # NaN after conversion that was not an explicit blank means the cell held text that is not a number.
    non_numeric = numeric.isna() & ~blank
    if int(non_numeric.values.sum()):
        report.add(ERROR, "non_numeric_feature_values",
                   f"{int(non_numeric.values.sum())} feature cells are not numeric", _cells(non_numeric, 50))
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if np.isinf(values).any():
        report.add(ERROR, "infinite_feature_values", "feature matrix contains infinite values")
    invalid = sorted(set(np.unique(finite)) - {0.0, 1.0})
    if invalid:
        report.add(ERROR, "non_binary_feature_values",
                   "feature values must be 0 (absent) or 1 (present) for the published encoding",
                   [float(v) for v in invalid[:20]])
    else:
        report.add(INFO, "binary_encoding", "all non-missing feature values are 0/1 presence/absence")
    return numeric


def _cells(mask: pd.DataFrame, limit: int) -> list[str]:
    rows, cols = np.nonzero(mask.to_numpy())
    return [f"{mask.index[r]} / {mask.columns[c]}" for r, c in list(zip(rows, cols))[:limit]]


def validate_raw_dataset(raw: RawDataset, cfg: Config, file_statuses: list[Any] | None = None) -> ValidationReport:
    """Run all data-quality checks on the raw published tables.

    Args:
        raw: Tables loaded by :mod:`xanthomonas_ml.data_loader`.
        cfg: Configuration (identifier/label column names and label rules).
        file_statuses: Optional checksum statuses to record in the report.

    Returns:
        A :class:`ValidationReport`; check ``report.ok`` before analysing.
    """
    report = ValidationReport()
    id_col = cfg["dataset"]["sample_id_column"]
    label_col = cfg["dataset"]["label_column"]
    pos, neg = cfg["labels"]["positive"], cfg["labels"]["negative"]
    genus = cfg["labels"].get("required_genus")

    if file_statuses:
        for st in file_statuses:
            level = INFO if st.checksum_ok else (WARNING if st.checksum_ok is None else ERROR)
            report.add(level, f"file_{st.key}",
                       f"{st.filename}: exists={st.exists}, sha256_ok={st.checksum_ok}, size={st.size_bytes}")

    feats, meta = raw.features, raw.metadata
    report.summary["feature_table_shape (rows x columns incl. id)"] = list(feats.shape)
    report.summary["metadata_table_shape"] = list(meta.shape)
    report.summary["metadata_columns"] = list(meta.columns)

    # --- schema ---------------------------------------------------------------
    for table, df, cols in (("features", feats, [id_col]), ("metadata", meta, [id_col, label_col])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            report.add(ERROR, f"{table}_missing_columns", f"{table} table lacks required columns {missing}")
    # Stop here if the expected columns are missing; the checks below all depend on them.
    if report.errors:
        return report

    dup_cols = feats.attrs.get("duplicate_columns", [])
    if dup_cols:
        report.add(ERROR, "duplicate_feature_names", f"{len(dup_cols)} feature names are repeated", dup_cols)

    # --- identifiers ------------------------------------------------------------
    check_identifiers(feats[id_col], "features", report)
    check_identifiers(meta[id_col], "metadata", report)

    # --- feature values ----------------------------------------------------------
    numeric = check_feature_values(feats, id_col, report)
    report.summary["n_feature_rows"] = int(numeric.shape[0])
    report.summary["n_features"] = int(numeric.shape[1])

    # --- alignment by identifier (never by row order) --------------------------
    f_ids, m_ids = set(numeric.index), set(meta[id_col].astype(str).str.strip())
    only_f, only_m, common = sorted(f_ids - m_ids), sorted(m_ids - f_ids), f_ids & m_ids
    report.summary["samples_in_both_tables"] = len(common)
    if not common:
        report.add(ERROR, "no_id_overlap", "no sample identifier is shared by the feature and metadata tables")
        return report
    if only_f:
        report.add(WARNING, "features_without_metadata",
                   f"{len(only_f)} feature rows have no metadata/label and will be excluded", only_f)
    if only_m:
        report.add(WARNING, "metadata_without_features",
                   f"{len(only_m)} metadata rows have no feature profile and will be excluded", only_m)
    if not only_f and not only_m:
        report.add(INFO, "id_alignment", f"all {len(common)} identifiers match one-to-one between tables")
    if list(numeric.index) != list(meta[id_col].astype(str).str.strip()):
        report.add(INFO, "row_order_differs",
                   "row order differs between tables; alignment is done by identifier, so this is harmless")

    # --- labels --------------------------------------------------------------------
    # Index labels by sample id so every lookup below is by identifier, never by position.
    labels = meta.set_index(meta[id_col].astype(str).str.strip())[label_col]
    counts = labels.fillna("<missing>").value_counts().to_dict()
    report.summary["source_label_counts"] = counts
    n_missing = int(labels.isna().sum())
    if n_missing:
        report.add(WARNING, "missing_labels", f"{n_missing} samples have no phenotype label and will be excluded",
                   sorted(labels.index[labels.isna()]))
    other = sorted(set(labels.dropna()) - {pos, neg})
    if other:
        affected = labels[labels.isin(other)]
        report.add(WARNING, "excluded_label_categories",
                   f"label values {other} are neither '{pos}' nor '{neg}'; these {len(affected)} samples are "
                   "excluded rather than re-mapped to a class",
                   {i: v for i, v in affected.items()})

    if genus and "genus" in meta.columns:
        g = meta.set_index(meta[id_col].astype(str).str.strip())["genus"]
        off = g[g != genus]
        if len(off):
            report.add(WARNING, "non_target_genus",
                       f"{len(off)} genomes are not genus {genus} and will be excluded",
                       {i: f"{v} {meta.set_index(id_col).loc[i].get('species', '')} (label={labels.get(i)})"
                        for i, v in off.items()})

    # --- analysis set preview -------------------------------------------------------
    usable = labels[labels.isin([pos, neg])]
    usable = usable[usable.index.isin(common)]
    if genus and "genus" in meta.columns:
        g = meta.set_index(meta[id_col].astype(str).str.strip())["genus"]
        usable = usable[g.reindex(usable.index) == genus]
    class_counts = usable.value_counts().to_dict()
    report.summary["analysis_set_class_counts"] = class_counts
    n_pos, n_neg = class_counts.get(pos, 0), class_counts.get(neg, 0)
    min_needed = int(cfg["cv"]["outer_folds"])
    if min(n_pos, n_neg) < min_needed:
        report.add(ERROR, "too_few_samples_per_class",
                   f"each class needs at least {min_needed} samples for {min_needed}-fold stratified CV "
                   f"(have {pos}={n_pos}, {neg}={n_neg})")
    elif n_pos + n_neg:
        minority = min(n_pos, n_neg) / (n_pos + n_neg)
        level = WARNING if minority < 0.2 else INFO
        report.add(level, "class_balance", f"{pos}={n_pos}, {neg}={n_neg} (minority fraction {minority:.2f})")

    # --- feature redundancy (descriptive; handled by in-fold filters) --------------
    analysis = numeric.loc[numeric.index.isin(usable.index)]
    if len(analysis) and not report.errors:
        prevalence = analysis.mean()
        n_const = int(((prevalence == 0) | (prevalence == 1)).sum())
        report.add(INFO, "constant_features",
                   f"{n_const} of {analysis.shape[1]} features are constant across the analysis set "
                   "(no information; removed during preparation)")
        variable = analysis.loc[:, (prevalence > 0) & (prevalence < 1)]
        # Transpose so duplicated() compares whole feature columns rather than rows.
        n_dup_feat = int(variable.T.duplicated().sum())
        report.add(INFO, "identical_feature_columns",
                   f"{n_dup_feat} of {variable.shape[1]} variable feature columns are exact copies of another column "
                   "(perfect co-occurrence, e.g. domains of one protein or operon); kept, and handled by the in-fold "
                   "correlation filter")
        dup_profiles = analysis[analysis.duplicated(keep=False)]
        if len(dup_profiles):
            # Group identical rows via a byte fingerprint of the profile.
            keys = pd.Series([row.tobytes() for row in dup_profiles.to_numpy()], index=dup_profiles.index)
            listing = [sorted(map(str, idx)) for idx in keys.groupby(keys).groups.values()]
            report.add(WARNING, "identical_sample_profiles",
                       f"{len(dup_profiles)} samples share an identical domain profile with another sample; "
                       "they are distinct strains and are kept, but CV estimates may be slightly optimistic "
                       "if they fall into different folds", listing)
    return report
