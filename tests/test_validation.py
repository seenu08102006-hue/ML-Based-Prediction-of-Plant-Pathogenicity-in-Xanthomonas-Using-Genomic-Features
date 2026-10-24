"""Tests for data validation (SYNTHETIC fixtures only)."""

from __future__ import annotations

from xanthomonas_ml.data_loader import RawDataset
from xanthomonas_ml.validation import ERROR, WARNING, validate_raw_dataset


def _raw(feats, meta):
    # Mirror the loader: every cell is read as text.
    feats = feats.astype(str)
    feats.attrs["duplicate_columns"] = []
    return RawDataset(features=feats, metadata=meta.astype(str).replace({"NA": None}), descriptions=None)


def _codes(report, level):
    return {i.code for i in report.issues if i.level == level}


def test_valid_synthetic_dataset_passes(synthetic_project, synthetic_tables):
    report = validate_raw_dataset(_raw(*synthetic_tables), synthetic_project)
    assert report.ok, [i.message for i in report.errors]
    assert report.summary["analysis_set_class_counts"] == {"pathogenic": 24, "non-pathogenic": 24}
    # 'unknown' labels and the non-Xanthomonas genome are reported, not silently mapped
    assert {"excluded_label_categories", "non_target_genus"} <= _codes(report, WARNING)


def test_shuffled_rows_are_harmless(synthetic_project, synthetic_tables):
    feats, meta = synthetic_tables
    report = validate_raw_dataset(_raw(feats.iloc[::-1], meta), synthetic_project)
    assert report.ok
    assert "row_order_differs" in {i.code for i in report.issues}


def test_duplicate_ids_are_errors(synthetic_project, synthetic_tables):
    feats, meta = synthetic_tables
    feats = feats.copy()
    feats.loc[1, "gca"] = feats.loc[0, "gca"]
    report = validate_raw_dataset(_raw(feats, meta), synthetic_project)
    assert "features_duplicate_ids" in _codes(report, ERROR)


def test_missing_labels_are_warnings(synthetic_project, synthetic_tables):
    feats, meta = synthetic_tables
    meta = meta.copy()
    meta.loc[0, "pathogenicity"] = "NA"
    report = validate_raw_dataset(_raw(feats, meta), synthetic_project)
    assert "missing_labels" in _codes(report, WARNING)


def test_non_binary_and_missing_values_are_errors(synthetic_project, synthetic_tables):
    feats, meta = synthetic_tables
    feats = feats.astype(str)
    feats.loc[0, "PF80000"] = "2"
    feats.loc[1, "PF80001"] = ""
    feats.loc[2, "PF80002"] = "abc"
    errors = _codes(validate_raw_dataset(_raw(feats, meta), synthetic_project), ERROR)
    assert {"non_binary_feature_values", "missing_feature_values", "non_numeric_feature_values"} <= errors


def test_no_identifier_overlap_is_error(synthetic_project, synthetic_tables):
    feats, meta = synthetic_tables
    meta = meta.copy()
    meta["gca"] = "OTHER_" + meta["gca"]
    assert "no_id_overlap" in _codes(validate_raw_dataset(_raw(feats, meta), synthetic_project), ERROR)


def test_missing_label_column_is_error(synthetic_project, synthetic_tables):
    feats, meta = synthetic_tables
    report = validate_raw_dataset(_raw(feats, meta.drop(columns=["pathogenicity"])), synthetic_project)
    assert "metadata_missing_columns" in _codes(report, ERROR)


def test_identical_profiles_warned(synthetic_project, synthetic_tables):
    feats, meta = synthetic_tables
    feats = feats.copy()
    feats.iloc[1, 1:] = feats.iloc[0, 1:].to_numpy()
    report = validate_raw_dataset(_raw(feats, meta), synthetic_project)
    assert "identical_sample_profiles" in _codes(report, WARNING)


def test_report_files_written(tmp_path, synthetic_project, synthetic_tables):
    report = validate_raw_dataset(_raw(*synthetic_tables), synthetic_project)
    json_path, md_path = report.save(tmp_path / "reports")
    assert json_path.is_file() and md_path.is_file()
    assert "Data validation report" in md_path.read_text(encoding="utf-8")
