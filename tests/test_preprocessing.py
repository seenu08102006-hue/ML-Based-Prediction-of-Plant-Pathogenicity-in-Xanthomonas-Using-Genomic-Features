"""Tests for analysis-set construction and the in-fold feature filters (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xanthomonas_ml.feature_engineering import (
    CorrelationFilter,
    PrevalenceFilter,
    benjamini_hochberg,
    feature_prevalence_by_class,
)
from xanthomonas_ml.preprocessing import (
    class_balance,
    encode_labels,
    load_prepared,
    prepare_or_load,
    processed_is_current,
)


def test_alignment_uses_identifiers(prepared, synthetic_tables):
    feats, meta = synthetic_tables
    # Metadata rows were shuffled in the fixture; every label must still match its id.
    truth = meta.set_index("gca")["pathogenicity"].map({"pathogenic": 1, "non-pathogenic": 0})
    assert (prepared.y == truth.loc[prepared.y.index]).all()
    raw_row = feats.set_index("gca").loc[prepared.X.index[0], prepared.X.columns]
    assert (prepared.X.iloc[0].to_numpy() == raw_row.to_numpy()).all()


def test_exclusions_are_recorded(prepared):
    reasons = prepared.removed_samples["reason"].tolist()
    assert len(prepared.y) == 48
    assert any("Xylella" in r for r in reasons)
    assert any("'unknown' is not a binary class" in r for r in reasons)


def test_constant_features_removed_and_recorded(prepared):
    assert "PF70000" not in prepared.X.columns and "PF70001" not in prepared.X.columns
    assert set(prepared.removed_features["feature"]) == {"PF70000", "PF70001"}
    # duplicates are biologically meaningful co-occurrence and are kept at this stage
    assert "PF70002" in prepared.X.columns


def test_encode_labels_rejects_unknown():
    encoded, rejected = encode_labels(pd.Series(["pathogenic", "unknown", "non-pathogenic", None]),
                                      {"pathogenic": 1, "non-pathogenic": 0})
    assert encoded.tolist() == [1, 0]
    assert len(rejected) == 2


def test_class_balance():
    cb = class_balance(pd.Series([1, 1, 1, 0]))
    assert cb["counts"] == {0: 1, 1: 3} and cb["minority_fraction"] == 0.25


def test_save_load_roundtrip_and_cache(synthetic_project, prepared):
    loaded = load_prepared(synthetic_project.processed_dir, synthetic_project)
    pd.testing.assert_frame_equal(loaded.X, prepared.X, check_dtype=False)
    assert loaded.y.equals(prepared.y)
    from xanthomonas_ml.dataset_download import require_valid_files

    paths = require_valid_files(synthetic_project)
    assert processed_is_current(synthetic_project.processed_dir, paths, synthetic_project)
    # Changing a source file invalidates the cache.
    with paths["genome_information"].open("a", encoding="utf-8") as fh:
        fh.write("\n")
    assert not processed_is_current(synthetic_project.processed_dir, paths, synthetic_project)
    rebuilt = prepare_or_load(synthetic_project)
    assert rebuilt.X.shape == prepared.X.shape


def test_load_prepared_missing_files(tmp_path, synthetic_project):
    with pytest.raises(FileNotFoundError):
        load_prepared(tmp_path, synthetic_project)


def test_prevalence_filter():
    X = pd.DataFrame({"rare": [1] + [0] * 99, "common": [1] * 50 + [0] * 50, "core": [1] * 99 + [0]})
    f = PrevalenceFilter(0.025, 0.975).fit(X)
    assert list(f.get_feature_names_out()) == ["common"]
    assert list(f.transform(X).columns) == ["common"]


def test_prevalence_filter_rejects_empty_result():
    with pytest.raises(ValueError):
        PrevalenceFilter().fit(pd.DataFrame({"a": [1, 1, 1]}))


def test_correlation_filter_keeps_first_and_records_group():
    a = np.array([1, 0, 1, 0, 1, 0, 1, 1])
    X = pd.DataFrame({"A": a, "B": a, "C": 1 - a, "D": [0, 0, 1, 1, 0, 1, 1, 0]})
    f = CorrelationFilter(0.8).fit(X)
    assert list(f.get_feature_names_out()) == ["A", "D"]
    assert f.correlated_groups() == {"A": ["B", "C"]}
    assert CorrelationFilter(None).fit(X).support_.all()


def test_filters_reject_different_columns():
    X = pd.DataFrame({"A": [0, 1, 0, 1], "B": [1, 1, 0, 0]})
    f = PrevalenceFilter(0, 1).fit(X)
    with pytest.raises(ValueError):
        f.transform(X[["B", "A"]])


def test_filters_fit_on_training_data_only(prepared):
    """The fitted filter state must depend only on the rows it was fitted on."""
    train = prepared.X.iloc[:30]
    full = PrevalenceFilter(0.05, 0.95).fit(prepared.X)
    part = PrevalenceFilter(0.05, 0.95).fit(train)
    np.testing.assert_allclose(part.prevalence_, train.mean().to_numpy())
    assert not np.allclose(part.prevalence_, full.prevalence_)


def test_prevalence_by_class_and_bh(prepared):
    table = feature_prevalence_by_class(prepared.X, prepared.y)
    informative = {f"PF9{k:04d}" for k in range(5)} | {"PF70002"}  # PF70002 duplicates PF90000
    assert len(set(table.index[:6]) & informative) >= 4
    assert ((table["fisher_q_bh"] >= table["fisher_p"]) | np.isclose(table["fisher_q_bh"], table["fisher_p"])).all()
    np.testing.assert_allclose(benjamini_hochberg(np.array([0.01, 0.04, 0.03])), [0.03, 0.04, 0.04])
