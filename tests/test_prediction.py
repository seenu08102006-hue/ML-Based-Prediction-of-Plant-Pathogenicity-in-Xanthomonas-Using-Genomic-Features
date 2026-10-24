"""Tests for the prediction interface (SYNTHETIC data and models)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xanthomonas_ml.predict import (
    PredictionInputError,
    build_feature_table,
    predict_dataframe,
    predict_file,
    validate_input,
)
from xanthomonas_ml.train import load_model_bundle


@pytest.fixture
def bundle(trained_project):
    cfg, _ = trained_project
    return load_model_bundle(cfg.models_dir / "random_forest.joblib")


@pytest.fixture
def input_table(trained_project):
    _, data = trained_project
    return data.X.iloc[:5].reset_index().astype(str)


def test_valid_input_aligned_to_schema(bundle, input_table):
    X, report = validate_input(input_table, bundle)
    assert list(X.columns) == bundle["feature_schema"]
    assert report["n_samples"] == 5 and report["n_extra_columns_ignored"] == 0


def test_missing_required_feature_is_error(bundle, input_table):
    required = bundle["selected_features"][0]
    with pytest.raises(PredictionInputError, match="required by the model are missing"):
        validate_input(input_table.drop(columns=[required]), bundle)


def test_fill_missing_as_absent(bundle, input_table):
    required = bundle["selected_features"][0]
    X, report = validate_input(input_table.drop(columns=[required]), bundle, fill_missing_as_absent=True)
    assert (X[required] == 0).all()
    assert report["missing_required_filled_as_absent"] == [required]


def test_extra_columns_ignored(bundle, input_table):
    X, report = validate_input(input_table.assign(PF99999="1"), bundle)
    assert "PF99999" not in X.columns and report["n_extra_columns_ignored"] == 1


@pytest.mark.parametrize("bad", ["2", "", "maybe", "-1"])
def test_invalid_values_rejected(bundle, input_table, bad):
    table = input_table.copy()
    table.loc[0, bundle["selected_features"][0]] = bad
    with pytest.raises(PredictionInputError, match="not valid presence/absence"):
        validate_input(table, bundle)


def test_textual_booleans_accepted(bundle, input_table):
    table = input_table.copy()
    col = bundle["selected_features"][0]
    table[col] = ["true", "False", "1.0", "0", "present"]
    X, _ = validate_input(table, bundle)
    assert X[col].tolist() == [1, 0, 1, 0, 1]


def test_identifier_problems(bundle, input_table):
    with pytest.raises(PredictionInputError, match="identifier column"):
        validate_input(input_table.drop(columns=["gca"]), bundle)
    dup = input_table.copy()
    dup.loc[1, "gca"] = dup.loc[0, "gca"]
    with pytest.raises(PredictionInputError, match="duplicate"):
        validate_input(dup, bundle)
    with pytest.raises(PredictionInputError, match="no samples"):
        validate_input(input_table.iloc[0:0], bundle)


def test_prediction_output_format(bundle, input_table):
    X, report = validate_input(input_table, bundle)
    out = predict_dataframe(bundle, X, report)
    assert list(out.columns[:5]) == ["sample_id", "predicted_class", "probability_pathogenic",
                                     "probability_non_pathogenic", "model"]
    assert set(out.predicted_class) <= {"pathogenic", "non-pathogenic"}
    np.testing.assert_allclose(out.probability_pathogenic + out.probability_non_pathogenic, 1.0)
    assert (out["note"].str.contains("not a diagnosis")).all()


def test_predict_file_writes_csv(tmp_path, trained_project, input_table):
    cfg, _ = trained_project
    src = tmp_path / "new.csv"
    input_table.to_csv(src, index=False)
    out = tmp_path / "sub" / "pred.csv"  # parent directory is created on demand
    result = predict_file(cfg.models_dir / "decision_tree.joblib", src, out)
    assert out.is_file() and len(pd.read_csv(out)) == len(result) == 5


def test_predict_file_missing(tmp_path, trained_project):
    cfg, _ = trained_project
    with pytest.raises(FileNotFoundError):
        predict_file(cfg.models_dir / "decision_tree.joblib", tmp_path / "absent.csv")


def test_build_feature_table_from_interproscan(tmp_path):
    # Minimal synthetic InterProScan TSV rows (first 5 of the standard columns + extras).
    rows = [
        "prot1\tmd5\t300\tPfam\tPF13855\tLRR\t10\t50\t1e-5\tT\t01-01-2024",
        "prot2\tmd5\t200\tPfam\tPF13855\tLRR\t5\t40\t1e-5\tT\t01-01-2024",
        "prot3\tmd5\t250\tSMART\tSM00369\tLRR\t5\t40\t1e-5\tT\t01-01-2024",
        "prot4\tmd5\t150\tPfam\tPF12840\tHTH\t1\t60\t1e-8\tT\t01-01-2024",
    ]
    f1 = tmp_path / "genomeA.tsv"
    f1.write_text("\n".join(rows) + "\n", encoding="utf-8")
    f2 = tmp_path / "genomeB.tsv"
    f2.write_text(rows[3] + "\n", encoding="utf-8")
    table = build_feature_table([f1, f2], schema=["PF00001"]).set_index("gca")
    assert list(table.index) == ["genomeA", "genomeB"]
    assert table.loc["genomeA", "PF13855"] == 1 and table.loc["genomeB", "PF13855"] == 0
    assert "SM00369" not in table.columns  # only Pfam signatures are used
    assert (table["PF00001"] == 0).all()  # schema domain not detected
    with pytest.raises(ValueError):
        build_feature_table([f1, f1])  # duplicate sample id
