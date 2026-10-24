"""Tests for file discovery and raw-table loading (SYNTHETIC fixtures only)."""

from __future__ import annotations

import numpy as np
import pytest

from xanthomonas_ml.data_loader import (
    discover_files,
    find_duplicate_header_names,
    load_feature_matrix,
    load_from_config,
    load_metadata,
    to_numeric_matrix,
)


def test_discover_files_lists_tabular_files(synthetic_project):
    names = [p.name for p in discover_files(synthetic_project.raw_dir)]
    assert names == sorted(["domain_descriptions.csv", "domain_matrix.csv", "genome_information.csv"])


def test_discover_files_missing_directory(tmp_path):
    assert discover_files(tmp_path / "does-not-exist") == []


def test_load_feature_matrix_strips_bom(synthetic_project):
    df = load_feature_matrix(synthetic_project.raw_dir / "domain_matrix.csv")
    assert df.columns[0] == "gca"  # not '﻿gca'
    assert df.attrs["duplicate_columns"] == []


def test_duplicate_header_detected(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("gca,PF1,PF1,PF2\nA,0,1,1\n", encoding="utf-8")
    assert find_duplicate_header_names(path) == ["PF1"]
    assert load_feature_matrix(path).attrs["duplicate_columns"] == ["PF1"]


def test_load_metadata_na_handling(synthetic_project):
    meta = load_metadata(synthetic_project.raw_dir / "genome_information.csv")
    assert meta["alias_of"].isna().all()  # literal "NA" -> missing
    assert (meta["subspecies"] == "unclassified").all()  # ordinary strings preserved


def test_to_numeric_matrix_flags_non_numeric(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("gca,PF1,PF2\nA,0,x\nB,1,0\n", encoding="utf-8")
    numeric = to_numeric_matrix(load_feature_matrix(path), "gca")
    assert np.isnan(numeric.loc["A", "PF2"])
    assert numeric.loc["B", "PF1"] == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_matrix(tmp_path / "absent.csv")


def test_too_few_columns_raises(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("gca\nA\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_feature_matrix(path)


def test_load_from_config_requires_files(synthetic_project):
    from xanthomonas_ml.dataset_download import DatasetError

    (synthetic_project.raw_dir / "domain_matrix.csv").unlink()
    with pytest.raises(DatasetError, match="missing required file"):
        load_from_config(synthetic_project)
