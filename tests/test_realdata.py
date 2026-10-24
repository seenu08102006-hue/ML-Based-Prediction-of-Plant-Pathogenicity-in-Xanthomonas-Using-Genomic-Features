"""Integration checks on the AUTHENTIC published dataset.

These tests run only when the verified files are present in data/raw (run
``python -m xanthomonas_ml download-data`` first); otherwise they are skipped,
which is reported by pytest. They never download anything themselves.
"""

from __future__ import annotations

import pytest

from xanthomonas_ml.config import load_config
from xanthomonas_ml.dataset_download import check_local_files

cfg = load_config()
_present = all(s.exists and s.checksum_ok for s in check_local_files(cfg) if s.required)
pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not _present, reason="authentic dataset not downloaded (run download-data)"),
]


def test_real_dataset_validates():
    from xanthomonas_ml.data_loader import load_from_config
    from xanthomonas_ml.validation import validate_raw_dataset

    report = validate_raw_dataset(load_from_config(cfg), cfg)
    assert report.ok
    # Counts reported by te Molder et al. (2021): 65 pathogenic, 53 non-pathogenic genomes.
    assert report.summary["analysis_set_class_counts"] == {"pathogenic": 65, "non-pathogenic": 53}


def test_real_analysis_set_matches_publication():
    from xanthomonas_ml.data_loader import load_from_config
    from xanthomonas_ml.feature_engineering import PrevalenceFilter
    from xanthomonas_ml.preprocessing import build_analysis_dataset

    data = build_analysis_dataset(load_from_config(cfg), cfg)
    assert data.X.shape[0] == 118
    assert bool((data.X.sum() > 0).all())  # constant-absent domains were removed
    # 3,609 domains occur in at least one Xanthomonas genome (paper); 1,692 pass the 2.5%/97.5% filter (paper).
    n_present = data.X.shape[1] + int((data.removed_features["reason"] == "present in all analysed strains").sum())
    assert n_present == 3609
    assert int(PrevalenceFilter(0.025, 0.975).fit(data.X).support_.sum()) == 1692
    assert list(data.removed_samples["sample_id"]) == ["GCA_000006725.1"]  # Xylella fastidiosa 9A5C
