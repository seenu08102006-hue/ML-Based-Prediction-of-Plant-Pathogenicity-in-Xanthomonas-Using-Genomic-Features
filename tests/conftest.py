"""Shared pytest fixtures.

!!! SYNTHETIC TEST FIXTURES !!!
Every dataset created here is randomly generated software-test data. It is not
biological data, is written only to pytest's temporary directories, and never
contributes to any scientific result of this project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_TAG = "SYNTHETIC_TEST_FIXTURE"


def make_synthetic_tables(n_per_class: int = 24, n_noise: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a synthetic domain matrix and metadata table (software tests only).

    Design: 4 'species' groups, each with both classes; 5 informative binary
    features, ``n_noise`` random features, 2 constant features and 1 exact
    duplicate column. Two extra genomes are added that must be excluded:
    one of another genus and one labelled 'unknown'.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    y = np.array([1] * n_per_class + [0] * n_per_class)
    ids = [f"GCA_{900000000 + i:09d}.1" for i in range(n + 2)]
    cols: dict[str, np.ndarray] = {}
    for k in range(5):  # informative: frequent in class 1, rare in class 0
        cols[f"PF9{k:04d}"] = np.where(y == 1, rng.random(n) < 0.85, rng.random(n) < 0.15).astype(int)
    for k in range(n_noise):
        cols[f"PF8{k:04d}"] = (rng.random(n) < rng.uniform(0.2, 0.8)).astype(int)
    cols["PF70000"] = np.ones(n, dtype=int)  # constant present
    cols["PF70001"] = np.zeros(n, dtype=int)  # constant absent
    cols["PF70002"] = cols["PF90000"].copy()  # perfect duplicate of an informative feature
    feats = pd.DataFrame(cols)
    extra = pd.DataFrame([rng.integers(0, 2, feats.shape[1]) for _ in range(2)], columns=feats.columns)
    feats = pd.concat([feats, extra], ignore_index=True)
    feats.insert(0, "gca", ids)

    species = np.array(["alpha", "beta", "gamma", "delta"] * (n // 4))
    meta = pd.DataFrame(
        {
            "strain": [f"SYN{i}" for i in range(n + 2)],
            "alias_of": ["NA"] * (n + 2),
            "genus": ["Xanthomonas"] * n + ["Xylella", "Xanthomonas"],
            "species": list(species) + ["fastidiosa", "alpha"],
            "subspecies": ["unclassified"] * (n + 2),
            "hostGenus": ["Testus"] * (n + 2),
            "hostSpecies": ["NA"] * (n + 2),
            "pathogenicity": ["pathogenic" if v else "non-pathogenic" for v in y] + ["unknown", "unknown"],
            "source": [SYNTHETIC_TAG] * (n + 2),
            "gca": ids,
            "genome_size": rng.integers(4_500_000, 5_200_000, n + 2),
            "gene_count": rng.integers(3800, 4400, n + 2),
        }
    )
    # Shuffle metadata rows so alignment must use identifiers, not row order.
    meta = meta.sample(frac=1, random_state=seed).reset_index(drop=True)
    return feats, meta


def write_project(root: Path, feats: pd.DataFrame, meta: pd.DataFrame, overrides: dict | None = None) -> Path:
    """Write a self-contained mini project (config + raw data) under ``root``."""
    with (REPO_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Synthetic files have no published checksum.
    for entry in cfg["dataset"]["files"].values():
        entry["sha256"] = None
    cfg["dataset"]["files"]["domain_descriptions"]["required"] = False
    cfg["cv"].update(outer_folds=3, outer_repeats=1, inner_folds=3, species_grouped_folds=3, n_jobs=1)
    cfg["models"]["decision_tree"]["param_grid"] = {"clf__max_depth": [2, None]}
    cfg["models"]["l1_logistic_regression"]["param_grid"] = {"clf__C": [0.3, 3.0]}
    cfg["models"]["random_forest"]["param_grid"] = {"clf__max_features": ["sqrt"]}
    cfg["models"]["random_forest"]["n_estimators"] = 25
    cfg["interpretation"].update(permutation_top_k=2, permutation_repeats=2, top_n=5)
    cfg["preprocessing"]["min_prevalence"] = 0.05
    cfg["preprocessing"]["max_prevalence"] = 0.95
    for key, value in (overrides or {}).items():
        cfg[key] = {**cfg.get(key, {}), **value} if isinstance(value, dict) else value
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    config_path = root / "configs" / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    # Write with a UTF-8 BOM, like the published files.
    feats.to_csv(root / "data" / "raw" / "domain_matrix.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(root / "data" / "raw" / "genome_information.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"accession": feats.columns[1:], "description": [f"synthetic domain {c}" for c in feats.columns[1:]]}
                 ).to_csv(root / "data" / "raw" / "domain_descriptions.csv", index=False, encoding="utf-8-sig")
    return config_path


@pytest.fixture
def synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_synthetic_tables()


@pytest.fixture
def synthetic_project(tmp_path: Path, synthetic_tables):
    """A temporary project root with SYNTHETIC raw data and a fast test config."""
    from xanthomonas_ml.config import load_config

    feats, meta = synthetic_tables
    config_path = write_project(tmp_path, feats, meta)
    return load_config(config_path)


@pytest.fixture
def prepared(synthetic_project):
    from xanthomonas_ml.preprocessing import prepare_or_load

    return prepare_or_load(synthetic_project)


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory):
    """Synthetic project with a completed (fast) training run, shared across tests."""
    from xanthomonas_ml.config import load_config
    from xanthomonas_ml.preprocessing import prepare_or_load
    from xanthomonas_ml.train import run_training

    root = tmp_path_factory.mktemp("trained")
    feats, meta = make_synthetic_tables()
    cfg = load_config(write_project(root, feats, meta))
    data = prepare_or_load(cfg)
    run_training(data, cfg)
    return cfg, data


@pytest.fixture
def no_network(monkeypatch):
    """Make any attempt to use the network fail loudly."""
    import socket

    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted during an offline test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
