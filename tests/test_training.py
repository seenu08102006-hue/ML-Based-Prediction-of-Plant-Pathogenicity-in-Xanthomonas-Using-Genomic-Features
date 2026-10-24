"""Tests for model construction, CV design, training and model persistence (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xanthomonas_ml.train import (
    DESIGN_RANDOM,
    DESIGN_SPECIES,
    MODEL_LABELS,
    enabled_models,
    load_model_bundle,
    make_l1_logistic,
    make_pipeline,
    nested_cv,
    outer_splits,
    save_model_bundle,
    selected_features,
)


@pytest.mark.parametrize("name", sorted(MODEL_LABELS))
def test_each_classifier_trains_and_predicts(name, synthetic_project, prepared):
    pipe = make_pipeline(name, synthetic_project).fit(prepared.X, prepared.y)
    proba = pipe.predict_proba(prepared.X)
    assert proba.shape == (len(prepared.y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert len(selected_features(pipe)) > 0


def test_l1_logistic_is_sparse(prepared):
    from xanthomonas_ml.feature_engineering import CorrelationFilter, PrevalenceFilter
    from sklearn.pipeline import Pipeline

    clf = make_l1_logistic(seed=0)
    clf.set_params(C=0.1)
    pipe = Pipeline([("prevalence", PrevalenceFilter(0.05, 0.95)), ("correlation", CorrelationFilter(0.8)),
                     ("clf", clf)]).fit(prepared.X, prepared.y)
    coef = pipe.named_steps["clf"].coef_.ravel()
    assert (coef == 0).sum() > 0, "L1 penalty should zero out some noise features"


def test_unknown_model_rejected(synthetic_project):
    with pytest.raises(ValueError):
        enabled_models(synthetic_project, ["svm"])


def test_stratified_outer_splits(synthetic_project, prepared):
    splits = list(outer_splits(prepared.y, synthetic_project))
    assert len(splits) == 3  # 3 folds x 1 repeat in the test config
    tested = np.concatenate([te for *_, te in splits])
    assert sorted(tested) == list(range(len(prepared.y)))  # each sample tested exactly once per repeat
    for *_, tr, te in splits:
        assert not set(tr) & set(te)
        assert prepared.y.iloc[te].nunique() == 2


def test_species_grouped_splits_never_share_species(synthetic_project, prepared):
    groups = prepared.groups
    for *_, tr, te in outer_splits(prepared.y, synthetic_project, groups):
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])


@pytest.mark.parametrize("design", [DESIGN_RANDOM, DESIGN_SPECIES])
def test_nested_cv_outputs(design, synthetic_project, prepared):
    res = nested_cv("decision_tree", prepared, synthetic_project, design=design)
    preds = pd.DataFrame(res.predictions)
    assert len(res.folds) >= 2
    assert set(preds.columns) >= {"sample_id", "y_true", "y_pred", "y_score"}
    assert preds["sample_id"].is_unique  # one repeat -> each strain predicted once
    assert preds["y_score"].between(0, 1).all()


def test_nested_cv_reproducible(synthetic_project, prepared):
    a = pd.DataFrame(nested_cv("random_forest", prepared, synthetic_project).predictions)
    b = pd.DataFrame(nested_cv("random_forest", prepared, synthetic_project).predictions)
    pd.testing.assert_frame_equal(a, b)


def test_training_run_outputs(trained_project):
    cfg, _ = trained_project
    res = cfg.outputs_dir / "results"
    for name in ("cv_predictions.csv", "cv_folds.csv", "cv_fold_importances.csv", "cv_permutation_importances.csv",
                 "training_run.json", "metrics_summary.csv", "best_model.json"):
        assert (res / name).is_file(), name
    for model in MODEL_LABELS:
        assert (cfg.models_dir / f"{model}.joblib").is_file()
        assert (cfg.models_dir / f"{model}.json").is_file()


def test_model_save_load_roundtrip(tmp_path, trained_project):
    cfg, data = trained_project
    bundle = load_model_bundle(cfg.models_dir / "random_forest.joblib")
    assert bundle["feature_schema"] == list(data.X.columns)
    assert bundle["label_mapping"] == {"pathogenic": 1, "non-pathogenic": 0}
    copy_path = save_model_bundle(tmp_path / "copy.joblib", bundle)
    again = load_model_bundle(copy_path)
    np.testing.assert_allclose(again["pipeline"].predict_proba(data.X), bundle["pipeline"].predict_proba(data.X))


def test_load_invalid_model_file(tmp_path):
    import joblib

    joblib.dump({"not": "a model"}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        load_model_bundle(tmp_path / "bad.joblib")
    with pytest.raises(FileNotFoundError):
        load_model_bundle(tmp_path / "missing.joblib")


def test_holdout_split_is_disjoint(tmp_path, synthetic_tables):
    from conftest import write_project

    from xanthomonas_ml.config import load_config
    from xanthomonas_ml.preprocessing import prepare_or_load
    from xanthomonas_ml.train import run_training

    cfg = load_config(write_project(tmp_path, *synthetic_tables,
                                    overrides={"cv": {"holdout_fraction": 0.25, "species_grouped": False}}))
    run = run_training(prepare_or_load(cfg), cfg, models=["decision_tree"])
    preds = pd.read_csv(cfg.outputs_dir / "results" / "cv_predictions.csv", dtype={"sample_id": str})
    cv_ids = set(preds[preds.design == DESIGN_RANDOM].sample_id)
    hold_ids = set(preds[preds.design == "holdout_test"].sample_id)
    assert hold_ids == set(run["holdout_ids"]) and not cv_ids & hold_ids
    assert "holdout_metrics" in run["final_models"]["decision_tree"]
