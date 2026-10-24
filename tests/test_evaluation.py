"""Tests for metrics, summaries, figures, interpretation and reporting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from xanthomonas_ml.evaluate import compute_metrics, generate_evaluation_outputs, headline_table


def test_metrics_known_values():
    # tp=2, fn=1, tn=3, fp=1
    m = compute_metrics([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1], [0.9, 0.8, 0.4, 0.1, 0.2, 0.3, 0.6])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 3, 1)
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["recall_sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 4) / 2)
    assert m["roc_auc"] == pytest.approx(11 / 12)
    assert m["undefined"] == {}


def test_undefined_metrics_are_nan_not_zero():
    m = compute_metrics([1, 1, 1], [1, 0, 1], [0.9, 0.2, 0.7])
    assert math.isnan(m["roc_auc"]) and math.isnan(m["specificity"]) and math.isnan(m["balanced_accuracy"])
    assert "roc_auc" in m["undefined"] and "specificity" in m["undefined"]
    m = compute_metrics([0, 1], [0, 0])
    assert math.isnan(m["precision"]) and "precision" in m["undefined"]
    assert math.isnan(m["roc_auc"])  # no scores given


def test_metrics_input_errors():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0])


def test_summaries_and_figures(trained_project):
    cfg, _ = trained_project
    out = generate_evaluation_outputs(cfg)
    assert out["best"]["best_model"] in {"decision_tree", "l1_logistic_regression", "random_forest"}
    for f in out["figures"]:
        assert f.endswith(".png")
    assert (cfg.figures_dir / "eval_model_comparison.png").is_file()
    assert (cfg.figures_dir / "eval_roc_curves_stratified_cv.png").is_file()
    summary = pd.read_csv(cfg.outputs_dir / "results" / "metrics_summary.csv")
    assert set(summary.level) == {"pooled_per_repeat", "per_outer_fold"}
    assert {"n_defined", "n_total", "mean", "std"} <= set(summary.columns)
    table = headline_table(cfg)
    assert len(table) == 6  # 3 models x 2 designs


def test_multi_panel_figures_show_every_model_and_design(trained_project, monkeypatch):
    """Regression test: one panel per model (confusion matrices) and per design (comparison)."""
    import xanthomonas_ml.plotting as plotting
    from xanthomonas_ml.evaluate import load_predictions, plot_confusion_matrices, plot_model_comparison

    cfg, _ = trained_project
    captured = []
    monkeypatch.setattr(plotting, "save_figure", lambda fig, path, dpi=150: captured.append(fig) or path)
    preds = load_predictions(cfg)
    plot_confusion_matrices(preds, "stratified_nested_cv", cfg.figures_dir / "x.png")
    plot_model_comparison(pd.read_csv(cfg.outputs_dir / "results" / "metrics_per_repeat.csv"), cfg.figures_dir / "y.png")
    assert len(captured[0].axes) == 3  # three models
    assert len(captured[1].axes) == 2  # stratified + species-grouped designs


def test_eda_interpret_report(trained_project):
    from xanthomonas_ml.eda import run_eda
    from xanthomonas_ml.interpretability import run_interpretation
    from xanthomonas_ml.reporting import generate_report

    cfg, data = trained_project
    summary = run_eda(data, cfg)
    assert summary["n_strains"] == 48
    for name in ("eda_class_counts.png", "eda_pca.png", "eda_strain_similarity_heatmap.png"):
        assert (cfg.figures_dir / name).is_file()
    top = run_interpretation(cfg, data)
    assert set(top) == {"decision_tree", "l1_logistic_regression", "random_forest"}
    imp = pd.read_csv(cfg.tables_dir / "feature_importance_random_forest.csv")
    assert {"feature", "selection_frequency", "mean_importance_cv", "final_model_importance"} <= set(imp.columns)
    assert imp["selection_frequency"].between(0, 1).all()
    report = generate_report(cfg).read_text(encoding="utf-8")
    for heading in ("## 1. Introduction", "## 10. Model results", "## 16. References"):
        assert heading in report
    assert "Results unavailable" not in report.split("## 10. Model results")[1].split("## 11.")[0]


def test_report_without_results_says_unavailable(synthetic_project):
    from xanthomonas_ml.reporting import generate_report

    text = generate_report(synthetic_project).read_text(encoding="utf-8")
    assert "Results unavailable" in text
    assert "Best model" not in text


def test_per_strain_probabilities():
    from xanthomonas_ml.evaluate import per_strain_probabilities

    preds = pd.DataFrame({"model": "m", "design": "stratified_nested_cv", "repeat": [0, 1, 0, 1],
                          "fold": 0, "sample_id": ["a", "a", "b", "b"], "y_true": [1, 1, 0, 0],
                          "y_pred": [1, 0, 0, 0], "y_score": [0.8, 0.4, 0.1, 0.3]})
    t = per_strain_probabilities(preds).set_index("sample_id")
    assert t.loc["a", "fraction_misclassified"] == 0.5 and t.loc["b", "fraction_misclassified"] == 0
    assert np.isclose(t.loc["a", "mean_prob_pathogenic"], 0.6)
