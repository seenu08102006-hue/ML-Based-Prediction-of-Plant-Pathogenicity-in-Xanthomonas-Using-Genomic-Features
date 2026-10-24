"""Model definitions, nested cross-validation and final-model fitting.

Validation design (chosen because only ~118 strains are available):

* **Primary estimate - nested repeated stratified CV.** The outer loop
  (``outer_repeats`` x ``outer_folds``) estimates performance; inside every
  outer training set an inner stratified K-fold grid search chooses the
  hyper-parameters. The feature filters are part of the pipeline, so they are
  re-fitted on each (inner and outer) training set: no information from a
  validation fold reaches the model. This mirrors the 20x5 / 4x10 nested design
  of te Molder et al. (2021) at a smaller default size.
* **Stricter estimate - species-grouped nested CV.** Strains of the same species
  are closely related and can share domain content. Holding whole species out
  together (StratifiedGroupKFold) estimates how well the model transfers to
  species not seen in training. Because X. arboricola alone holds ~46% of the
  strains, folds are unequal; this estimate is noisy and reported separately.
* **Final model.** For prediction, hyper-parameters are tuned by stratified CV
  on all strains and the pipeline is refitted on all strains. Its training-set
  (resubstitution) accuracy is recorded only as a diagnostic, never as evidence
  of generalisation.
"""

from __future__ import annotations

import platform
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from xanthomonas_ml import __version__
from xanthomonas_ml.config import Config
from xanthomonas_ml.feature_engineering import CorrelationFilter, PrevalenceFilter
from xanthomonas_ml.interpretability import permutation_importance_subset
from xanthomonas_ml.preprocessing import PreparedData
from xanthomonas_ml.utils import ensure_dir, get_logger, utc_now_iso, write_json

log = get_logger("train")

MODEL_LABELS = {
    "decision_tree": "Decision tree (CART)",
    "l1_logistic_regression": "L1-regularised logistic regression",
    "random_forest": "Random forest",
}
DESIGN_RANDOM = "stratified_nested_cv"
DESIGN_SPECIES = "species_grouped_nested_cv"
DESIGN_HOLDOUT = "holdout_test"


def _sklearn_at_least(major: int, minor: int) -> bool:
    parts = sklearn.__version__.split(".")
    return (int(parts[0]), int(parts[1])) >= (major, minor)


def make_l1_logistic(seed: int) -> LogisticRegression:
    """L1-penalised logistic regression, portable across scikit-learn versions.

    scikit-learn >= 1.8 deprecates ``penalty`` in favour of ``l1_ratio=1``;
    older versions need ``penalty='l1'``. The liblinear solver supports pure L1.
    Note: this is a sparse linear classifier comparable to, but not identical
    with, the glmnet Lasso used in the reference study (different solver,
    regularisation path and scaling of the penalty).
    """
    common = dict(solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed)
    if _sklearn_at_least(1, 8):
        return LogisticRegression(l1_ratio=1.0, **common)
    return LogisticRegression(penalty="l1", **common)


def make_classifier(name: str, cfg: Config) -> Any:
    """Instantiate an unfitted classifier by configuration name."""
    seed = cfg.seed
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    if name == "l1_logistic_regression":
        return make_l1_logistic(seed)
    if name == "random_forest":
        n_trees = int(cfg["models"]["random_forest"].get("n_estimators", 500))
        # n_jobs=1: parallelism is applied at the grid-search level instead.
        return RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODEL_LABELS)}")


def make_pipeline(name: str, cfg: Config) -> Pipeline:
    """Feature filters + classifier. Filters are re-fitted on every training set."""
    # Step order matters: the filters run first, so the classifier only sees the surviving features.
    pp = cfg["preprocessing"]
    return Pipeline(
        [
            ("prevalence", PrevalenceFilter(pp["min_prevalence"], pp["max_prevalence"])),
            ("correlation", CorrelationFilter(pp.get("correlation_threshold"))),
            ("clf", make_classifier(name, cfg)),
        ]
    )


def enabled_models(cfg: Config, only: list[str] | None = None) -> list[str]:
    """Names of models enabled in the configuration (optionally restricted)."""
    names = [n for n, spec in cfg["models"].items() if spec.get("enabled", True)]
    unknown = [n for n in names if n not in MODEL_LABELS]
    if unknown:
        raise ValueError(f"Unknown models in configuration: {unknown}")
    if only:
        bad = [n for n in only if n not in MODEL_LABELS]
        if bad:
            raise ValueError(f"Unknown model(s) {bad}. Choose from {sorted(MODEL_LABELS)}")
        names = [n for n in names if n in only]
    if not names:
        raise ValueError("No models selected")
    return names


def param_grid(name: str, cfg: Config) -> dict[str, list[Any]]:
    """Hyper-parameter grid for ``name`` (YAML ``null`` becomes ``None``)."""
    grid = cfg["models"][name].get("param_grid", {})
    return {k: list(v) for k, v in grid.items()}


def fitted_importances(pipeline: Pipeline) -> pd.Series:
    """Model-specific importance per *input* feature of a fitted pipeline.

    * Decision tree / random forest: impurity-based ``feature_importances_``.
    * L1 logistic regression: signed coefficient (positive = pushes towards
      'pathogenic'); exactly zero means the feature was not selected.
    Features removed by the filters are absent from the returned series.
    """
    names = selected_features(pipeline)
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values = np.ravel(clf.coef_)
    else:
        values = clf.feature_importances_
    return pd.Series(values, index=names, dtype=float)


def selected_features(pipeline: Pipeline) -> list[str]:
    """Input feature names that survive both filters of a fitted pipeline."""
    after_prev = pipeline.named_steps["prevalence"].get_feature_names_out()
    return [str(f) for f in pipeline.named_steps["correlation"].get_feature_names_out(after_prev)]


@dataclass
class CVResult:
    """Out-of-fold outputs of one model under one validation design."""

    model: str
    design: str
    predictions: list[dict[str, Any]] = field(default_factory=list)
    folds: list[dict[str, Any]] = field(default_factory=list)
    importances: list[dict[str, Any]] = field(default_factory=list)
    permutation: list[dict[str, Any]] = field(default_factory=list)


def _inner_cv(groups: np.ndarray | None, n_splits: int, seed: int) -> Any:
    """Inner (tuning) splitter; grouped when the outer design is grouped.

    A training fold of the grouped design may contain fewer species than the
    requested number of inner folds; the number of folds is then reduced to the
    number of species. With fewer than two species, grouped tuning is impossible
    and stratified folds are used instead (logged).
    """
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        log.warning("Only %d species in this training fold; using stratified inner CV", n_groups)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)


def outer_splits(
    y: pd.Series, cfg: Config, groups: pd.Series | None = None
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (repeat, fold, train_idx, test_idx) for the outer loop.

    Random design: RepeatedStratifiedKFold. Grouped design: StratifiedGroupKFold
    with species as groups (single repeat; groups make repeats near-identical).
    """
    cv = cfg["cv"]
    k = int(cv["outer_folds"])
    if groups is None:
        splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=int(cv["outer_repeats"]), random_state=cfg.seed)
        for i, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y)):
            yield i // k, i % k, tr, te
        return
    n_groups = groups.nunique()
    k = min(int(cv.get("species_grouped_folds", k)), n_groups)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    for i, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        yield 0, i, tr, te


def _grid_search(name: str, cfg: Config, inner_groups: np.ndarray | None, seed: int) -> GridSearchCV:
    return GridSearchCV(
        make_pipeline(name, cfg),
        param_grid(name, cfg),
        scoring=cfg.data.get("selection_metric", "balanced_accuracy"),
        cv=_inner_cv(inner_groups, int(cfg["cv"]["inner_folds"]), seed),
        n_jobs=cfg["cv"].get("n_jobs", -1),
        refit=True,
        error_score=np.nan,
    )


def nested_cv(
    name: str,
    data: PreparedData,
    cfg: Config,
    design: str = DESIGN_RANDOM,
    sample_index: pd.Index | None = None,
) -> CVResult:
    """Run nested cross-validation for one model.

    Args:
        name: Model name (key of :data:`MODEL_LABELS`).
        data: Analysis dataset.
        cfg: Configuration.
        design: ``DESIGN_RANDOM`` or ``DESIGN_SPECIES``.
        sample_index: Optional subset of sample ids (e.g. the training part when
            a hold-out test set is used).

    Returns:
        A :class:`CVResult` with out-of-fold predictions, per-fold settings and
        per-fold feature importances.
    """
    index = data.X.index if sample_index is None else sample_index
    X, y = data.X.loc[index], data.y.loc[index]
    groups = data.groups.loc[index] if design == DESIGN_SPECIES else None
    result = CVResult(model=name, design=design)
    interp = cfg.data.get("interpretation", {}) or {}
    perm_top_k = int(interp.get("permutation_top_k", 0) or 0)
    perm_repeats = int(interp.get("permutation_repeats", 5) or 5)
    # Each outer fold gives one test estimate; hyper-parameters are tuned only inside its training part (tr).
    for repeat, fold, tr, te in outer_splits(y, cfg, groups):
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        inner_groups = groups.iloc[tr].to_numpy() if groups is not None else None
        if y_tr.nunique() < 2:
            log.warning("%s %s fold %d: training set has one class; skipped", name, design, fold)
            continue
        # Each repeat and fold gets its own seed so the inner splits differ between them.
        search = _grid_search(name, cfg, inner_groups, seed=cfg.seed + 1000 * repeat + fold)
        start = time.perf_counter()
        with warnings.catch_warnings():
            # Small inner folds can contain one class for grouped CV; such
            # candidates score NaN (error_score) instead of aborting the run.
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            search.fit(X.iloc[tr], y_tr, **({"groups": inner_groups} if inner_groups is not None else {}))
        model = search.best_estimator_
        # Look up the column for class 1 (pathogenic) by label, not by position.
        proba = model.predict_proba(X.iloc[te])[:, list(model.classes_).index(1)]
        pred = model.predict(X.iloc[te])
        n_selected = len(selected_features(model))
        imp = fitted_importances(model)
        result.folds.append(
            {
                "model": name,
                "design": design,
                "repeat": repeat,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "n_test_pathogenic": int(y_te.sum()),
                "n_test_non_pathogenic": int((y_te == 0).sum()),
                "test_groups": ";".join(sorted(set(groups.iloc[te]))) if groups is not None else "",
                "inner_best_score": float(search.best_score_),
                "best_params": repr(search.best_params_),
                "n_features_after_filters": n_selected,
                "n_features_used": int((imp != 0).sum()),
                "fit_seconds": round(time.perf_counter() - start, 3),
            }
        )
        for sid, yt, yp, ys in zip(y_te.index, y_te.to_numpy(), pred, proba):
            result.predictions.append(
                {"model": name, "design": design, "repeat": repeat, "fold": fold, "sample_id": sid,
                 "y_true": int(yt), "y_pred": int(yp), "y_score": float(ys)}
            )
        # Only non-zero importances are stored; zero means the feature was not used by this fold's model.
        for feature, value in imp[imp != 0].items():
            result.importances.append(
                {"model": name, "design": design, "repeat": repeat, "fold": fold, "feature": feature,
                 "value": float(value)}
            )
        if design == DESIGN_RANDOM and perm_top_k > 0:
            # Held-out permutation importance for the fold model's top features.
            top = imp.abs()[imp != 0].sort_values(ascending=False).head(perm_top_k).index.tolist()
            drops = permutation_importance_subset(
                model, X.iloc[te], y_te, top, perm_repeats, seed=cfg.seed + 1000 * repeat + fold
            )
            for feature, drop in drops.items():
                result.permutation.append({"model": name, "design": design, "repeat": repeat, "fold": fold,
                                           "feature": feature, "importance": drop})
        log.debug("%s %s r%d f%d done in %.1fs", name, design, repeat, fold, result.folds[-1]["fit_seconds"])
    return result


def fit_final_model(name: str, X: pd.DataFrame, y: pd.Series, cfg: Config) -> GridSearchCV:
    """Tune hyper-parameters by stratified CV on (X, y) and refit on all of it."""
    search = _grid_search(name, cfg, None, seed=cfg.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.fit(X, y)
    return search


def environment_versions() -> dict[str, str]:
    """Library versions stored with every saved model (needed to reload safely)."""
    import matplotlib
    import numpy
    import pandas

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "xanthomonas_ml": __version__,
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }


def save_model_bundle(path: Path, bundle: dict[str, Any]) -> Path:
    """Serialise a model bundle with joblib and a human-readable JSON sidecar."""
    ensure_dir(path.parent)
    joblib.dump(bundle, path, compress=3)
    sidecar = {k: v for k, v in bundle.items() if k != "pipeline"}
    write_json(path.with_suffix(".json"), sidecar)
    return path


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load a bundle saved by :func:`save_model_bundle`.

    Raises:
        FileNotFoundError: if the model file does not exist.
        ValueError: if the file is not a valid model bundle.

    Security note: joblib/pickle files can execute code when loaded. Only load
    model files that you created yourself or obtained from a trusted source.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}. Train first with 'python -m xanthomonas_ml train'.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_schema" not in bundle:
        raise ValueError(f"{path} is not a model bundle created by xanthomonas_ml")
    saved = bundle.get("versions", {}).get("scikit-learn")
    if saved and saved != sklearn.__version__:
        log.warning(
            "Model was saved with scikit-learn %s but %s is installed; results may differ. Retrain if possible.",
            saved, sklearn.__version__,
        )
    return bundle


def run_training(
    data: PreparedData, cfg: Config, models: list[str] | None = None, skip_species_cv: bool = False
) -> dict[str, Any]:
    """Run the full training protocol and write all raw results to disk.

    Writes to ``outputs/results`` (predictions, fold details, importances) and
    ``models/`` (one joblib bundle + JSON sidecar per model).

    Returns:
        A summary dictionary (also saved as ``outputs/results/training_run.json``).
    """
    from xanthomonas_ml.evaluate import compute_metrics, summarise_results

    results_dir = ensure_dir(cfg.outputs_dir / "results")
    names = enabled_models(cfg, models)
    holdout = float(cfg["cv"].get("holdout_fraction", 0) or 0)
    # Without a hold-out fraction, every strain is used for both CV and the final model.
    train_index, test_index = data.X.index, pd.Index([])
    if holdout > 0:
        train_index, test_index = train_test_split(
            data.X.index, test_size=holdout, stratify=data.y, random_state=cfg.seed
        )
        train_index, test_index = pd.Index(train_index), pd.Index(test_index)
        log.info("Hold-out test set: %d strains (never used for CV or tuning)", len(test_index))

    designs = [DESIGN_RANDOM]
    if cfg["cv"].get("species_grouped", True) and not skip_species_cv:
        designs.append(DESIGN_SPECIES)

    all_preds, all_folds, all_imps, all_perms, model_records = [], [], [], [], {}
    for name in names:
        for design in designs:
            log.info("Nested CV: %s / %s", MODEL_LABELS[name], design)
            start = time.perf_counter()
            res = nested_cv(name, data, cfg, design=design, sample_index=train_index)
            log.info("  finished %d outer folds in %.1fs", len(res.folds), time.perf_counter() - start)
            all_preds += res.predictions
            all_folds += res.folds
            all_imps += res.importances
            all_perms += res.permutation

        # Final model: tune and refit on the CV-eligible strains only (the hold-out, if any, is excluded).
        X_fit, y_fit = data.X.loc[train_index], data.y.loc[train_index]
        search = fit_final_model(name, X_fit, y_fit, cfg)
        final = search.best_estimator_
        train_pred = final.predict(X_fit)
        train_score = final.predict_proba(X_fit)[:, list(final.classes_).index(1)]
        # Metrics on the training strains themselves: a sanity check only, never reported as performance.
        record: dict[str, Any] = {
            "model_name": name,
            "model_label": MODEL_LABELS[name],
            "trained_utc": utc_now_iso(),
            "best_params": {k: v for k, v in search.best_params_.items()},
            "inner_cv_best_score": float(search.best_score_),
            "selection_metric": cfg.data.get("selection_metric", "balanced_accuracy"),
            "n_training_samples": int(len(y_fit)),
            "training_class_counts": {int(k): int(v) for k, v in y_fit.value_counts().items()},
            "n_features_after_filters": len(selected_features(final)),
            "resubstitution_metrics_NOT_generalisation": compute_metrics(y_fit.to_numpy(), train_pred, train_score),
        }
        if len(test_index):
            X_te, y_te = data.X.loc[test_index], data.y.loc[test_index]
            te_score = final.predict_proba(X_te)[:, list(final.classes_).index(1)]
            te_pred = final.predict(X_te)
            record["holdout_metrics"] = compute_metrics(y_te.to_numpy(), te_pred, te_score)
            for sid, yt, yp, ys in zip(y_te.index, y_te.to_numpy(), te_pred, te_score):
                all_preds.append({"model": name, "design": DESIGN_HOLDOUT, "repeat": 0, "fold": 0,
                                  "sample_id": sid, "y_true": int(yt), "y_pred": int(yp), "y_score": float(ys)})
        # Everything needed to reuse the model later: the fitted pipeline, the input schema and provenance.
        bundle = {
            **record,
            "pipeline": final,
            "feature_schema": [str(c) for c in data.X.columns],
            "selected_features": selected_features(final),
            "correlated_groups": final.named_steps["correlation"].correlated_groups(),
            "label_mapping": data.info["label_mapping"],
            "positive_class": data.info["positive_class"],
            "feature_encoding": "binary Pfam domain presence (1) / absence (0) per genome",
            "training_data_sha256": data.info.get("source_sha256"),
            "versions": environment_versions(),
            "config_seed": cfg.seed,
        }
        save_model_bundle(cfg.models_dir / f"{name}.joblib", bundle)
        model_records[name] = record
        log.info("Saved final %s to %s", name, (cfg.models_dir / f"{name}.joblib").as_posix())

    # Raw outputs are saved first; metrics are computed from these files afterwards.
    preds = pd.DataFrame(all_preds)
    folds = pd.DataFrame(all_folds)
    imps = pd.DataFrame(all_imps, columns=["model", "design", "repeat", "fold", "feature", "value"])
    preds.to_csv(results_dir / "cv_predictions.csv", index=False)
    folds.to_csv(results_dir / "cv_folds.csv", index=False)
    imps.to_csv(results_dir / "cv_fold_importances.csv", index=False)
    pd.DataFrame(all_perms, columns=["model", "design", "repeat", "fold", "feature", "importance"]).to_csv(
        results_dir / "cv_permutation_importances.csv", index=False
    )
    run = {
        "created_utc": utc_now_iso(),
        "models": names,
        "designs": designs + ([DESIGN_HOLDOUT] if len(test_index) else []),
        "n_samples_total": int(len(data.y)),
        "n_samples_cv": int(len(train_index)),
        "n_samples_holdout": int(len(test_index)),
        "holdout_ids": [str(i) for i in test_index],
        "cv_settings": cfg["cv"],
        "preprocessing": cfg["preprocessing"],
        "param_grids": {n: param_grid(n, cfg) for n in names},
        "model_settings": cfg["models"],
        "selection_metric": cfg.data.get("selection_metric", "balanced_accuracy"),
        "seed": cfg.seed,
        "final_models": model_records,
        "versions": environment_versions(),
    }
    write_json(results_dir / "training_run.json", run)
    # Compute the metric summaries and the best-model ranking from the saved predictions.
    summarise_results(cfg)
    return run
