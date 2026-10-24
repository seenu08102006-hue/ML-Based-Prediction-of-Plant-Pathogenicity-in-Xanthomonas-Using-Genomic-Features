"""Interpretation of trained models: which Pfam domains drive the predictions.

Three complementary, honest views are produced for every model:

1. **Cross-validated stability** - for every outer CV fold of the stratified
   design, the fold model's importances (tree/forest impurity importance or
   L1 logistic coefficient). We report how often a domain was used and its
   mean importance; a domain that is used in only a few folds is unstable.
2. **Held-out permutation importance** - for each outer fold, the drop in
   balanced accuracy on the *held-out* fold when one of the fold model's top
   domains is randomly permuted. Computed only for the top-k domains of each
   fold model to keep runtime manageable.
3. **Final model** - importances/coefficients of the model refitted on all
   strains, with the correlated domains it represents (removed by the
   correlation filter, so they are interchangeable with the kept domain).

Important caveats (also written into the outputs):
* Correlated/co-inherited domains share or swap importance; a low importance
  does not mean a domain is irrelevant.
* Importance reflects statistical association in ~118 related strains. It is not
  evidence that a domain causes pathogenicity, and no domain here is a
  validated biomarker.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from xanthomonas_ml.config import Config
from xanthomonas_ml.utils import ensure_dir, get_logger

log = get_logger("interpret")

CAVEAT = (
    "Importances describe statistical associations learned from ~118 related strains; correlated domains can "
    "substitute for each other. They do not show that a domain causes pathogenicity and are not validated biomarkers."
)


def permutation_importance_subset(
    model: Any, X: pd.DataFrame, y: pd.Series, features: list[str], n_repeats: int, seed: int
) -> dict[str, float]:
    """Mean drop in balanced accuracy when each listed feature is permuted.

    Only the columns in ``features`` are permuted (the pipeline's input has
    thousands of columns, most of which the fitted filters discard).

    Returns:
        Mapping feature -> mean score drop (positive = feature helped). Empty if
        ``y`` contains a single class (balanced accuracy undefined).
    """
    if y.nunique() < 2 or not features:
        return {}
    rng = np.random.default_rng(seed)
    # Baseline score on the unpermuted held-out data.
    base = balanced_accuracy_score(y, model.predict(X))
    drops: dict[str, float] = {}
    for feature in features:
        scores = []
        for _ in range(n_repeats):
            permuted = X.copy()
            # Shuffle one column and leave the rest intact; a large score drop means the model relied on it.
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            scores.append(balanced_accuracy_score(y, model.predict(permuted)))
        drops[feature] = float(base - np.mean(scores))
    return drops


def _descriptions(cfg: Config) -> pd.Series:
    from xanthomonas_ml.data_loader import load_descriptions

    name = cfg["dataset"]["files"].get("domain_descriptions", {}).get("filename")
    df = load_descriptions(cfg.raw_dir / name) if name else None
    return pd.Series(dtype=str) if df is None else df.set_index("accession")["description"]


def cv_stability_table(imps: pd.DataFrame, folds: pd.DataFrame, model: str, design: str) -> pd.DataFrame:
    """Per-feature usage frequency and mean importance across outer folds."""
    n_folds = int(((folds.model == model) & (folds.design == design)).sum())
    sub = imps[(imps.model == model) & (imps.design == design)]
    if sub.empty or n_folds == 0:
        return pd.DataFrame()
    g = sub.groupby("feature")["value"]
    table = pd.DataFrame(
        {
            "folds_used": g.size(),
            "selection_frequency": g.size() / n_folds,
            # Folds in which the feature was filtered out or unused count as 0.
            "mean_importance_cv": g.sum() / n_folds,
            "mean_abs_importance_cv": g.apply(lambda v: v.abs().sum()) / n_folds,
        }
    )
    if model == "l1_logistic_regression":
        table["sign_consistency"] = g.apply(lambda v: max((v > 0).mean(), (v < 0).mean()))
    table["n_outer_folds"] = n_folds
    return table.sort_values("mean_abs_importance_cv", ascending=False)


def run_interpretation(cfg: Config, data: Any | None = None) -> dict[str, Any]:
    """Build importance tables and figures for every trained model.

    Args:
        cfg: Configuration.
        data: Optional prepared dataset (for class-wise prevalence); loaded if None.

    Returns:
        Mapping model -> list of the top features (by cross-validated importance).
    """
    from xanthomonas_ml.feature_engineering import feature_prevalence_by_class
    from xanthomonas_ml.preprocessing import load_prepared
    from xanthomonas_ml.train import MODEL_LABELS, fitted_importances, load_model_bundle

    res_dir = cfg.outputs_dir / "results"
    paths = {name: res_dir / name for name in ("cv_fold_importances.csv", "cv_folds.csv")}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found. Run 'python -m xanthomonas_ml train' first.")
    imps = pd.read_csv(paths["cv_fold_importances.csv"])
    folds = pd.read_csv(paths["cv_folds.csv"])
    perm_path = res_dir / "cv_permutation_importances.csv"
    perms = pd.read_csv(perm_path) if perm_path.is_file() else pd.DataFrame()

    data = data if data is not None else load_prepared(cfg.processed_dir, cfg)
    # Descriptive class prevalence, attached to each table to help read the importances (never used for selection).
    prevalence = feature_prevalence_by_class(data.X, data.y)[
        ["prevalence_pathogenic", "prevalence_non_pathogenic", "more_common_in"]
    ]
    desc = _descriptions(cfg)
    tables_dir, fig_dir = ensure_dir(cfg.tables_dir), ensure_dir(cfg.figures_dir)
    top_n = int(cfg.data.get("interpretation", {}).get("top_n", 25))
    design = "stratified_nested_cv"
    out: dict[str, Any] = {}

    for model in sorted(folds.model.unique()):
        table = cv_stability_table(imps, folds, model, design)
        if table.empty:
            continue
        # Permutation drops exist only for each fold model's top domains; other domains stay empty here.
        if not perms.empty:
            p = perms[(perms.model == model) & (perms.design == design)]
            n_f = int(((folds.model == model) & (folds.design == design)).sum())
            agg = p.groupby("feature")["importance"].agg(["mean", "size"])
            table["heldout_permutation_drop_mean"] = agg["mean"]
            table["heldout_permutation_folds"] = agg["size"].reindex(table.index).fillna(0).astype(int)
            table.attrs["n_folds"] = n_f
        bundle_path = cfg.models_dir / f"{model}.joblib"
        # Add the final model's importances and the correlated domains each kept domain stands for.
        if bundle_path.is_file():
            bundle = load_model_bundle(bundle_path)
            final = fitted_importances(bundle["pipeline"])
            table = table.join(final.rename("final_model_importance"), how="outer")
            groups = bundle.get("correlated_groups", {})
            table["correlated_domains_represented"] = [";".join(groups.get(f, [])) for f in table.index]
        table = table.join(prevalence, how="left")
        table["description"] = desc.reindex(table.index).to_numpy()
        table.index.name = "feature"
        table = table.sort_values("mean_abs_importance_cv", ascending=False, na_position="last")
        table.to_csv(tables_dir / f"feature_importance_{model}.csv")
        plot_top_features(table.head(top_n), model, MODEL_LABELS[model], fig_dir / f"interp_top_features_{model}.png",
                          cfg)
        out[model] = table.head(top_n).index.tolist()

    _write_caveat(tables_dir)
    plot_selected_feature_prevalence(cfg, out, data, prevalence, desc, fig_dir / "interp_top_feature_prevalence.png")
    log.info("Feature-importance tables written to %s", tables_dir.as_posix())
    return out


def _write_caveat(tables_dir) -> None:
    (tables_dir / "feature_importance_README.txt").write_text(
        "Columns of feature_importance_<model>.csv\n"
        "  folds_used / selection_frequency: number / share of outer CV folds whose model used the domain\n"
        "  mean_importance_cv: mean fold importance (impurity importance, or signed L1 coefficient; 0 when unused)\n"
        "  mean_abs_importance_cv: mean absolute fold importance (ranking key)\n"
        "  sign_consistency (L1 LR only): share of folds where the coefficient had its majority sign\n"
        "  heldout_permutation_drop_mean: mean drop in held-out balanced accuracy when the domain is permuted\n"
        "      (only computed for each fold model's top domains; empty otherwise)\n"
        "  final_model_importance: importance/coefficient in the model refitted on all strains\n"
        "  correlated_domains_represented: domains removed as |r| > threshold that the kept domain stands for\n"
        "  prevalence_* / more_common_in: descriptive class-wise prevalence on all strains\n"
        "Positive L1 coefficients push predictions towards 'pathogenic', negative towards 'non-pathogenic'.\n\n"
        + CAVEAT + "\n",
        encoding="utf-8",
    )


def plot_top_features(table: pd.DataFrame, model: str, label: str, path, cfg: Config) -> None:
    """Horizontal bar chart of the top domains by cross-validated importance."""
    from xanthomonas_ml.plotting import CLASS_COLORS, DIVERGING_NEG, DIVERGING_POS, add_note, new_figure, save_figure

    if table.empty:
        return
    t = table.iloc[::-1]
    names = [f"{f}  {str(d)[:45] if isinstance(d, str) else ''}" for f, d in zip(t.index, t["description"])]
    fig, ax = new_figure(9.5, 0.3 * len(t) + 1.8)
    if model == "l1_logistic_regression":
        values = t["mean_importance_cv"]
        colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in values]
        ax.barh(names, values, color=colors, height=0.65)
        ax.axvline(0, color="#c3c2b7", linewidth=1)
        ax.set_xlabel("Mean L1 coefficient across outer CV folds (0 when not selected)")
        ax.legend(handles=[_patch(DIVERGING_POS, "towards pathogenic"), _patch(DIVERGING_NEG, "towards non-pathogenic")],
                  loc="lower right")
    else:
        values = t["mean_importance_cv"]
        colors = [CLASS_COLORS.get(c, "#898781") for c in t["more_common_in"].fillna("equal")]
        ax.barh(names, values, color=colors, height=0.65)
        ax.set_xlabel("Mean impurity importance across outer CV folds (0 when not used)")
        ax.legend(handles=[_patch(CLASS_COLORS["pathogenic"], "more common in pathogenic strains"),
                           _patch(CLASS_COLORS["non-pathogenic"], "more common in non-pathogenic strains")],
                  loc="lower right")
    for i, freq in enumerate(t["selection_frequency"]):
        ax.text(ax.get_xlim()[1], i, f" {freq:.0%}", va="center", fontsize=7.5, color="#52514e")
    ax.set_title(f"Top Pfam domains - {label}")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    add_note(fig, "Right-hand %: share of outer CV folds in which the domain was used. Association only - not causation.")
    save_figure(fig, path)
    _register_caption(cfg, path.name,
                      f"Top {len(t)} Pfam domains of the {label} ranked by mean absolute importance across the outer "
                      "folds of nested stratified CV; percentages give the share of folds using the domain. "
                      + ("Bar colour shows the coefficient sign." if model == "l1_logistic_regression"
                         else "Bar colour shows the class in which the domain is more prevalent."))


def plot_selected_feature_prevalence(cfg, top: dict[str, list[str]], data, prevalence, desc, path) -> None:
    """Dot plot of class-wise prevalence for domains ranked highly by any model."""
    from xanthomonas_ml.plotting import CLASS_COLORS, add_note, new_figure, save_figure

    features: list[str] = []
    for feats in top.values():
        for f in feats[:8]:
            if f not in features and f in prevalence.index:
                features.append(f)
    if not features:
        return
    p = prevalence.loc[features].sort_values("prevalence_pathogenic")
    fig, ax = new_figure(8.5, 0.32 * len(p) + 1.8)
    y = np.arange(len(p))
    ax.hlines(y, p["prevalence_non_pathogenic"], p["prevalence_pathogenic"], color="#c3c2b7", linewidth=1.5)
    ax.scatter(p["prevalence_non_pathogenic"], y, color=CLASS_COLORS["non-pathogenic"], s=30, label="non-pathogenic", zorder=3)
    ax.scatter(p["prevalence_pathogenic"], y, color=CLASS_COLORS["pathogenic"], s=30, label="pathogenic", zorder=3)
    labels = [f"{f}  {str(desc.get(f, ''))[:40]}" for f in p.index]
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Prevalence (fraction of strains in the class with the domain)")
    ax.set_title("Class-wise prevalence of top-ranked domains")
    ax.legend(title="Strains", loc="lower right")
    ax.grid(axis="y", visible=False)
    add_note(fig, "Domains among the top 8 of any model. Descriptive prevalence on all strains; not causal evidence.")
    save_figure(fig, path)
    _register_caption(cfg, path.name,
                      "Prevalence of the domains ranked highest by any model in pathogenic and non-pathogenic strains. "
                      "Most informative domains are present in only part of one class, consistent with the view that the "
                      "phenotype reflects an accumulation of traits rather than any single domain.")


def _patch(color: str, label: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=color, label=label)


def _register_caption(cfg: Config, filename: str, caption: str) -> None:
    from xanthomonas_ml.eda import FigureRegistry

    reg = FigureRegistry(cfg.figures_dir)
    reg.add(filename, caption)
    reg.save()
