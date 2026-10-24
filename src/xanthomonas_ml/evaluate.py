"""Metrics, result summaries and evaluation figures.

Conventions:
* Positive class = 1 = 'pathogenic' (as in the reference study).
* A metric that is mathematically undefined for a set of predictions (e.g.
  ROC-AUC when the validation fold holds a single class, precision when no
  sample is predicted positive) is reported as NaN with a reason. It is never
  replaced by zero, and summaries state how many folds contributed.
* Out-of-fold metrics are computed in two ways for repeated CV:
  - per outer fold (small test sets, high variance), and
  - pooled per repeat (every strain is predicted exactly once per repeat),
  and both are reported with their spread across folds/repeats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from xanthomonas_ml.config import Config
from xanthomonas_ml.utils import ensure_dir, get_logger, read_json, write_json

log = get_logger("evaluate")

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc_average_precision",
]
DESIGN_LABELS = {
    "stratified_nested_cv": "Nested stratified CV",
    "species_grouped_nested_cv": "Nested species-grouped CV",
    "holdout_test": "Held-out test set",
}


def _safe_div(num: float, den: float) -> float:
    # Return NaN for an empty denominator so the metric stays visibly undefined, not 0.
    return float(num) / float(den) if den else float("nan")


def compute_metrics(y_true: Any, y_pred: Any, y_score: Any | None = None) -> dict[str, Any]:
    """Binary classification metrics with explicit handling of undefined values.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        y_score: Predicted probability of class 1 (optional; needed for AUCs).

    Returns:
        Dict with each metric (float or NaN), confusion counts and an
        ``undefined`` mapping metric -> reason for every NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on an empty set")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    undefined: dict[str, str] = {}
    out: dict[str, Any] = {"n": int(len(y_true)), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    out["accuracy"] = _safe_div(tp + tn, len(y_true))
    out["recall_sensitivity"] = _safe_div(tp, tp + fn)
    if tp + fn == 0:
        undefined["recall_sensitivity"] = "no pathogenic samples in this set"
    out["specificity"] = _safe_div(tn, tn + fp)
    if tn + fp == 0:
        undefined["specificity"] = "no non-pathogenic samples in this set"
    out["precision"] = _safe_div(tp, tp + fp)
    if tp + fp == 0:
        undefined["precision"] = "no sample predicted pathogenic"
    out["f1"] = _safe_div(2 * tp, 2 * tp + fp + fn)
    if 2 * tp + fp + fn == 0:
        undefined["f1"] = "no positive samples and no positive predictions"
    # Balanced accuracy and both AUCs need at least one sample from each class.
    both = len(np.unique(y_true)) == 2
    if both:
        out["balanced_accuracy"] = (out["recall_sensitivity"] + out["specificity"]) / 2
    else:
        out["balanced_accuracy"] = float("nan")
        undefined["balanced_accuracy"] = "only one class present"
    if y_score is None:
        out["roc_auc"] = out["pr_auc_average_precision"] = float("nan")
        undefined["roc_auc"] = undefined["pr_auc_average_precision"] = "no probability scores"
    elif not both:
        out["roc_auc"] = out["pr_auc_average_precision"] = float("nan")
        undefined["roc_auc"] = undefined["pr_auc_average_precision"] = "only one class present"
    else:
        y_score = np.asarray(y_score, dtype=float)
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc_average_precision"] = float(average_precision_score(y_true, y_score))
    out["undefined"] = undefined
    return out


def _describe(values: pd.Series) -> dict[str, Any]:
    v = values.dropna()
    if v.empty:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan,
                "min": np.nan, "max": np.nan, "n_defined": 0, "n_total": int(len(values))}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        "median": float(v.median()),
        "q25": float(v.quantile(0.25)),
        "q75": float(v.quantile(0.75)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n_defined": int(len(v)),
        "n_total": int(len(values)),
    }


def results_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "results"


def load_predictions(cfg: Config) -> pd.DataFrame:
    """Load out-of-fold predictions written by training.

    Raises:
        FileNotFoundError: if training has not been run.
    """
    path = results_dir(cfg) / "cv_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run 'python -m xanthomonas_ml train' first.")
    return pd.read_csv(path, dtype={"sample_id": str})


def metric_table(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Metrics for every group of predictions defined by ``by`` columns."""
    rows = []
    for keys, grp in preds.groupby(by, sort=True):
        # A single grouping column yields a scalar key in pandas; wrap it so zip() works for both cases.
        keys = keys if isinstance(keys, tuple) else (keys,)
        m = compute_metrics(grp["y_true"], grp["y_pred"], grp["y_score"])
        undefined = m.pop("undefined")
        rows.append({**dict(zip(by, keys)), **m, "undefined": "; ".join(f"{k}: {v}" for k, v in undefined.items())})
    return pd.DataFrame(rows)


def summarise_results(cfg: Config) -> dict[str, Any]:
    """Compute per-fold, per-repeat and summary metrics from saved predictions.

    Writes ``metrics_per_fold.csv``, ``metrics_per_repeat.csv``,
    ``metrics_summary.csv/json`` and ``best_model.json`` to ``outputs/results``.
    """
    out_dir = ensure_dir(results_dir(cfg))
    preds = load_predictions(cfg)
    per_fold = metric_table(preds, ["model", "design", "repeat", "fold"])
    per_repeat = metric_table(preds, ["model", "design", "repeat"])
    per_fold.to_csv(out_dir / "metrics_per_fold.csv", index=False)
    per_repeat.to_csv(out_dir / "metrics_per_repeat.csv", index=False)

    # Each metric is summarised twice: across repeats (pooled) and across outer folds.
    rows = []
    for (model, design), grp in per_repeat.groupby(["model", "design"]):
        folds = per_fold[(per_fold.model == model) & (per_fold.design == design)]
        for metric in METRICS:
            rows.append({"model": model, "design": design, "metric": metric, "level": "pooled_per_repeat",
                         **_describe(grp[metric])})
            rows.append({"model": model, "design": design, "metric": metric, "level": "per_outer_fold",
                         **_describe(folds[metric])})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    # Rank models on the selection metric from the stratified CV, using the pooled-per-repeat values.
    selection = cfg.data.get("selection_metric", "balanced_accuracy")
    primary = summary[(summary.design == "stratified_nested_cv") & (summary.level == "pooled_per_repeat")
                      & (summary.metric == selection)].sort_values("mean", ascending=False)
    best: dict[str, Any] = {"selection_metric": selection, "design": "stratified_nested_cv",
                            "level": "pooled_per_repeat", "ranking": []}
    for _, r in primary.iterrows():
        best["ranking"].append({"model": r["model"], "mean": r["mean"], "std": r["std"], "n_repeats": r["n_defined"]})
    if best["ranking"]:
        best["best_model"] = best["ranking"][0]["model"]
        if len(best["ranking"]) > 1:
            a, b = best["ranking"][0], best["ranking"][1]
            spread = (f"spread across repeats: sd {a['std']:.3f} vs {b['std']:.3f}"
                      if not (pd.isna(a["std"]) or pd.isna(b["std"])) else "sd undefined with a single repeat")
            best["note"] = (
                f"Difference between the top two models in mean {selection}: {a['mean'] - b['mean']:.3f} "
                f"({spread}). Small differences relative to the spread should not be interpreted as one model "
                "being genuinely better."
            )
    write_json(out_dir / "best_model.json", best)
    write_json(out_dir / "metrics_summary.json", summary.to_dict(orient="records"))
    return best


def per_strain_probabilities(preds: pd.DataFrame, design: str = "stratified_nested_cv") -> pd.DataFrame:
    """Mean out-of-fold P(pathogenic) per strain and model across repeats."""
    # One row per strain and model, averaged over all CV repeats.
    sub = preds[preds.design == design]
    table = sub.groupby(["model", "sample_id"]).agg(
        y_true=("y_true", "first"), mean_prob_pathogenic=("y_score", "mean"),
        sd_prob_pathogenic=("y_score", "std"), fraction_misclassified=("y_pred", "mean"),
    ).reset_index()
    # fraction misclassified = share of repeats with a wrong prediction
    table["fraction_misclassified"] = np.where(
        table["y_true"] == 1, 1 - table["fraction_misclassified"], table["fraction_misclassified"]
    )
    return table


# --------------------------------------------------------------------------- plots
def plot_confusion_matrices(preds: pd.DataFrame, design: str, path: Path) -> Path | None:
    """Row-normalised confusion matrix per model, averaged over CV repeats."""
    from xanthomonas_ml.plotting import INK, add_note, new_figure, save_figure

    sub = preds[preds.design == design]
    if sub.empty:
        return None
    models = sorted(sub.model.unique())
    fig, axes = new_figure(4.0 * len(models), 4.0, ncols=len(models))
    axes = np.atleast_1d(axes).ravel()
    n_rep = sub.groupby("model")["repeat"].nunique().max()
    for ax, model in zip(axes, models):
        g = sub[sub.model == model]
        # Divide by the number of repeats so the counts are per-repeat averages, not totals over all repeats.
        cm = confusion_matrix(g.y_true, g.y_pred, labels=[0, 1]).astype(float) / g["repeat"].nunique()
        rates = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{rates[i, j]:.0%}\n({cm[i, j]:.1f})", ha="center", va="center",
                        color="white" if rates[i, j] > 0.6 else INK, fontsize=10)
        ax.set_xticks([0, 1], ["non-pathogenic", "pathogenic"])
        ax.set_yticks([0, 1], ["non-pathogenic", "pathogenic"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class (source label)")
        # Imported here, not at the top, to avoid a circular import with train.py.
        from xanthomonas_ml.train import MODEL_LABELS

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10)
    fig.suptitle(f"Confusion matrices - {DESIGN_LABELS.get(design, design)} (out-of-fold)", fontweight="bold")
    add_note(fig, f"Cells: % of the true class (mean strain count per CV repeat in brackets); averaged over {n_rep} repeat(s).")
    return save_figure(fig, path)


def plot_roc_pr(preds: pd.DataFrame, design: str, roc_path: Path, pr_path: Path) -> list[Path]:
    """ROC and precision-recall curves per model (one pooled curve per repeat)."""
    from xanthomonas_ml.plotting import INK_MUTED, MODEL_COLORS, add_note, new_figure, save_figure
    from xanthomonas_ml.train import MODEL_LABELS

    sub = preds[preds.design == design]
    if sub.empty or sub.y_true.nunique() < 2:
        log.warning("ROC/PR curves not drawn for %s: need both classes", design)
        return []
    saved = []
    fig_r, ax_r = new_figure(6.0, 5.2)
    fig_p, ax_p = new_figure(6.0, 5.2)
    for model in sorted(sub.model.unique()):
        color = MODEL_COLORS.get(model)
        aucs, aps = [], []
        for _, g in sub[sub.model == model].groupby("repeat"):
            if g.y_true.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(g.y_true, g.y_score)
            prec, rec, _ = precision_recall_curve(g.y_true, g.y_score)
            aucs.append(auc(fpr, tpr))
            aps.append(average_precision_score(g.y_true, g.y_score))
            ax_r.plot(fpr, tpr, color=color, alpha=0.25, linewidth=1)
            ax_p.plot(rec, prec, color=color, alpha=0.25, linewidth=1)
        label = MODEL_LABELS.get(model, model)
        # Invisible proxy line carries the legend entry with summary statistics.
        ax_r.plot([], [], color=color, label=f"{label}: AUC {np.mean(aucs):.2f} ± {np.std(aucs):.2f}")
        ax_p.plot([], [], color=color, label=f"{label}: AP {np.mean(aps):.2f} ± {np.std(aps):.2f}")
    prevalence = sub.drop_duplicates("sample_id").y_true.mean()
    ax_r.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1, label="Chance")
    ax_p.axhline(prevalence, linestyle="--", color=INK_MUTED, linewidth=1, label=f"Chance (prevalence {prevalence:.2f})")
    ax_r.set(xlabel="False-positive rate (1 - specificity)", ylabel="True-positive rate (sensitivity)",
             xlim=(0, 1), ylim=(0, 1.01), title=f"ROC curves - {DESIGN_LABELS.get(design, design)}")
    ax_p.set(xlabel="Recall (sensitivity, pathogenic class)", ylabel="Precision (pathogenic class)",
             xlim=(0, 1), ylim=(0, 1.01), title=f"Precision-recall curves - {DESIGN_LABELS.get(design, design)}")
    ax_r.legend(loc="lower right")
    ax_p.legend(loc="lower left")
    note = "One thin line per CV repeat (pooled out-of-fold predictions); legend: mean ± sd over repeats."
    add_note(fig_r, note)
    add_note(fig_p, note)
    saved.append(save_figure(fig_r, roc_path))
    saved.append(save_figure(fig_p, pr_path))
    return saved


def plot_model_comparison(per_repeat: pd.DataFrame, path: Path) -> Path | None:
    """Dot/box plot of pooled-per-repeat metrics for each model and design."""
    from xanthomonas_ml.plotting import MODEL_COLORS, add_note, new_figure, save_figure
    from xanthomonas_ml.train import MODEL_LABELS

    if per_repeat.empty:
        return None
    metrics = ["balanced_accuracy", "accuracy", "recall_sensitivity", "specificity", "f1", "roc_auc"]
    designs = [d for d in DESIGN_LABELS if d in set(per_repeat.design)]
    models = sorted(per_repeat.model.unique())
    fig, axes = new_figure(11.0, 3.8 * len(designs), nrows=len(designs))
    axes = np.atleast_1d(axes).ravel()
    width = 0.8 / len(models)
    for ax, design in zip(axes, designs):
        sub = per_repeat[per_repeat.design == design]
        for k, model in enumerate(models):
            g = sub[sub.model == model]
            for i, metric in enumerate(metrics):
                vals = g[metric].dropna().to_numpy()
                # Place each model's box side by side inside the column of its metric.
                x = i - 0.4 + width * (k + 0.5)
                if len(vals) > 1:
                    ax.boxplot(vals, positions=[x], widths=width * 0.8, showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor=MODEL_COLORS[model], alpha=0.35, edgecolor=MODEL_COLORS[model]),
                               medianprops=dict(color=MODEL_COLORS[model], linewidth=2),
                               whiskerprops=dict(color=MODEL_COLORS[model]), capprops=dict(color=MODEL_COLORS[model]))
                ax.scatter(np.full(len(vals), x), vals, s=14, color=MODEL_COLORS[model], zorder=3,
                           label=MODEL_LABELS[model] if i == 0 else None)
        ax.set_xticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
        ax.set_ylim(0, 1.02)
        ax.axhline(0.5, color="#898781", linestyle=":", linewidth=1)
        ax.set_ylabel("Score (0-1)")
        n_rep = sub["repeat"].nunique()
        ax.set_title("Held-out test set" if design == "holdout_test" else
                     f"{DESIGN_LABELS[design]} - " + ("one point per CV repeat" if n_rep > 1
                                                       else "single repeat (whole species held out)"))
        ax.legend(loc="lower left", ncol=len(models))
    fig.suptitle("Model comparison on out-of-fold predictions", fontweight="bold")
    add_note(fig, "Dotted line = 0.5 (chance level for balanced accuracy and ROC-AUC). Higher is better for all metrics.")
    return save_figure(fig, path)


def plot_strain_probabilities(table: pd.DataFrame, path: Path) -> Path | None:
    """Mean out-of-fold P(pathogenic) per strain, split by source label."""
    from xanthomonas_ml.plotting import CLASS_COLORS, add_note, new_figure, save_figure
    from xanthomonas_ml.train import MODEL_LABELS

    if table.empty:
        return None
    models = sorted(table.model.unique())
    fig, ax = new_figure(8.0, 4.5)
    # Fixed seed: the horizontal jitter is the same on every run, so figures are reproducible.
    rng = np.random.default_rng(0)
    for k, model in enumerate(models):
        for cls, name in ((0, "non-pathogenic"), (1, "pathogenic")):
            v = table[(table.model == model) & (table.y_true == cls)]["mean_prob_pathogenic"]
            x = k + (-0.18 if cls == 0 else 0.18) + rng.uniform(-0.08, 0.08, len(v))
            ax.scatter(x, v, s=12, color=CLASS_COLORS[name], alpha=0.8,
                       label=f"Source label: {name}" if k == 0 else None)
    ax.axhline(0.5, color="#898781", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Mean out-of-fold P(pathogenic)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Predicted probability per strain (nested stratified CV)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    add_note(fig, "Each dot is one strain averaged over CV repeats. Pathogenic-labelled strains far below 0.5 may be "
                  "mislabelled or atypical\n(as discussed by te Molder et al. 2021) - this is a hypothesis, not evidence.")
    return save_figure(fig, path)


def generate_evaluation_outputs(cfg: Config) -> dict[str, Any]:
    """Recompute summaries from saved predictions and draw all evaluation figures."""
    best = summarise_results(cfg)
    preds = load_predictions(cfg)
    per_repeat = pd.read_csv(results_dir(cfg) / "metrics_per_repeat.csv")
    fig_dir = ensure_dir(cfg.figures_dir)
    figures: list[Path] = []
    for design in sorted(preds.design.unique()):
        short = {"stratified_nested_cv": "stratified_cv", "species_grouped_nested_cv": "species_grouped_cv",
                 "holdout_test": "holdout"}.get(design, design)
        p = plot_confusion_matrices(preds, design, fig_dir / f"eval_confusion_matrices_{short}.png")
        figures += [p] if p else []
        figures += plot_roc_pr(preds, design, fig_dir / f"eval_roc_curves_{short}.png",
                               fig_dir / f"eval_precision_recall_curves_{short}.png")
    p = plot_model_comparison(per_repeat, fig_dir / "eval_model_comparison.png")
    figures += [p] if p else []
    strain_table = per_strain_probabilities(preds)
    ensure_dir(cfg.tables_dir)
    strain_table.to_csv(cfg.tables_dir / "per_strain_oof_probabilities.csv", index=False)
    p = plot_strain_probabilities(strain_table, fig_dir / "eval_per_strain_probabilities.png")
    figures += [p] if p else []
    _register_eval_captions(cfg, [f.name for f in figures])
    log.info("Evaluation figures written to %s", fig_dir.as_posix())
    return {"best": best, "figures": [f.as_posix() for f in figures]}


def _register_eval_captions(cfg: Config, filenames: list[str]) -> None:
    """Store a caption for every evaluation figure (read by the report generator)."""
    from xanthomonas_ml.eda import FigureRegistry

    designs = {"stratified_cv": "nested stratified CV (repeated)",
               "species_grouped_cv": "nested species-grouped CV (whole species held out)",
               "holdout": "the held-out test set"}
    templates = {
        "eval_confusion_matrices_": "Out-of-fold confusion matrices under {d}, as percentages of each true class.",
        "eval_roc_curves_": "ROC curves of out-of-fold predicted probabilities under {d}; the legend gives mean ± sd AUC.",
        "eval_precision_recall_curves_": "Precision-recall curves (pathogenic = positive class) under {d}; the dashed "
                                         "line is the prevalence baseline.",
    }
    reg = FigureRegistry(cfg.figures_dir)
    for name in filenames:
        if name == "eval_model_comparison.png":
            reg.add(name, "Out-of-fold metrics per model. Top: nested stratified CV, one point per repeat. Bottom: "
                          "species-grouped CV. Dotted line: 0.5.")
        elif name == "eval_per_strain_probabilities.png":
            reg.add(name, "Mean out-of-fold probability of being pathogenic for every strain, grouped by source label. "
                          "Strains on the wrong side of 0.5 in all models are candidates for re-examination of their "
                          "label or biology, not proven errors.")
        else:
            for prefix, text in templates.items():
                if name.startswith(prefix):
                    # Strip the prefix and '.png' to recover the design key used in the caption table.
                    key = name[len(prefix):-4]
                    reg.add(name, text.format(d=designs.get(key, key)))
    reg.save()


def best_model_info(cfg: Config) -> dict[str, Any]:
    """Return the saved model ranking (raises if training has not been run)."""
    path = results_dir(cfg) / "best_model.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run 'python -m xanthomonas_ml train' first.")
    return read_json(path)


def headline_table(cfg: Config) -> pd.DataFrame:
    """Compact table: mean ± sd of key pooled-per-repeat metrics per model/design."""
    summary = pd.read_csv(results_dir(cfg) / "metrics_summary.csv")
    sub = summary[summary.level == "pooled_per_repeat"]
    rows = []
    for (model, design), g in sub.groupby(["model", "design"]):
        row = {"model": model, "design": design}
        for metric in METRICS:
            r = g[g.metric == metric].iloc[0]
            if r["n_defined"] == 0:
                row[metric] = "undefined"
            elif r["n_defined"] == 1 or pd.isna(r["std"]):
                row[metric] = f"{r['mean']:.3f}"
            else:
                row[metric] = f"{r['mean']:.3f} ± {r['std']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
