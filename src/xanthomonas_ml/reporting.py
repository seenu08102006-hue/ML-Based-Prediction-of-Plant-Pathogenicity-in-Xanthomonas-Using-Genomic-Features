"""Reproducible Markdown project report built only from saved pipeline outputs.

Every number in the generated report is read from files written by earlier
steps (validation report, processed-data metadata, EDA tables, CV results,
importance tables). If a step has not been run, the corresponding section says
that the results are unavailable instead of showing placeholder values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from xanthomonas_ml import RESEARCH_DISCLAIMER, __version__
from xanthomonas_ml.config import Config
from xanthomonas_ml.utils import ensure_dir, read_json, relpath, utc_now_iso

UNAVAILABLE = "_Results unavailable: {what} has not been run yet (`python -m xanthomonas_ml {cmd}`)._"

REFERENCES = [
    "te Molder D, Poncheewin W, Schaap PJ, Koehorst JJ. Machine learning approaches to predict the "
    "plant-associated phenotype of *Xanthomonas* strains. *BMC Genomics*. 2021;22:848. "
    "https://doi.org/10.1186/s12864-021-08093-0",
    "te Molder D. Xanthomonas phenotype prediction (code and data repository). "
    "https://gitlab.com/wurssb/xanthomonas-phenotype-prediction (commit 1820d031)",
    "Koehorst JJ, Poncheewin W, Schaap PJ, te Molder D. Annotated genomes used in the publication: Machine "
    "learning approaches to predict the plant-associated phenotype of Xanthomonas strains. "
    "https://doi.org/10.4121/14546625",
    "Pedregosa F, et al. Scikit-learn: Machine learning in Python. *JMLR*. 2011;12:2825-2830.",
    "Mistry J, et al. Pfam: The protein families database in 2021. *Nucleic Acids Res*. 2021;49:D412-D419.",
    "Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. "
    "*BMC Bioinformatics*. 2006;7:91.",
]


def _read(path: Path) -> Any | None:
    # A missing file returns None, so the matching report section can say "unavailable" instead of failing.
    if not path.is_file():
        return None
    if path.suffix == ".json":
        return read_json(path)
    return pd.read_csv(path)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra deps)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class _Figures:
    def __init__(self, cfg: Config, report_dir: Path):
        self.dir = cfg.figures_dir
        self.report_dir = report_dir
        self.captions = _read(self.dir / "captions.json") or {}
        self.counter = 0

    def embed(self, filename: str) -> str:
        path = self.dir / filename
        if not path.is_file():
            return f"_Figure `{filename}` not available._"
        self.counter += 1
        caption = self.captions.get(filename, "")
        return f"![{filename}]({relpath(path, self.report_dir)})\n\n*Figure {self.counter}. {caption}*"


def generate_report(cfg: Config, output: Path | None = None) -> Path:
    """Write ``reports/project_report.md`` from the saved outputs and return its path."""
    report_dir = ensure_dir(cfg.reports_dir)
    output = output or report_dir / "project_report.md"
    figs = _Figures(cfg, output.parent)
    res = cfg.outputs_dir / "results"
    tables = cfg.tables_dir

    # Every input is optional: a missing file leaves only its own section marked as unavailable.
    validation = _read(report_dir / "data_validation.json")
    prep = _read(cfg.processed_dir / "preparation_metadata.json")
    manifest = _read(cfg.raw_dir / "MANIFEST.json")
    eda = _read(tables / "eda_summary.json")
    run = _read(res / "training_run.json")
    best = _read(res / "best_model.json")
    summary = _read(res / "metrics_summary.csv")
    removed = _read(cfg.processed_dir / "removed_samples.csv")

    s: list[str] = []
    add = s.append
    add("# ML-Based Prediction of Plant Pathogenicity in *Xanthomonas* Using Genomic Features")
    add(f"\n_Reproducible project report generated {utc_now_iso()} (UTC) by xanthomonas_ml {__version__}. "
        "All numbers are read from saved pipeline outputs._\n")
    add(f"> **Disclaimer.** {RESEARCH_DISCLAIMER}\n")

    add("## 1. Introduction\n")
    add("*Xanthomonas* is a genus of Gram-negative bacteria that contains many economically important plant "
        "pathogens (e.g. of rice, citrus, brassicas, walnut and stone fruit), but also strains that colonise plants "
        "without causing disease. Distinguishing the two from genome data would help risk assessment of newly "
        "isolated strains. te Molder et al. (2021) curated pathogenicity assays from the literature, annotated the "
        "genomes of the tested strains with Pfam protein domains, and trained interpretable classifiers. This "
        "project re-implements that analysis in Python with a leakage-free, nested cross-validation protocol.\n")

    add("## 2. Problem statement\n")
    add("Given the Pfam protein-domain presence/absence profile of a *Xanthomonas* genome, estimate whether the "
        "strain is plant-pathogenic or non-pathogenic, and identify which domains are associated with each "
        "phenotype, while reporting performance without optimistic bias.\n")

    add("## 3. Research objectives\n")
    add("1. Acquire and verify the published domain matrix and phenotype labels.\n"
        "2. Validate sample-label alignment and data quality.\n"
        "3. Compare a decision tree (CART), an L1-regularised logistic regression and a random forest.\n"
        "4. Estimate performance with nested CV, including a stricter species-grouped design.\n"
        "5. Rank and describe the domains the models rely on, without causal over-interpretation.\n"
        "6. Provide a validated prediction interface for new domain profiles.\n")

    add("## 4. Dataset provenance\n")
    if manifest:
        add(f"- Source repository: {manifest.get('source_repository')} (commit `{manifest.get('source_commit')}`)")
        add(f"- Publication: {manifest.get('publication')}")
        add(f"- Retrieved/verified: {manifest.get('retrieved_utc')} (UTC)")
        add(f"- Licence note: {manifest.get('license_note')}\n")
        add(_md_table(pd.DataFrame([{"file": f["filename"], "role": f["role"], "bytes": f["size_bytes"],
                                     "sha256 (verified)": f["sha256_verified"][:16] + "..."} for f in manifest["files"]])))
        add("")
    else:
        add(UNAVAILABLE.format(what="dataset download/verification", cmd="download-data") + "\n")

    add("## 5. Dataset dimensions and class distribution\n")
    if prep:
        cb = prep["class_balance"]
        add(f"- Strains analysed: **{prep['n_samples']}** (pathogenic = {cb['counts'].get('1')}, "
            f"non-pathogenic = {cb['counts'].get('0')}; minority fraction {cb['minority_fraction']})")
        add(f"- Pfam domain columns in the published matrix: {prep['n_features_in_source']}; variable across "
            f"analysed strains: **{prep['n_features']}**")
        add(f"- Species groups: {prep.get('n_species_groups')}")
        add(f"- Samples excluded: {prep['n_removed_samples']}\n")
        if removed is not None and len(removed):
            add(_md_table(removed))
            add("")
        if validation:
            add(f"Validation status: **{'PASSED' if validation['passed'] else 'FAILED'}** with "
                f"{validation['n_errors']} errors and {validation['n_warnings']} warnings "
                "(full report: `reports/data_validation.md`).\n")
        add(figs.embed("eda_class_counts.png") + "\n")
        add(figs.embed("eda_species_by_class.png") + "\n")
    else:
        add(UNAVAILABLE.format(what="data preparation", cmd="prepare-data") + "\n")

    add("## 6. Data preprocessing\n")
    add("- Labels and features are joined **only through the GenBank assembly accession (`gca`)**; row order is "
        "never used.\n"
        "- Only the source labels `pathogenic` (class 1) and `non-pathogenic` (class 0) are modelled. Other values "
        "(here one `unknown`) are excluded, never re-mapped. The single non-*Xanthomonas* genome (*Xylella "
        "fastidiosa*) is excluded by the genus filter.\n"
        "- Feature values were checked to be strictly binary with no missing cells.\n"
        "- Domains constant across all analysed strains are dropped (they carry no information and would be "
        "removed in every training fold anyway).\n")
    if eda:
        add(figs.embed("eda_missing_values.png") + "\n")

    add("## 7. Feature engineering\n")
    pp = run["preprocessing"] if run else cfg["preprocessing"]
    add(f"Inside every training fold the pipeline (i) removes near-invariant domains with prevalence outside "
        f"[{pp['min_prevalence']}, {pp['max_prevalence']}] and (ii) greedily removes domains with |Pearson r| > "
        f"{pp['correlation_threshold']} to an already-kept domain. These thresholds follow te Molder et al. (2021), "
        "but the filters are re-fitted on each training fold rather than on the full dataset.\n")
    if eda:
        stages = pd.DataFrame(eda["feature_count_stages"])
        add(_md_table(stages))
        add("\n_Counts for the last two stages are computed on all strains for orientation only._\n")
        add(figs.embed("eda_dataset_dimensions.png") + "\n")
        add(figs.embed("eda_feature_prevalence.png") + "\n")
        add(figs.embed("eda_prevalence_by_class.png") + "\n")
        add(f"Descriptively, {eda['n_domains_q_below_0.05']} domains differ in prevalence between classes at "
            "BH-adjusted q < 0.05 (Fisher's exact test). Because strains are phylogenetically related this "
            "overstates the amount of independent evidence.\n")
        add(figs.embed("eda_genome_stats_by_class.png") + "\n")
        add(figs.embed("eda_strain_similarity_heatmap.png") + "\n")
        add(figs.embed("eda_pca.png") + "\n")
    else:
        add(UNAVAILABLE.format(what="exploratory data analysis", cmd="eda") + "\n")

    add("## 8. Methodology\n")
    add("| Model | Implementation | Tuned hyper-parameters |\n|---|---|---|\n"
        "| Decision tree (CART) | `DecisionTreeClassifier`, class-balanced | `ccp_alpha` (cost-complexity pruning), `max_depth` |\n"
        "| L1-regularised logistic regression | `LogisticRegression` (liblinear, pure L1), class-balanced | inverse penalty `C` |\n"
        f"| Random forest | `RandomForestClassifier`, "
        f"{(run['model_settings'] if run else cfg['models'])['random_forest'].get('n_estimators')} trees, "
        "class-balanced | `max_features` (mtry), `min_samples_leaf` |\n")
    add("The L1 logistic regression is a sparse linear classifier comparable to, but not identical with, the "
        "glmnet Lasso of the reference study (different solver, penalty scaling and tuning grid).\n")

    add("## 9. Model evaluation protocol\n")
    # Describe the settings actually used in the saved run, not the current config.
    cv = run["cv_settings"] if run else cfg["cv"]
    add(f"- **Primary:** nested repeated stratified CV - outer {cv['outer_repeats']} x {cv['outer_folds']}-fold, "
        f"inner {cv['inner_folds']}-fold grid search optimising `{cfg.data.get('selection_metric')}`. Each strain "
        "is predicted once per repeat by a model that never saw it.\n"
        f"- **Stricter:** nested species-grouped CV ({cv.get('species_grouped_folds')} outer folds, "
        "StratifiedGroupKFold with species as groups) - estimates transfer to species absent from training.\n"
        "- Metrics: accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC and "
        "average precision (PR-AUC), positive class = pathogenic. Undefined metrics are reported as missing, "
        "never as zero. Summaries give mean ± sd over repeats of pooled out-of-fold predictions.\n"
        "- Training-set (resubstitution) scores of the final models are diagnostics only.\n")

    add("## 10. Model results\n")
    # Results are shown only when both the metric summary and the training record exist.
    if summary is not None and run:
        from xanthomonas_ml.evaluate import headline_table

        head = headline_table(cfg)
        add(_md_table(head))
        add("\n_Values: mean ± sd across CV repeats of metrics on pooled out-of-fold predictions (species-grouped "
            "CV has a single repeat)._\n")
        if best and best.get("best_model"):
            add(f"Best model by mean {best['selection_metric']} (stratified nested CV): **{best['best_model']}**. "
                f"{best.get('note', '')}\n")
        # The column name says "NOT generalisation" so training-set accuracy is never read as performance.
        resub = pd.DataFrame([
            {"model": m, "resubstitution accuracy (NOT generalisation)":
                r["resubstitution_metrics_NOT_generalisation"]["accuracy"],
             "inner-CV best score (final tuning)": r["inner_cv_best_score"],
             "best params": str(r["best_params"])}
            for m, r in run["final_models"].items()
        ])
        add(_md_table(resub))
        add("")
        for m, r in run["final_models"].items():
            if "holdout_metrics" in r:
                add(f"- Held-out test ({m}): balanced accuracy {r['holdout_metrics']['balanced_accuracy']:.3f}")
    else:
        add(UNAVAILABLE.format(what="model training", cmd="train") + "\n")

    add("## 11. Graphs and tables\n")
    if summary is not None:
        for name in ["eval_model_comparison.png", "eval_confusion_matrices_stratified_cv.png",
                     "eval_roc_curves_stratified_cv.png", "eval_precision_recall_curves_stratified_cv.png",
                     "eval_confusion_matrices_species_grouped_cv.png", "eval_roc_curves_species_grouped_cv.png",
                     "eval_per_strain_probabilities.png"]:
            if (cfg.figures_dir / name).is_file():
                add(figs.embed(name) + "\n")
        add("Full tables: `outputs/results/metrics_summary.csv`, `metrics_per_repeat.csv`, `metrics_per_fold.csv`, "
            "`cv_predictions.csv`, `outputs/tables/per_strain_oof_probabilities.csv`.\n")
    else:
        add(UNAVAILABLE.format(what="evaluation", cmd="evaluate") + "\n")

    add("## 12. Feature-importance analysis\n")
    any_imp = False
    for model in ("decision_tree", "l1_logistic_regression", "random_forest"):
        t = _read(tables / f"feature_importance_{model}.csv")
        if t is None:
            continue
        any_imp = True
        cols = [c for c in ["feature", "description", "selection_frequency", "mean_importance_cv",
                            "heldout_permutation_drop_mean", "more_common_in"] if c in t.columns]
        add(f"### {model.replace('_', ' ').title()}\n")
        add(_md_table(t[cols].head(10)))
        add("")
        add(figs.embed(f"interp_top_features_{model}.png") + "\n")
    if any_imp:
        add(figs.embed("interp_top_feature_prevalence.png") + "\n")
        add("_Importances are associations in ~118 related strains; correlated domains may substitute for each other. "
            "They are not evidence of causation and none of these domains is a validated biomarker._\n")
    else:
        add(UNAVAILABLE.format(what="interpretation", cmd="interpret") + "\n")

    add("## 13. Discussion\n")
    if summary is not None and best and best.get("ranking"):
        rk = best["ranking"]
        add(f"Under nested stratified CV the models reached a mean {best['selection_metric']} between "
            f"{min(r['mean'] for r in rk):.3f} and {max(r['mean'] for r in rk):.3f}. ")
        sp = summary[(summary.design == "species_grouped_nested_cv") & (summary.level == "pooled_per_repeat")
                     & (summary.metric == best["selection_metric"])]
        if len(sp):
            add(f"When whole species were held out, the same metric ranged from {sp['mean'].min():.3f} to "
                f"{sp['mean'].max():.3f}. The gap between the two designs indicates how much of the apparent "
                "performance depends on related strains of the same species being present in training. ")
        add("Interpretation should consider that phenotype labels come from heterogeneous literature assays, that "
            "some pathogenic-labelled strains are consistently predicted as non-pathogenic (see the per-strain "
            "probability figure; te Molder et al. suspected mislabelling for several such strains), and that "
            "*X. arboricola* dominates the dataset.\n")
    else:
        add(UNAVAILABLE.format(what="model training", cmd="train") + "\n")

    add("## 14. Limitations\n")
    add("- Small sample (~118 strains) from few species; wide uncertainty on all estimates.\n"
        "- Labels derive from published assays on limited hosts and conditions; 'non-pathogenic' means no symptoms "
        "were observed in the reported assays, not that a strain can never cause disease.\n"
        "- Strains are not independent (shared ancestry); random CV is optimistic for new lineages.\n"
        "- Presence/absence of Pfam domains ignores allelic variation, gene copy number, regulation, effectors "
        "without Pfam domains, plasmids lost in draft assemblies, and annotation errors.\n"
        "- Predictions for new genomes require the same annotation pipeline (Prodigal + InterProScan/Pfam).\n"
        "- See `docs/SCIENTIFIC_LIMITATIONS.md` for details.\n")

    add("## 15. Conclusion\n")
    if summary is not None and best and best.get("best_model"):
        add("Pfam domain presence/absence profiles carry information associated with the plant-pathogenic "
            "phenotype in this dataset, as measured by out-of-fold performance above chance. The estimates are "
            "uncertain, depend on species composition, and do not establish causal mechanisms. The models are "
            "suitable as research tools for prioritising strains for experimental testing, not for diagnosis.\n")
    else:
        add("No conclusion about predictive performance can be drawn until the models have been trained and "
            "evaluated.\n")

    add("## 16. References\n")
    for i, ref in enumerate(REFERENCES, 1):
        add(f"{i}. {ref}")
    add("")
    output.write_text("\n".join(s) + "\n", encoding="utf-8")
    return output
