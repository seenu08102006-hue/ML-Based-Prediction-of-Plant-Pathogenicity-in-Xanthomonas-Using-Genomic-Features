"""Exploratory data analysis of the authentic analysis dataset.

All figures are drawn from the prepared data (published strains only). The
univariate statistics computed here use the whole dataset and are therefore
*descriptive*: they are never used to choose model features.

Every figure is registered with a caption in ``outputs/figures/captions.json``,
which the report generator uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from xanthomonas_ml.config import Config
from xanthomonas_ml.feature_engineering import CorrelationFilter, PrevalenceFilter, feature_prevalence_by_class
from xanthomonas_ml.plotting import (
    CLASS_COLORS,
    INK_MUTED,
    SERIES,
    add_note,
    new_figure,
    save_figure,
)
from xanthomonas_ml.preprocessing import PreparedData
from xanthomonas_ml.utils import ensure_dir, get_logger, read_json, write_json

log = get_logger("eda")
CLASS_NAMES = {0: "non-pathogenic", 1: "pathogenic"}
SOURCE_NOTE = "Data: te Molder et al. (2021) BMC Genomics 22:848, published domain matrix."


class FigureRegistry:
    """Collects figure paths and captions and persists them as JSON."""

    def __init__(self, fig_dir: Path):
        self.fig_dir = ensure_dir(fig_dir)
        self.path = fig_dir / "captions.json"
        # Load captions already on disk, so figures written by other steps are kept.
        self.items: dict[str, str] = read_json(self.path) if self.path.is_file() else {}

    def add(self, filename: str, caption: str) -> Path:
        self.items[filename] = caption
        return self.fig_dir / filename

    def save(self) -> None:
        write_json(self.path, self.items)


def plot_class_counts(data: PreparedData, reg: FigureRegistry) -> None:
    excluded = data.removed_samples
    fig, ax = new_figure(6.5, 4.0)
    labels = ["non-pathogenic", "pathogenic"]
    counts = [int((data.y == 0).sum()), int((data.y == 1).sum())]
    bars = ax.bar(labels, counts, color=[CLASS_COLORS[label] for label in labels], width=0.55)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 0.8, f"{n} ({n / sum(counts):.0%})", ha="center", fontsize=10)
    ax.set_ylabel("Number of strains (genomes)")
    ax.set_xlabel("Phenotype label used for modelling")
    ax.set_title("Strains per phenotype class (analysis set)")
    ax.set_ylim(0, max(counts) * 1.18)
    add_note(fig, f"{len(excluded)} genome(s) excluded before analysis (see removed_samples.csv). {SOURCE_NOTE}")
    path = reg.add(
        "eda_class_counts.png",
        f"Number of strains per phenotype class after excluding {len(excluded)} genome(s) without a binary "
        f"Xanthomonas phenotype label. Percentages show class balance (minority fraction "
        f"{min(counts) / sum(counts):.2f}).",
    )
    save_figure(fig, path)


def plot_species_by_class(data: PreparedData, reg: FigureRegistry, tables: Path) -> None:
    species = data.groups.str.replace("Xanthomonas ", "X. ", regex=False)
    table = pd.crosstab(species, data.y.map(CLASS_NAMES)).reindex(columns=["non-pathogenic", "pathogenic"], fill_value=0)
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total")
    table.to_csv(tables / "species_by_class.csv")
    fig, ax = new_figure(7.5, 0.28 * len(table) + 1.6)
    left = np.zeros(len(table))
    for cls in ["non-pathogenic", "pathogenic"]:
        ax.barh(table.index, table[cls], left=left, color=CLASS_COLORS[cls], label=cls, height=0.7,
                edgecolor="#fcfcfb", linewidth=1)
        left += table[cls].to_numpy()
    for i, total in enumerate(table["total"]):
        ax.text(total + 0.4, i, str(total), va="center", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("Number of strains")
    ax.set_ylabel("Species (from metadata)")
    ax.set_title("Class composition per species")
    ax.legend(title="Phenotype label", loc="lower right")
    ax.grid(axis="y", visible=False)
    add_note(fig, "Species is a strong confounder: several species contain only one class.")
    save_figure(fig, reg.add(
        "eda_species_by_class.png",
        "Number of pathogenic and non-pathogenic strains per species. Species with a single class can make the "
        "phenotype partly predictable from phylogeny alone, which motivates the species-grouped CV estimate.",
    ))


def feature_count_stages(data: PreparedData, cfg: Config) -> pd.DataFrame:
    """Feature counts at each filtering stage (full-data, descriptive only)."""
    pp = cfg["preprocessing"]
    prev = PrevalenceFilter(pp["min_prevalence"], pp["max_prevalence"]).fit(data.X)
    after_prev = data.X.loc[:, prev.get_feature_names_out()]
    corr = CorrelationFilter(pp.get("correlation_threshold")).fit(after_prev)
    return pd.DataFrame(
        [
            ("Columns in published matrix", data.info.get("n_features_in_source")),
            ("Variable across analysis strains", data.X.shape[1]),
            (f"Prevalence within [{pp['min_prevalence']}, {pp['max_prevalence']}]", int(prev.support_.sum())),
            (f"... and |r| <= {pp.get('correlation_threshold')}", int(corr.support_.sum())),
        ],
        columns=["stage", "n_features"],
    )


def plot_dimensions(data: PreparedData, cfg: Config, reg: FigureRegistry, tables: Path) -> pd.DataFrame:
    stages = feature_count_stages(data, cfg)
    stages.to_csv(tables / "feature_count_stages.csv", index=False)
    fig, ax = new_figure(7.5, 3.6)
    ax.barh(stages["stage"][::-1], stages["n_features"][::-1], color=SERIES[0], height=0.6)
    for i, n in enumerate(stages["n_features"][::-1]):
        ax.text(n + 30, i, f"{n:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of Pfam domain features")
    ax.set_title(f"Pfam feature counts ({data.X.shape[0]} strains)")
    ax.set_xlim(0, stages["n_features"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    add_note(fig, "The last two filters are shown on the full dataset for orientation only; during model\n"
                  "training they are re-fitted inside every training fold (no information leakage).")
    save_figure(fig, reg.add(
        "eda_dataset_dimensions.png",
        "Number of Pfam domain features at each filtering stage. The prevalence (near-invariance) and correlation "
        "filters mirror te Molder et al. (2021); here they are shown on all strains for orientation, whereas model "
        "training re-fits them inside each training fold.",
    ))
    return stages


def plot_missing_values(data: PreparedData, reg: FigureRegistry, tables: Path) -> pd.DataFrame:
    meta_missing = data.samples.drop(columns=["target"]).isna().sum()
    feature_missing = int(data.X.isna().sum().sum())
    table = pd.concat([meta_missing.rename("n_missing"), pd.Series({"Pfam feature matrix (all cells)": feature_missing},
                                                                   name="n_missing")])
    table.to_csv(tables / "missing_values.csv", header=True)
    fig, ax = new_figure(7.0, 3.8)
    ax.barh(table.index[::-1], table.values[::-1], color=SERIES[0], height=0.6)
    for i, n in enumerate(table.values[::-1]):
        ax.text(n + 0.3, i, str(int(n)), va="center", fontsize=8)
    ax.set_xlabel(f"Missing values (out of {len(data.y)} strains)")
    ax.set_title("Missing values per column (analysis set)")
    ax.set_xlim(0, max(len(data.y) * 0.25, table.max() * 1.2 + 1))
    ax.grid(axis="y", visible=False)
    add_note(fig, "Metadata gaps are 'NA' in the source; they do not affect modelling (only Pfam features are used).")
    save_figure(fig, reg.add(
        "eda_missing_values.png",
        f"Missing values per metadata column and in the feature matrix ({feature_missing} missing feature cells). "
        "Only the Pfam features and the phenotype label enter the models.",
    ))
    return table


def plot_prevalence_distribution(data: PreparedData, cfg: Config, reg: FigureRegistry) -> None:
    prevalence = data.X.mean()
    pp = cfg["preprocessing"]
    fig, ax = new_figure(7.0, 4.0)
    ax.hist(prevalence, bins=40, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.8)
    for x in (pp["min_prevalence"], pp["max_prevalence"]):
        ax.axvline(x, color=SERIES[1], linestyle="--", linewidth=1.5)
    ax.plot([], [], color=SERIES[1], linestyle="--", label=f"Near-invariance bounds ({pp['min_prevalence']}, "
                                                            f"{pp['max_prevalence']})")
    ax.set_xlabel("Prevalence (fraction of strains containing the domain)")
    ax.set_ylabel("Number of Pfam domains")
    ax.set_title("Distribution of Pfam domain prevalence")
    ax.legend(loc="upper center")
    add_note(fig, "Domains present in all or no strains were already removed. Many domains are core (right) or rare (left).")
    save_figure(fig, reg.add(
        "eda_feature_prevalence.png",
        "Histogram of domain prevalence across analysed strains. The bimodal shape reflects a large set of core "
        "domains and many rare (accessory) domains; dashed lines mark the near-invariance bounds of the in-fold filter.",
    ))


def plot_prevalence_by_class(enrichment: pd.DataFrame, reg: FigureRegistry, top_n: int = 10) -> None:
    fig, ax = new_figure(6.5, 6.0)
    sig = enrichment["fisher_q_bh"] < 0.05
    ax.scatter(enrichment.loc[~sig, "prevalence_non_pathogenic"], enrichment.loc[~sig, "prevalence_pathogenic"],
               s=8, color="#c3c2b7", label="q >= 0.05 (BH)")
    for direction, color in (("pathogenic", CLASS_COLORS["pathogenic"]), ("non-pathogenic", CLASS_COLORS["non-pathogenic"])):
        m = sig & (enrichment["more_common_in"] == direction)
        ax.scatter(enrichment.loc[m, "prevalence_non_pathogenic"], enrichment.loc[m, "prevalence_pathogenic"],
                   s=14, color=color, label=f"q < 0.05, more common in {direction}")
    # Domains with identical prevalences sit on the same point; label each point once.
    top = enrichment.head(top_n).reset_index()
    for (x, y), grp in top.groupby(["prevalence_non_pathogenic", "prevalence_pathogenic"], sort=False):
        names = grp["feature"].tolist()
        text = names[0] + (f" (+{len(names) - 1})" if len(names) > 1 else "")
        ax.annotate(text, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#52514e")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Prevalence in non-pathogenic strains")
    ax.set_ylabel("Prevalence in pathogenic strains")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Domain prevalence by phenotype class")
    ax.legend(loc="upper left", fontsize=8)
    add_note(fig, "Univariate Fisher tests on all strains: descriptive associations only (not used for model training,\n"
                  "not corrected for phylogenetic relatedness, not evidence of causation). Labels: 10 smallest p-values;\n"
                  "'(+n)' = n further domains with identical prevalences.")
    save_figure(fig, reg.add(
        "eda_prevalence_by_class.png",
        "Prevalence of each Pfam domain in pathogenic versus non-pathogenic strains; points off the diagonal are "
        "more common in one class. Colour marks Benjamini-Hochberg q < 0.05 from Fisher's exact test. Because "
        "strains are phylogenetically related, these p-values overstate independent evidence.",
    ))


def plot_genome_stats(data: PreparedData, cfg: Config, reg: FigureRegistry) -> None:
    from xanthomonas_ml.data_loader import load_metadata

    meta_path = cfg.raw_dir / cfg["dataset"]["files"]["genome_information"]["filename"]
    meta = load_metadata(meta_path).set_index(cfg["dataset"]["sample_id_column"]).reindex(data.X.index)
    stats = {
        "Genome size (Mb)": pd.to_numeric(meta.get("genome_size"), errors="coerce") / 1e6,
        "Predicted genes (count)": pd.to_numeric(meta.get("gene_count"), errors="coerce"),
        "Variable Pfam domains (count)": data.X.sum(axis=1),
    }
    # Three side-by-side panels, one per statistic; new_figure() alone would give a single axes.
    fig, plt_axes = new_figure(10.0, 3.8)
    fig.clf()
    plt_axes = fig.subplots(1, 3)
    rng = np.random.default_rng(1)
    for ax, (label, values) in zip(plt_axes, stats.items()):
        groups = [values[data.y == c].dropna() for c in (0, 1)]
        ax.boxplot(groups, widths=0.5, showfliers=False, medianprops=dict(color="#0b0b0b"))
        for i, (vals, name) in enumerate(zip(groups, ["non-pathogenic", "pathogenic"])):
            ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(vals)), vals, s=9, color=CLASS_COLORS[name], alpha=0.8)
        ax.set_xticks([1, 2], ["non-pathogenic", "pathogenic"])
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0], fontsize=10)
    fig.suptitle("Genome characteristics by phenotype class", fontweight="bold")
    add_note(fig, "Variable Pfam domains counts only domains that vary across the analysis strains. Sizes/gene counts "
                  "from the published metadata;\nassembly quality varies (most genomes are contig-level drafts).")
    save_figure(fig, reg.add(
        "eda_genome_stats_by_class.png",
        "Genome size, number of predicted genes and number of variable Pfam domains per strain, by class. "
        "Differences may partly reflect species composition and assembly quality.",
    ))


def plot_similarity_heatmap(data: PreparedData, reg: FigureRegistry) -> None:
    X = data.X.to_numpy().astype(bool)
    # pdist returns Jaccard distance; similarity is 1 minus distance.
    sim = 1 - squareform(pdist(X, metric="jaccard"))
    order = leaves_list(linkage(pdist(X, metric="jaccard"), method="average"))
    fig, _ = new_figure(7.5, 7.8)
    fig.clf()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 28])
    ax_bar, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    classes = data.y.to_numpy()[order]
    ax_bar.imshow(classes[None, :], aspect="auto", cmap=_class_cmap(), vmin=0, vmax=1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([0], ["class"])
    ax_bar.grid(False)
    # Start the colour scale at the 1st percentile so differences between typical pairs remain visible.
    im = ax.imshow(sim[np.ix_(order, order)], cmap="Blues", vmin=np.percentile(sim, 1), vmax=1, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel(f"Strains ({len(order)}), ordered by average-linkage clustering")
    ax.set_ylabel("Strains")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Jaccard similarity of domain content")
    fig.suptitle("Strain similarity by Pfam domain content", fontweight="bold")
    ax.legend(handles=[_patch(CLASS_COLORS["non-pathogenic"], "non-pathogenic"),
                       _patch(CLASS_COLORS["pathogenic"], "pathogenic")],
              title="Class bar (top)", loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8)
    save_figure(fig, reg.add(
        "eda_strain_similarity_heatmap.png",
        "Pairwise Jaccard similarity of variable Pfam domain content between strains, ordered by hierarchical "
        "clustering; the top bar shows the phenotype class. Blocks of similar strains largely follow species, "
        "illustrating the relatedness structure that random CV splits do not control for.",
    ))


def _class_cmap():
    from matplotlib.colors import ListedColormap

    return ListedColormap([CLASS_COLORS["non-pathogenic"], CLASS_COLORS["pathogenic"]])


def _patch(color: str, label: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=color, label=label)


def plot_pca(data: PreparedData, cfg: Config, reg: FigureRegistry) -> None:
    pca = PCA(n_components=2, random_state=cfg.seed)
    coords = pca.fit_transform(data.X.to_numpy(dtype=float))
    fig, ax = new_figure(6.5, 5.2)
    for cls, name in CLASS_NAMES.items():
        m = data.y.to_numpy() == cls
        ax.scatter(coords[m, 0], coords[m, 1], s=22, color=CLASS_COLORS[name], label=name, alpha=0.85,
                   edgecolor="#fcfcfb", linewidth=0.6)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} of variance)")
    ax.set_title("PCA of Pfam domain presence/absence")
    ax.legend(title="Phenotype label")
    add_note(fig, "Unsupervised projection of all strains; labels were not used to compute the axes.")
    save_figure(fig, reg.add(
        "eda_pca.png",
        f"Principal component analysis of the binary domain matrix (PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%} of variance), "
        "coloured by phenotype label. Overlap between classes indicates that no single dominant axis of domain "
        "content separates the phenotypes.",
    ))


def run_eda(data: PreparedData, cfg: Config) -> dict[str, Any]:
    """Generate every EDA table and figure; return a summary dictionary."""
    reg = FigureRegistry(cfg.figures_dir)
    tables = ensure_dir(cfg.tables_dir)

    # Computed once here and shared by the enrichment table, the figure and the summary.
    enrichment =feature_prevalence_by_class(data.X, data.y)
    enrichment.to_csv(tables / "feature_prevalence_by_class.csv")

    plot_class_counts(data, reg)
    plot_species_by_class(data, reg, tables)
    stages = plot_dimensions(data, cfg, reg, tables)
    missing = plot_missing_values(data, reg, tables)
    plot_prevalence_distribution(data, cfg, reg)
    plot_prevalence_by_class(enrichment, reg)
    plot_genome_stats(data, cfg, reg)
    plot_similarity_heatmap(data, reg)
    plot_pca(data, cfg, reg)
    reg.save()

    summary = {
        "n_strains": int(data.X.shape[0]),
        "n_features": int(data.X.shape[1]),
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in data.y.value_counts().items()},
        "n_species_groups": int(data.groups.nunique()),
        "feature_count_stages": stages.to_dict(orient="records"),
        "missing_metadata_values": {k: int(v) for k, v in missing.items()},
        "n_domains_q_below_0.05": int((enrichment["fisher_q_bh"] < 0.05).sum()),
        "removed_samples": data.removed_samples.to_dict(orient="records"),
    }
    write_json(tables / "eda_summary.json", summary)
    log.info("EDA: %d figures in %s", len([k for k in reg.items if k.startswith("eda_")]), cfg.figures_dir.as_posix())
    return summary
