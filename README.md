# ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features

**Machine-learning prediction of the plant-pathogenic phenotype of *Xanthomonas* strains from Pfam protein-domain
presence/absence profiles**: a reproducible, leakage-free Python re-implementation of te Molder et al. (2021),
*BMC Genomics* 22:848.

[![CI](https://github.com/seenu08102006-hue/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features/actions/workflows/ci.yml/badge.svg)](https://github.com/seenu08102006-hue/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features/actions/workflows/ci.yml)

> **Research use only.** Predictions are statistical estimates from ~118 published strains. They are not a
> diagnosis and not proof that a strain is, or is not, pathogenic on any host.

---

## Contents

[Abstract](#abstract) · [Background](#problem-statement-and-scientific-background) · [Dataset](#dataset) ·
[Architecture](#project-architecture) · [Setup](#setup) · [Commands](#commands) · [Workflow](#step-by-step-workflow) ·
[Results](#results) · [Prediction](#predicting-new-strains) · [Outputs](#where-outputs-are-saved) · [Tests](#automated-tests) ·
[Troubleshooting](#common-errors-and-solutions) · [Limitations](#scientific-limitations) · [Licensing](#data-privacy-and-licensing) ·
[References](#references)

## Abstract

*Xanthomonas* contains many important plant pathogens but also strains that live on plants without causing
disease. Using the curated phenotype labels and Pfam domain matrix published by te Molder et al. (2021)
(118 genomes: 65 pathogenic, 53 non-pathogenic), this project trains three interpretable classifiers: a CART
decision tree, an L1-regularised logistic regression and a random forest. Performance is estimated with nested
repeated stratified cross-validation, where every filtering and tuning step is re-fitted inside each training
fold, and with a stricter species-grouped design. The project ranks the domains the models rely on and provides
a validated command-line tool to predict new strains from their domain profiles.

## Problem statement and scientific background

Plant-pathogenic *Xanthomonas* species cause diseases such as bacterial leaf blight of rice, citrus canker and
black rot of crucifers. Non-pathogenic strains, often found in the same niches, are increasingly sequenced.
Telling the two apart from genome data would help risk assessment, but pathogenicity is a complex phenotype.
Type III secretion and its effectors, adhesins, cell-wall-degrading enzymes and LPS all contribute.

te Molder et al. (2021) curated pathogenicity assays from 39 studies and annotated each sequenced genome with
Pfam domains. Random forest and Lasso models reached a median accuracy of about 0.83. Leucine-rich-repeat
domains (associated with the effectors XopL/XopAE) were linked to pathogens, and a helix-turn-helix domain
and cellulose-synthesis domains to non-pathogens.

**Objectives of this project**

1. Acquire and verify the published data with full provenance.
2. Validate sample–label alignment and data quality.
3. Train and compare CART, L1 logistic regression and random forest without information leakage.
4. Estimate performance honestly, including transfer to unseen species.
5. Identify and describe informative domains without causal over-interpretation.
6. Offer a validated prediction interface and run identically on Windows and Ubuntu.

## Dataset

| | |
|---|---|
| Source | Authors' repository <https://gitlab.com/wurssb/xanthomonas-phenotype-prediction>, files `Rcode/Data/domain_matrix.csv`, `genome_information.csv`, `domain_descriptions.csv`, pinned to commit `1820d031` and verified by SHA-256 |
| Publication | te Molder et al. (2021), [doi:10.1186/s12864-021-08093-0](https://doi.org/10.1186/s12864-021-08093-0) |
| Related data | Annotated genomes (RDF/HDT) [doi:10.4121/14546625](https://doi.org/10.4121/14546625), not needed here |
| Samples | 119 genomes → 118 *Xanthomonas* strains with a binary label (65 pathogenic, 53 non-pathogenic); 1 *Xylella fastidiosa* genome labelled `unknown` is excluded |
| Features | 3,714 Pfam domains (binary presence/absence), of which 2,579 vary across the 118 strains |
| Identifier | `gca` (GenBank assembly accession) in both files; labels and features are joined only through it |

"Non-pathogenic" means no symptoms in the reported assays. It is not the same as "non-plant-associated", and
unknown labels are never re-mapped. Full details are in **[docs/DATASET.md](docs/DATASET.md)**.

**Acquisition:** `python -m xanthomonas_ml download-data` downloads the files and verifies their checksums. To
place the files manually instead, follow [data/raw/README.md](data/raw/README.md). The data files are not
committed to this repository (see [Licensing](#data-privacy-and-licensing)).

### Input data format

* `domain_matrix.csv`: column `gca` plus one column per Pfam accession (`PF#####`), values `0`/`1`.
* `genome_information.csv`: column `gca`, label column `pathogenicity` (`pathogenic` / `non-pathogenic`), and metadata.

### Output data format

* Predictions (`predict`): CSV with `sample_id, predicted_class, probability_pathogenic, probability_non_pathogenic,
  model, n_required_features_filled_as_absent, note`.
* Metrics: `outputs/results/metrics_summary.csv` (one row per model × design × metric × aggregation level, with
  mean, sd, median, IQR and the number of folds where the metric was defined).
* Feature importance: `outputs/tables/feature_importance_<model>.csv`.

## Project architecture

```
├── configs/config.yaml          # all settings: paths, data source + checksums, labels, CV, grids
├── src/xanthomonas_ml/
│   ├── cli.py / __main__.py     # command-line interface (python -m xanthomonas_ml)
│   ├── config.py                # YAML loading, path resolution relative to the project root
│   ├── dataset_download.py      # download, SHA-256 verification, manifest
│   ├── data_loader.py           # explicit, type-safe reading of the raw CSVs
│   ├── validation.py            # data-quality checks -> reports/data_validation.md
│   ├── preprocessing.py         # identifier alignment, label encoding, exclusions, caching
│   ├── feature_engineering.py   # in-fold prevalence/correlation filters, descriptive statistics
│   ├── eda.py                   # exploratory figures and tables
│   ├── train.py                 # models, nested CV designs, final models, model bundles
│   ├── evaluate.py              # metrics (undefined -> NaN + reason), summaries, evaluation plots
│   ├── interpretability.py      # CV-stable importances, held-out permutation importance
│   ├── predict.py               # input validation, prediction, InterProScan -> features
│   ├── reporting.py             # Markdown report built only from saved outputs
│   ├── plotting.py              # headless matplotlib style
│   └── utils.py                 # logging, hashing, JSON helpers
├── tests/                       # pytest suite (synthetic fixtures + real-data checks)
├── scripts/                     # setup_windows.ps1, setup_ubuntu.sh, run_pipeline.sh
├── docs/                        # DATASET, METHODOLOGY, SCIENTIFIC_LIMITATIONS, WINDOWS_SETUP, UBUNTU_SETUP
├── data/{raw,processed,external}/   # data (not versioned; README in each)
├── models/  outputs/  reports/      # generated artefacts (not versioned; README in each)
└── .github/workflows/ci.yml     # Ubuntu CI: unit tests + real-data smoke run
```

Pipeline: `download → validate → prepare → EDA → nested CV + final fit → evaluate → interpret → report`.

## Setup

**Requirements:** Python 3.10 or newer (3.12 recommended; developed with 3.12.1, also tested with 3.11 and
older library versions), and Git. The dependencies are numpy, pandas, scikit-learn, scipy, matplotlib, PyYAML,
joblib, requests and pytest (`requirements.txt`).

### Windows (PowerShell)

```powershell
git clone https://github.com/seenu08102006-hue/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features.git
cd ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

Details: [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md).

### Ubuntu

```bash
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip   # only if missing
git clone https://github.com/seenu08102006-hue/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features.git
cd ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

Complete guide, including how to update and troubleshoot: [docs/UBUNTU_SETUP.md](docs/UBUNTU_SETUP.md).

## Commands

All commands work identically on Windows and Ubuntu (activate `.venv` first, or call its Python directly).
Global options: `--config PATH`, `-v/--verbose`. Add `--help` to any command for its options.

| Command | Purpose |
|---|---|
| `python -m xanthomonas_ml --help` | List commands |
| `python -m xanthomonas_ml check-env` | Check Python, packages, plotting backend and paths |
| `python -m xanthomonas_ml download-data [--force]` | Download (only what is missing) and verify the dataset; write `data/raw/MANIFEST.json` |
| `python -m xanthomonas_ml inspect-data` | Show files, checksums, shapes, columns and label values |
| `python -m xanthomonas_ml validate-data [--allow-checksum-mismatch]` | Validate local files without any download |
| `python -m xanthomonas_ml prepare-data [--force]` | Build/reuse the aligned analysis set in `data/processed/` |
| `python -m xanthomonas_ml eda` | Exploratory figures and tables |
| `python -m xanthomonas_ml train [--models ...] [--skip-species-cv] [--outer-repeats N] [--quick]` | Nested CV, final models, saved bundles, evaluation figures |
| `python -m xanthomonas_ml evaluate` | Recompute metrics and evaluation figures from saved CV results |
| `python -m xanthomonas_ml evaluate --input FILE --label-column COL --model NAME` | Evaluate a saved model on an independent labelled feature file |
| `python -m xanthomonas_ml interpret` | Feature-importance tables and figures |
| `python -m xanthomonas_ml show-best` | Model ranking from the last training run |
| `python -m xanthomonas_ml model-info [--model NAME\|best\|PATH]` | Load a saved model and print its metadata |
| `python -m xanthomonas_ml predict --input FILE [--model ...] [--output CSV] [--id-column gca] [--fill-missing-as-absent]` | Predict new strains |
| `python -m xanthomonas_ml build-features --interproscan TSV... --output CSV [--model NAME]` | Convert InterProScan output into a prediction input file |
| `python -m xanthomonas_ml report` | Generate `reports/project_report.md` |
| `python -m xanthomonas_ml run-all [--quick] [--force]` | download → validate → prepare → eda → train → interpret → report |
| `python -m xanthomonas_ml test [pytest args]` | Run the automated tests |

`--quick` (2 CV repeats, 100 trees) is for smoke tests only; its numbers must not be reported. Exit codes:
0 success, 1 unexpected error, 2 usage error, 3 data/validation error, 4 missing prerequisite.

## Step-by-step workflow

```bash
python -m xanthomonas_ml download-data
python -m xanthomonas_ml validate-data     # -> reports/data_validation.md
python -m xanthomonas_ml prepare-data
python -m xanthomonas_ml eda               # -> outputs/figures/eda_*.png
python -m xanthomonas_ml train             # ~20-30 min; -> models/, outputs/results/, outputs/figures/eval_*.png
python -m xanthomonas_ml interpret         # -> outputs/tables/feature_importance_*.csv
python -m xanthomonas_ml report            # -> reports/project_report.md
python -m xanthomonas_ml show-best
```

### How the models are evaluated

* **Primary:** nested repeated stratified CV (5 folds × 10 repeats outer, 5-fold grid search inner). The
  prevalence filter (2.5–97.5 %) and correlation filter (|r| > 0.8) are fitted inside every training fold, so
  there is no leakage.
* **Stricter:** nested species-grouped CV, which holds out whole species and estimates transfer to unseen species.
* Metrics: accuracy, balanced accuracy, precision, recall, specificity, F1, ROC-AUC and PR-AUC, with mean ± sd
  over repeats. Undefined metrics are reported as missing, never as 0. Training-set scores are never presented
  as performance.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for all details.

### How to interpret the results

* Compare models on the **stratified nested CV** summary. A difference smaller than the sd across repeats is not
  meaningful.
* The **gap to the species-grouped CV** shows how much performance depends on close relatives being in training.
* In the importance tables, prefer domains with a **high selection frequency** across folds and a positive
  held-out permutation drop. Check `correlated_domains_represented`: those domains are interchangeable with the
  listed one.
* An importance is an **association** in this dataset, not a mechanism or a validated biomarker.

## Results

These results come from one full run of `python -m xanthomonas_ml train` on the authentic data (2026-10-24, Windows 11,
Python 3.12.1, scikit-learn 1.9.1, seed 42, outer 10 × 5-fold, inner 5-fold). They are **out-of-fold** estimates:
every strain was predicted by models that never saw it. Re-running the pipeline regenerates them in
`reports/project_report.md`, and small numerical differences across library versions are possible.

| Model | Design | Balanced accuracy | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|
| Decision tree (CART) | nested stratified CV | 0.757 ± 0.037 | 0.753 ± 0.035 | 0.722 ± 0.036 | 0.792 ± 0.065 | 0.779 ± 0.041 |
| L1 logistic regression | nested stratified CV | 0.838 ± 0.028 | 0.834 ± 0.028 | 0.802 ± 0.034 | 0.874 ± 0.031 | 0.888 ± 0.019 |
| Random forest | nested stratified CV | 0.834 ± 0.022 | 0.826 ± 0.022 | 0.757 ± 0.032 | 0.911 ± 0.022 | 0.883 ± 0.020 |
| Decision tree (CART) | species-grouped CV | 0.694 | 0.703 | 0.785 | 0.604 | 0.675 |
| L1 logistic regression | species-grouped CV | 0.752 | 0.763 | 0.862 | 0.642 | 0.786 |
| Random forest | species-grouped CV | 0.774 | 0.763 | 0.662 | 0.887 | 0.832 |

(mean ± sd over 10 CV repeats; the species-grouped design has a single repeat, so no sd.)

* L1 logistic regression and random forest perform almost identically. The 0.003 difference in balanced accuracy is
  far below the spread across repeats. Both outperform the single decision tree. The published medians (CART
  0.750, Lasso 0.826, RF 0.833 accuracy) are reproduced closely.
* When **whole species are held out**, balanced accuracy drops by roughly 0.06–0.09. Part of the apparent
  performance comes from close relatives being present in training.
* The most consistently used domains (selection frequency across the 50 outer folds) agree with the publication.
  **PF13855** (leucine-rich repeat) is more common in pathogens and ranks first for the tree and the forest.
  **PF13579** (glycosyltransferase family 4-like) is the top L1 coefficient. **PF12840** (helix-turn-helix) is
  associated with non-pathogens in all three models. These are associations, not demonstrated mechanisms.
* 14 strains (9 labelled pathogenic, 5 non-pathogenic) are misclassified on average by all three models
  (`outputs/tables/per_strain_oof_probabilities.csv`). They are candidates for re-examining labels or biology,
  not proven errors.
* Training-set accuracy of the final models is 0.98–0.99. That reflects memorisation, not generalisation, and is
  not used as evidence.

## Predicting new strains

The model needs **Pfam domain presence/absence**. It cannot read DNA sequences. To get features for a new genome:

1. Call genes (e.g. Prodigal) and annotate the proteins with InterProScan, including the Pfam member database.
   The training data used InterProScan 5.44-79.0 via SAPP; other versions may shift domain calls.
2. Convert the TSV output, filling all domains the model knows (0 = not detected):

   ```bash
   python -m xanthomonas_ml build-features --interproscan genomeA.tsv genomeB.tsv --model random_forest --output new_strains.csv
   ```
3. Predict:

   ```bash
   python -m xanthomonas_ml predict --input new_strains.csv --model random_forest --output outputs/predictions/new_strains.csv
   ```

Input rules: one row per strain; identifier column `gca` (or `--id-column NAME`); Pfam columns with `0`/`1`
(`true`/`false` accepted). Missing required domains are an error unless `--fill-missing-as-absent` is given.
Extra columns are ignored. Invalid values and duplicate IDs are errors that name the problem. Probabilities are
**not calibrated field risks** (see limitations).

## Where outputs are saved

| Location | Content |
|---|---|
| `data/raw/` | Downloaded files + `MANIFEST.json` |
| `data/processed/` | Aligned analysis set, removed samples/features, provenance |
| `reports/data_validation.md` | Data-quality report |
| `outputs/figures/` | `eda_*`, `eval_*`, `interp_*` PNG figures + `captions.json` |
| `outputs/tables/` | Descriptive tables, per-strain out-of-fold probabilities, feature importances |
| `outputs/results/` | Every out-of-fold prediction, fold details, metric summaries, `best_model.json` |
| `models/` | `<model>.joblib` bundles + `<model>.json` metadata |
| `outputs/predictions/` | Prediction CSVs |
| `reports/project_report.md` | Full report generated from the files above |

## Automated tests

```bash
python -m xanthomonas_ml test          # or: python -m pytest
python -m pytest -m realdata           # checks on the authentic data (skipped if not downloaded)
```

The suite has 102 tests covering file discovery, loading, identifier and label validation, duplicate and missing
value detection, alignment, the filters (including a leakage check), training of each classifier, CV split
properties, reproducibility, metrics including undefined cases, model save/load, prediction input validation and
output format, missing files, CLI help, exit codes and an offline end-to-end run with network access blocked.
Unit tests use **clearly labelled synthetic fixtures** (`tests/conftest.py`) that never touch `data/`. GitHub
Actions runs them on Ubuntu with Python 3.10 and 3.12, plus a real-data smoke run of the whole pipeline.

## Common errors and solutions

| Error | Solution |
|---|---|
| `Dataset verification failed: missing required file` | `python -m xanthomonas_ml download-data`, or place the files manually ([data/raw/README.md](data/raw/README.md)) |
| `SHA-256 ... does not match the pinned published version` | The file was edited (e.g. re-saved in Excel) or is a different version; re-download it |
| `Could not download ...` | Network/proxy problem; place the files manually |
| `... not found. Run 'python -m xanthomonas_ml train' first.` | Train before `evaluate`, `interpret`, `show-best` or `predict` |
| `feature(s) required by the model are missing` | Supply the listed Pfam columns, use `build-features --model`, or deliberately add `--fill-missing-as-absent` |
| `not valid presence/absence values` | Use only 0/1 in Pfam columns |
| `Model was saved with scikit-learn X` | Retrain in the current environment |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Ubuntu `externally-managed-environment` | Activate `.venv` before using pip |

## Scientific limitations

The dataset is small and dominated by *X. arboricola*. Labels come from heterogeneous assays, where
"non-pathogenic" means no symptoms observed. Species are confounded with phenotype, and related strains are not
independent. Presence/absence ignores allelic and regulatory variation. No external validation set exists.
Importance is association, not causation. Read **[docs/SCIENTIFIC_LIMITATIONS.md](docs/SCIENTIFIC_LIMITATIONS.md)**.

## Data privacy and licensing

* The data concern bacterial genomes and published phenotypes. There is no personal data.
* When checked (2026-10-24), the source GitLab repository declared no licence. The data files are therefore
  **not redistributed** here; the pipeline downloads them from the original source. Cite the publication and the
  repository when using them.
* No credentials are needed or stored. Downloads use public HTTPS URLs.
* No licence has been chosen yet for this project's own code. Add a `LICENSE` file (e.g. MIT) if you want
  others to reuse it.

## References

1. te Molder D, Poncheewin W, Schaap PJ, Koehorst JJ. Machine learning approaches to predict the plant-associated
   phenotype of *Xanthomonas* strains. *BMC Genomics*. 2021;22:848. https://doi.org/10.1186/s12864-021-08093-0
2. te Molder D. Xanthomonas phenotype prediction (code and data). https://gitlab.com/wurssb/xanthomonas-phenotype-prediction
3. Koehorst JJ, Poncheewin W, Schaap PJ, te Molder D. Annotated genomes used in the publication *Machine learning
   approaches to predict the plant-associated phenotype of Xanthomonas strains*. https://doi.org/10.4121/14546625
4. Pedregosa F, et al. Scikit-learn: Machine learning in Python. *JMLR*. 2011;12:2825–2830.
5. Mistry J, et al. Pfam: The protein families database in 2021. *Nucleic Acids Research*. 2021;49:D412–D419.
6. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91.

Citation metadata for this project: [CITATION.cff](CITATION.cff).
