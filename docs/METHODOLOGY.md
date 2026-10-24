# Methodology

This document describes what the code does, step by step. The module named in each heading
implements that step.

## 1. Data acquisition (`dataset_download.py`)

The three published CSV files are downloaded from a pinned commit of the authors' repository, streamed to a
temporary `.part` file, checked for truncation (Content-Length) and verified by SHA-256 before being renamed into
place. A verified local copy is never downloaded again. A file that fails verification is renamed
`*.checksum-mismatch` so it cannot be used by accident. Network errors are retried (3 attempts by default) and then
reported with a clear message. `validate-data` works fully offline.

## 2. Loading and validation (`data_loader.py`, `validation.py`)

All cells are read as text, so no type is inferred silently. The `"NA"` strings in the metadata become missing
values. Feature cells are converted to numbers only after validation has located every non-numeric or missing cell.
See `docs/DATASET.md` for the list of checks. Any ERROR stops the analysis; WARNINGs are written to
`reports/data_validation.md`.

## 3. Analysis set (`preprocessing.py`)

1. **Alignment by identifier.** Feature rows and metadata rows are joined on `gca`. Row order is never used. Samples
   present in only one table are excluded and listed.
2. **Genus filter.** Only *Xanthomonas* genomes are kept (this removes one *Xylella* genome).
3. **Labels.** `pathogenic` → 1 and `non-pathogenic` → 0. Every other value is excluded with a recorded reason, never
   re-mapped.
4. **Globally constant domains** (present in all or none of the 118 strains, 1,135 domains) are dropped. This step is
   label-free, and it is equivalent to what the in-fold filter would do in every fold: a column constant in the
   full set is constant in every subset.
5. The result (118 × 2,579, 65 pathogenic / 53 non-pathogenic) and its provenance (source SHA-256, settings
   fingerprint) are saved in `data/processed/`. They are rebuilt only if the sources or the label settings change.

Removed samples and features are stored with their reasons in `removed_samples.csv` and `removed_features.csv`.

## 4. Feature engineering inside the folds (`feature_engineering.py`)

Each model is a scikit-learn `Pipeline`:

```
PrevalenceFilter(min 0.025, max 0.975)  ->  CorrelationFilter(|r| > 0.8)  ->  classifier
```

* **PrevalenceFilter** removes domains whose prevalence in the *training* strains is outside [2.5 %, 97.5 %].
  Near-invariant domains carry almost no information and are unstable in small folds.
* **CorrelationFilter** visits domains in accession order and drops a domain when its absolute Pearson
  correlation with an already-kept domain exceeds 0.8. Domains of the same protein or gene cluster often
  co-occur perfectly; keeping one representative avoids splitting importance among copies. The mapping
  dropped → kept is stored, so interpretation can report which domains a kept domain stands for.

The thresholds are those of te Molder et al. (2021). Unlike the original analysis, which filtered the full matrix once
before CV, **both filters are fitted on the training part of every inner and outer split**. On the full dataset
they give 1,692 domains after the prevalence filter (identical to the paper) and 884 after the correlation
filter. The paper reports 871; the small difference comes from a different rule for choosing which of two
correlated domains to keep. These full-data counts are shown in the EDA for orientation only.

No supervised feature selection is performed outside the models. The class-wise prevalence and Fisher tests
in `feature_prevalence_by_class` are used only for description, never for choosing model inputs.

## 5. Models (`train.py`)

| Model | scikit-learn estimator | Tuned grid (default config) |
|---|---|---|
| CART decision tree | `DecisionTreeClassifier(class_weight="balanced")` | `ccp_alpha` ∈ {0, 0.005, 0.01, 0.02, 0.05}; `max_depth` ∈ {2, 3, 5, None} |
| L1-regularised logistic regression | `LogisticRegression(l1_ratio=1, solver="liblinear", class_weight="balanced")` (`penalty="l1"` on scikit-learn < 1.8) | `C` ∈ {0.03, 0.1, 0.3, 1, 3, 10} |
| Random forest | `RandomForestClassifier(n_estimators=500, class_weight="balanced")` | `max_features` ∈ {sqrt, 0.1, 0.3}; `min_samples_leaf` ∈ {1, 3} |

* `class_weight="balanced"` compensates for the mild class imbalance (45 % / 55 %).
* The **L1 logistic regression is comparable to, but not identical with,** the glmnet Lasso used in the paper.
  The solver, the penalty scaling (`C` = 1/λ with a different loss normalisation) and the tuning grid all
  differ. Its coefficients are sparse: domains with a zero coefficient are not used.
* All random seeds derive from `seed: 42` in the config, so runs are reproducible on the same library versions.

## 6. Validation design (`train.py`)

### Primary: nested repeated stratified CV

* **Outer loop** (performance estimation): `RepeatedStratifiedKFold`, 5 folds × 10 repeats = 50 test folds.
  Every strain is predicted 10 times, each time by a model that never saw it.
* **Inner loop** (hyper-parameter choice): `GridSearchCV` with stratified 5-fold CV on the outer training part,
  optimising balanced accuracy.
* The filters and the model are re-fitted inside every inner and outer training split. **No information from a
  test fold reaches feature filtering, tuning or fitting.**

The paper used a larger design (20 × 5 outer, 4 × 10 inner). A smaller default keeps the runtime at roughly
20 minutes on a laptop; `outer_repeats` can be raised in the config.

### Stricter: nested species-grouped CV

*Xanthomonas* strains of the same species share much of their domain content (see the similarity heatmap). Random CV
therefore puts close relatives of every test strain into the training set. To estimate how the models transfer to
**species absent from training**, a second outer loop uses `StratifiedGroupKFold` (5 folds) with species as
groups. The inner loop is grouped too. The 7 strains without a species assignment form one group, which is
conservative. Because *X. arboricola* alone contains 54 of the 118 strains, the folds are unequal in size and
the estimate is noisy. Groups make further repeats nearly identical, so a single repeat is run.

The group-aware design still does not address host or geographic generalisation. The metadata do not
support such claims.

### Optional hold-out test set

`cv.holdout_fraction > 0` sets aside a stratified test set before any CV or tuning. The saved model is then
fitted without it, and its hold-out metrics are reported separately. The option is off by default: with
118 strains a single split of about 24 strains is much noisier than nested CV.

### Final models

For prediction, each model is tuned by stratified 5-fold CV on all 118 strains and refitted on all of them. Its
training-set (resubstitution) metrics are stored as `resubstitution_metrics_NOT_generalisation`, a name chosen
on purpose. They are never used as performance estimates.

## 7. Metrics (`evaluate.py`)

The positive class is *pathogenic*. Reported metrics are accuracy, balanced accuracy, precision, recall
(sensitivity), specificity, F1, ROC-AUC and average precision (PR-AUC), plus confusion counts.

* A metric that cannot be computed (e.g. ROC-AUC on a fold with one class, or precision without positive
  predictions) is **reported as missing, with the reason**, never as zero.
* Two aggregation levels:
  * **pooled per repeat**: the metric over all out-of-fold predictions of one repeat (118 strains), then the
    mean ± sd over repeats. This is the primary summary.
  * **per outer fold**: mean, sd, median, IQR and the number of folds where the metric was defined.
* The model ranking (`show-best`) uses the mean pooled balanced accuracy of the stratified design. Differences
  smaller than the spread across repeats should not be read as one model being better.

## 8. Interpretation (`interpretability.py`)

For every model:

1. **CV stability.** For each of the 50 outer folds, the fold model's impurity importances (tree, forest) or
   signed coefficients (L1 LR) are recorded. The report gives each domain's *selection frequency* (share of folds
   that used it) and its mean importance.
2. **Held-out permutation importance.** In each outer fold, each of the fold model's 10 most important domains
   is permuted 5 times in the **held-out** fold, and the drop in balanced accuracy is recorded.
3. **Final model.** Importances or coefficients of the refitted model, with the correlated domains each kept
   domain represents.
4. **Class-wise prevalence**, giving the direction of the association.

Positive L1 coefficients push towards *pathogenic*. Impurity importances have no sign; the direction is read from
the class-wise prevalence. The caveats are written next to the tables: correlated domains substitute for one
another, and importance is association, not causation.

## 9. Prediction (`predict.py`)

Input is one row per strain: an identifier column and Pfam columns with 0/1. The model checks the input
against the schema it was trained on. The required columns are the domains its fitted filters kept.

* A missing required column is an error. With `--fill-missing-as-absent` it is set to 0 and reported.
* Extra columns are ignored and counted.
* Invalid values (blank, 2, "maybe" …) are an error that names the cells concerned.
* Duplicate or empty identifiers are an error.

The output has the predicted class, P(pathogenic), P(non-pathogenic), the model name, the number of filled
columns and a "research estimate only" note.

Raw sequences are not accepted. To obtain features for a new genome, call genes (Prodigal), annotate the
proteins with InterProScan (Pfam member database), and convert the TSV output with
`python -m xanthomonas_ml build-features --interproscan genome.tsv --model random_forest --output new.csv`. With
`--model`, every schema domain gets an explicit 0 when InterProScan did not detect it. This is legitimate,
because the whole proteome was scanned. Annotation with a different tool or Pfam release can shift which domains
are detected; the reference data used InterProScan 5.44-79.0.

## 10. Reporting (`reporting.py`)

`report` assembles `reports/project_report.md` from the saved files only: manifest, validation report, processed
metadata, EDA tables, CV results, importance tables and figure captions. A section whose inputs are missing
says "Results unavailable" instead of showing numbers.
