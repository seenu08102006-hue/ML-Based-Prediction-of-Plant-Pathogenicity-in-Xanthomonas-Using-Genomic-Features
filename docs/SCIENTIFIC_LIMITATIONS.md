# Scientific limitations

Read this before interpreting any result of this project.

## Data

* **Small sample.** 118 strains, of which 54 are *X. arboricola*. Every performance estimate has wide uncertainty.
  The spread across CV repeats is reported for this reason, and the spread across *datasets* would be larger
  still.
* **Label quality.** Labels come from heterogeneous published assays (different hosts, inoculation methods
  and conditions). "Non-pathogenic" means *no symptoms in the reported assays*: a strain may be pathogenic on an
  untested host or under other conditions. te Molder et al. (2021) suspected that several strains labelled
  pathogenic are mislabelled; such strains appear as confidently misclassified in
  `outputs/tables/per_strain_oof_probabilities.csv`. That is a hypothesis to test experimentally, not evidence.
* **Selection bias.** Only strains that were both assay-tested and sequenced are included, and how common
  non-pathogens are in nature is unknown. Class proportions (55 % pathogenic) do not reflect any real population,
  so predicted probabilities are **not calibrated risks** for field isolates.
* **Confounding by phylogeny.** Several species contain only one class (e.g. all *X. oryzae*, *X. translucens* and
  *X. vasicola* strains are pathogenic). A model can partly learn "which species is this" instead of "what makes a
  strain pathogenic". The species-grouped CV quantifies this but cannot remove it.
* **Non-independent samples.** Related strains, and two strains with identical profiles, make random CV optimistic.

## Features

* Pfam presence/absence ignores allelic variation, point mutations, gene copy number, regulation, expression,
  effectors without a Pfam domain, and genes on plasmids that draft assemblies (75 of 118 are contig-level) may miss.
* Annotation depends on the gene caller and the Pfam/InterProScan release. Profiles produced with another
  pipeline may differ systematically from the training data.
* Correlated domains are interchangeable for the models. A domain absent from the top list may matter just as
  much as a correlated domain that is on it.

## Models and evaluation

* Hyper-parameter grids are deliberately small. A different grid or seed can change the selected domains.
* Impurity importance is biased towards features with more split opportunities, and permutation importance on
  ~24 held-out strains is noisy.
* The L1 logistic regression is not the glmnet Lasso of the paper, so coefficients are not directly comparable.
* No external validation set exists. All estimates come from resampling within one dataset. Generalisation to new
  species, hosts, regions or sequencing technologies has **not** been demonstrated.

## Interpretation rules used throughout this project

1. Feature importance shows **statistical association** in this dataset, **not causation**.
2. No domain identified here is a **validated biomarker** of pathogenicity.
3. Training-set accuracy is never used as evidence of generalisation.
4. A prediction for a new strain is a **research estimate** to help prioritise experiments, not a diagnosis.
