"""Machine-learning prediction of the plant-pathogenic phenotype of *Xanthomonas*
strains from Pfam protein-domain presence/absence profiles.

The package re-implements, in Python/scikit-learn, the analysis design of
te Molder et al. (2021) "Machine learning approaches to predict the
plant-associated phenotype of Xanthomonas strains", BMC Genomics 22:848,
https://doi.org/10.1186/s12864-021-08093-0, with leakage-free nested
cross-validation. Predictions are research estimates, not diagnoses.
"""

# Written into every saved model and shown in the CLI and the report.
__version__ = "0.1.0"

# Shown on every prediction run and in the report; the wording is deliberately cautious.
RESEARCH_DISCLAIMER = (
    "Predictions are statistical research estimates derived from Pfam domain "
    "presence/absence associations in ~118 published strains. They are NOT a "
    "diagnosis and NOT proof that a strain is (or is not) pathogenic on any host."
)
