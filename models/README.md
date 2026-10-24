# models — trained model bundles (generated, not versioned)

`python -m xanthomonas_ml train` writes one bundle per model:

* `<model>.joblib` — fitted scikit-learn pipeline (prevalence filter → correlation filter → classifier) plus metadata
  (feature schema, required features, label mapping, tuned hyper-parameters, library versions, data checksums).
* `<model>.json` — human-readable copy of the metadata (no pipeline).

Models: `decision_tree`, `l1_logistic_regression`, `random_forest`. Inspect one with
`python -m xanthomonas_ml model-info --model random_forest`.

Security: joblib files are pickles and can execute code when loaded. Only load model files you created yourself.
Bundles should be loaded with the same scikit-learn version that created them (a warning is shown otherwise).
