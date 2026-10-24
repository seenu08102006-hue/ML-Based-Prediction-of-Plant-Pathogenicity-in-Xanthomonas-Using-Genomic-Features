# outputs — figures, tables and raw CV results (generated, not versioned)

| Path | Created by | Content |
|---|---|---|
| `figures/eda_*.png` | `eda` | Class counts, species composition, feature counts, missing values, prevalence, genome statistics, similarity heatmap, PCA |
| `figures/eval_*.png` | `train` / `evaluate` | Confusion matrices, ROC and precision–recall curves, model comparison, per-strain probabilities |
| `figures/interp_*.png` | `interpret` | Top domains per model, class-wise prevalence of top domains |
| `figures/captions.json` | all | Caption for every figure (used by the report) |
| `tables/` | `eda`, `evaluate`, `interpret` | Descriptive statistics, per-strain out-of-fold probabilities, `feature_importance_<model>.csv` |
| `results/` | `train` / `evaluate` | `cv_predictions.csv` (every out-of-fold prediction), `cv_folds.csv`, fold importances, `metrics_*.csv`, `best_model.json`, `training_run.json` |
| `predictions/` | `predict` | Prediction CSVs (default location) |
