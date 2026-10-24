# data/external — your own independent data (optional, not versioned)

Place independently produced, labelled Pfam presence/absence tables here to test a
trained model on strains it has never seen:

```bash
python -m xanthomonas_ml evaluate --input data/external/my_strains.csv --label-column pathogenicity --model random_forest
```

Format: one row per strain, an identifier column (default `gca`), a label column
with `pathogenic` / `non-pathogenic`, and Pfam accession columns with 0/1.
Domains must be annotated as in the training data (Prodigal + InterProScan/Pfam);
see `python -m xanthomonas_ml build-features --help`.
