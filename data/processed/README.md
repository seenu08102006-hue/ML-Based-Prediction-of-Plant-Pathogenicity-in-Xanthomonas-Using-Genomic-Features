# data/processed — analysis dataset (generated, not versioned)

Created by `python -m xanthomonas_ml prepare-data`:

| File | Content |
|---|---|
| `features.csv.gz` | Binary strain × Pfam matrix (118 × 2,579 for the published data), index `gca` |
| `labels.csv` | `target` per strain: 1 = pathogenic, 0 = non-pathogenic |
| `sample_metadata.csv` | Strain name, species, host, original label per strain |
| `removed_samples.csv` | Excluded genomes and the reason (e.g. non-*Xanthomonas*, label `unknown`) |
| `removed_features.csv` | Domains constant across all analysed strains (no information) |
| `preparation_metadata.json` | Counts, label mapping, SHA-256 of the source files, settings fingerprint |

The folder is rebuilt automatically when the source files or label settings change;
otherwise it is reused. Force a rebuild with `prepare-data --force`.
