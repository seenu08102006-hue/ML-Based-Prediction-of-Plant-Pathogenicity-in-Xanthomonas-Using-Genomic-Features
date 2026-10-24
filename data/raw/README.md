# data/raw — authentic published input files (not versioned)

The pipeline expects these files from te Molder et al. (2021), pinned to commit
`1820d031e38cc9fffee8c7fc4526890a5be5ff42` of
<https://gitlab.com/wurssb/xanthomonas-phenotype-prediction> (`Rcode/Data/`):

| File | Required | Role | SHA-256 |
|---|---|---|---|
| `domain_matrix.csv` | yes | Binary Pfam presence/absence matrix; column `gca` = GenBank assembly accession, other columns = Pfam accessions | `eb7f81c5…5665ad` |
| `genome_information.csv` | yes | Genome metadata incl. the phenotype label column `pathogenicity`, keyed by `gca` | `530c223e…277e47` |
| `domain_descriptions.csv` | no | Pfam accession → description (interpretation tables only) | `1a42e800…2761d7` |

Full checksums are in `configs/config.yaml`.

## Automatic download

```bash
python -m xanthomonas_ml download-data      # downloads only what is missing, verifies SHA-256
python -m xanthomonas_ml validate-data      # re-checks local files, no network access
```

A `MANIFEST.json` recording source URL, commit, checksums, retrieval time and
licence note is written here after a successful download.

## Manual placement (no network / blocked download)

1. Open the GitLab repository above, select commit `1820d031`, and download the
   three files from `Rcode/Data/` ("Download" / raw file button).
2. Copy them into this folder **without renaming or editing** them (do not open
   and re-save them in Excel — that changes the bytes and the checksum).
3. Run `python -m xanthomonas_ml validate-data`.

If you deliberately use a different version of the files, add
`--allow-checksum-mismatch`; results are then no longer tied to the publication.

These files are not committed to Git: the source repository declares no
licence for redistribution. Cite the original publication when using them.
