# Dataset

## Source and provenance

| Item | Value |
|---|---|
| Publication | te Molder D, Poncheewin W, Schaap PJ, Koehorst JJ (2021). *Machine learning approaches to predict the plant-associated phenotype of Xanthomonas strains.* BMC Genomics 22:848. [doi:10.1186/s12864-021-08093-0](https://doi.org/10.1186/s12864-021-08093-0) · [PMC8612006](https://pmc.ncbi.nlm.nih.gov/articles/PMC8612006/) |
| Files used | `Rcode/Data/domain_matrix.csv`, `genome_information.csv`, `domain_descriptions.csv` from the authors' repository <https://gitlab.com/wurssb/xanthomonas-phenotype-prediction> |
| Pinned version | commit `1820d031e38cc9fffee8c7fc4526890a5be5ff42` (25 May 2021, latest commit of `master`) |
| Integrity | SHA-256 per file in `configs/config.yaml`; verified on every download and before every analysis |
| Related dataset | Annotated genomes in RDF/HDT format, [doi:10.4121/14546625](https://doi.org/10.4121/14546625) (landing page: [WUR research portal](https://research.wur.nl/en/datasets/annotated-genomes-used-in-the-publication-machine-learning-approa/)). It holds the full SAPP annotation graphs from which the authors derived `domain_matrix.csv`; it is **not needed** here, and querying it requires an RDF/SPARQL stack. |
| Licence | The article is open access. When checked on 2026-10-24, the GitLab repository declared **no licence**. The data files are therefore not redistributed in this Git repository; the pipeline downloads them from the original source at run time. Cite the publication when you use them. |

### Why these files

The paper's supplementary file *Additional file 2* (`12864_2021_8093_MOESM2_ESM.xlsx`) contains the
same genome metadata and domain matrix. PMC blocks scripted downloads of it (an HTML page is
returned instead of the file), whereas the authors' GitLab repository serves the identical
analysis inputs as plain CSV at a fixed commit. Those CSVs are the files the authors' own R code
(`06_model_building.R`) used for model building.

## How the labels were produced (from the publication)

The authors mined 39 studies with 895 pathogenicity assays on 578 strains. A strain was labelled:

* **pathogenic**: it caused symptoms on at least one tested host;
* **non-pathogenic**: it caused no symptoms on its isolation host or any other tested host under
  conditions considered adequate. Trunk-incision and *Fragaria* assays were not accepted as
  evidence of non-pathogenicity;
* ambiguous strains (29) were excluded by the authors.

118 of the labelled strains had a public genome (65 pathogenic, 53 non-pathogenic). The authors
annotated all genomes with SAPP (Prodigal gene calling + InterProScan 5.44-79.0) and built a binary
Pfam domain presence/absence matrix.

## Contents of the files (as inspected)

### `domain_matrix.csv` (915,675 bytes)

* 119 rows × 3,715 columns, UTF-8 with byte-order mark.
* Column `gca`: GenBank assembly accession (e.g. `GCA_009769165.1`). This is the sample identifier.
* 3,714 Pfam accession columns (`PF00004` …). Values are integers 0/1 with no missing cells.

### `genome_information.csv` (30,463 bytes)

119 rows × 19 columns, with `"NA"` for missing values:

| Column | Meaning |
|---|---|
| `strain`, `alias_of` | Strain name and alternative name (11 strains have an alias) |
| `genus`, `species`, `subspecies` | Taxonomy (`unclassified` for 7 strains without species) |
| `hostGenus`, `hostSpecies` | Isolation host |
| **`pathogenicity`** | **Phenotype label**: `pathogenic` (65), `non-pathogenic` (53), `unknown` (1) |
| `source` | BibTeX keys of the source studies |
| **`gca`** | **Sample identifier** (joins to the domain matrix) |
| `refseq_category`, `assembly_level`, `seq_rel_date`, `ftp_path` | NCBI assembly metadata (75 contig-level, 31 scaffold, 12 complete, 1 chromosome) |
| `genome_size`, `gene_count`, `abundance`, `distinct`, `gene_w_domain` | Annotation statistics |

### `domain_descriptions.csv` (163,902 bytes)

3,714 rows: `accession`, `description` (InterPro-style names). Used only to annotate
interpretation tables.

## Label handling in this project

| Source label | Count | Used as | Reason |
|---|---|---|---|
| `pathogenic` | 65 | class 1 (positive) | Defined by the publication |
| `non-pathogenic` | 53 | class 0 | Defined by the publication |
| `unknown` | 1 | **excluded** | The genome is *Xylella fastidiosa* 9A5C (`GCA_000006725.1`). It is a different genus, probably included by the authors as an outgroup, and has no phenotype label. It is not re-mapped to either class. |
| missing / other | 0 | excluded | Would be excluded and listed if present |

"Non-pathogenic" in this dataset means **no disease was observed in the reported assays**. It does
not mean "non-plant-associated", and it does not mean the strain can never cause disease. This
dataset has no "non-plant-pathogenic" (e.g. human- or animal-associated) category.

## Validation performed (`python -m xanthomonas_ml validate-data`)

Checks: file presence and SHA-256; required columns; empty/duplicate identifiers in both tables;
duplicate feature names (checked on the raw header, because pandas silently renames duplicates);
missing, non-numeric, infinite and non-binary feature values; identifier overlap between tables
(alignment is by `gca` only, never by row order); missing and unexpected labels; genus; class
balance; constant features; identical feature columns; identical sample profiles.

Result on the published files (`reports/data_validation.md`): **passed with 0 errors and 3 warnings**:

1. one `unknown` label, excluded;
2. one non-*Xanthomonas* genome (the same *Xylella* genome), excluded;
3. two distinct strains, *X. arboricola* CFBP6825 and CFBP6827 (both pathogenic), have **identical domain
   profiles**. Both are kept because they are separate isolates in the source. When they fall into
   different CV folds, random CV is slightly optimistic.

Informational findings: all 119 identifiers match one-to-one; values are strictly 0/1; 1,135 of
3,714 domains are constant across the 118 strains; 910 of the remaining 2,579 variable domains are exact
copies of another column (perfect co-occurrence).

## Data manifest

After `download-data`, `data/raw/MANIFEST.json` records for each file the source URL, the pinned
commit, its role, size, verified SHA-256, the action taken (downloaded/reused) and the retrieval
time, together with the publication reference and the licence note.

| File | Format | Source | Role |
|---|---|---|---|
| `data/raw/domain_matrix.csv` | CSV (UTF-8 BOM) | GitLab, pinned commit | Features |
| `data/raw/genome_information.csv` | CSV (UTF-8 BOM, `NA` = missing) | GitLab, pinned commit | Labels + metadata |
| `data/raw/domain_descriptions.csv` | CSV (UTF-8 BOM, CRLF) | GitLab, pinned commit | Interpretation |
| `data/processed/*` | CSV / gzip CSV / JSON | Generated | Aligned analysis set (see `data/processed/README.md`) |
