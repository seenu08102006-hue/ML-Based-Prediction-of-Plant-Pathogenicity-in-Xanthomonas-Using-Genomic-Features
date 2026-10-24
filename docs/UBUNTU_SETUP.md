# Ubuntu setup and execution guide

A copy-paste workflow for a fresh Ubuntu terminal (22.04 or 24.04 LTS). The Windows workspace is not needed:
everything comes from GitHub plus the public data source.

> Verification status: these commands run in GitHub Actions on `ubuntu-latest` (`.github/workflows/ci.yml`), which
> executes `scripts/setup_ubuntu.sh` and the full pipeline with `--quick` settings. They were not run on a
> physical Ubuntu machine during development.

## A. Install Git (if missing)

```bash
git --version || { sudo apt update && sudo apt install -y git; }
```

## B. Python and virtual-environment support

Ubuntu 22.04 ships Python 3.10 and 24.04 ships Python 3.12; both are supported (3.10+ is required).

```bash
python3 --version
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
```

`python3-venv` provides the module that creates the isolated `.venv`, and `python3-pip` installs Python
packages. On 24.04 you may need the versioned package: `sudo apt install -y python3.12-venv`.

## C–D. Clone and enter the project

The repository is public, so cloning over HTTPS needs no username or password.

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/seenu08102006-hue/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features.git
cd ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features
```

## E–H. Environment, dependencies, package, check (simplest method)

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

The script checks Python ≥ 3.10, Git and venv support, creates `.venv` (or reuses it), installs
`requirements.txt` and the package (`pip install -e .`), tests the imports and runs `check-env`. It never calls
`sudo`; if something is missing it prints the exact `apt` command to run.

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m xanthomonas_ml check-env
```

## I. Get and validate the authentic dataset

```bash
python -m xanthomonas_ml download-data     # ~1 MB from GitLab, SHA-256 verified, skipped if already present
python -m xanthomonas_ml inspect-data      # files, shapes, columns, label values
python -m xanthomonas_ml validate-data     # offline checks -> reports/data_validation.md
python -m xanthomonas_ml prepare-data      # aligned analysis set -> data/processed/
```

No network? Copy the three CSV files into `data/raw/` as described in `data/raw/README.md`, then run
`validate-data`.

## J. Exploratory data analysis

```bash
python -m xanthomonas_ml eda               # figures -> outputs/figures/eda_*.png, tables -> outputs/tables/
```

Plots render headlessly (matplotlib Agg backend); no display server is needed.

## K–L. Train and evaluate

```bash
python -m xanthomonas_ml train             # nested CV for all 3 models + final models (about 20-30 min)
python -m xanthomonas_ml train --quick     # 3-5 min smoke test; do not report these numbers
python -m xanthomonas_ml evaluate          # recompute metrics and evaluation figures from saved results
python -m xanthomonas_ml show-best         # model ranking
python -m xanthomonas_ml model-info --model random_forest
```

For long runs over SSH, keep the job alive after logout:

```bash
nohup python -m xanthomonas_ml train > train.log 2>&1 &
tail -f train.log
```

## M. Interpretation, report and graphs

```bash
python -m xanthomonas_ml interpret         # outputs/tables/feature_importance_<model>.csv + figures
python -m xanthomonas_ml report            # reports/project_report.md
```

Or run steps I–M in one go: `bash scripts/run_pipeline.sh` (add `--quick` for a smoke test). It also runs the tests.

## N. Automated tests

```bash
python -m xanthomonas_ml test              # = python -m pytest tests
python -m pytest -m realdata               # only the checks on the authentic dataset
```

## O. Predict for new strains

The input must be Pfam presence/absence, **not DNA sequence**:

```bash
# from InterProScan TSV output (one file per genome; Pfam must be among the analyses)
python -m xanthomonas_ml build-features --interproscan genomeA.tsv genomeB.tsv --model random_forest --output new_strains.csv
python -m xanthomonas_ml predict --input new_strains.csv --model random_forest --output outputs/predictions/new_strains.csv
```

`--model` accepts `best` (the default), a model name, or a path to a `.joblib` file. See `README.md` for the input format.

## P. Update your Ubuntu copy later

```bash
cd ~/projects/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features
git status                 # check for local changes first
git pull --ff-only         # fast-forward only: never creates surprise merges
source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .   # in case dependencies changed
python -m xanthomonas_ml test
```

* If `git status` shows **local changes you want to keep**, commit them first (`git add -A && git commit -m "..."`),
  then run `git pull --rebase` and resolve conflicts if Git reports any (edit the files, `git add <file>`,
  `git rebase --continue`).
* If you **only want to set local edits aside**, run `git stash` before pulling and `git stash pop` after.
* If `git pull --ff-only` refuses, your branch has diverged. Inspect it with `git log --oneline --graph --all -n 20`
  before deciding. Do not use `git reset --hard` or `git clean` as routine fixes; they delete work.
* Data, models and outputs are not tracked by Git, so pulling never overwrites them.

## Pushing from Ubuntu (optional)

Cloning a public repository needs no login, but **pushing does**. GitHub does not accept account passwords for
Git over HTTPS. Use one of:

* **GitHub CLI**: `sudo apt install -y gh && gh auth login`, then `git push` works over HTTPS.
* **SSH key**: `ssh-keygen -t ed25519 -C "you@example.com"`, add `~/.ssh/id_ed25519.pub` under GitHub → Settings → SSH
  keys, then `git remote set-url origin git@github.com:seenu08102006-hue/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features.git`.
* **Personal access token**: create a fine-grained token with *Contents: read and write* for this repository and enter
  it when Git asks for a password. Never put the token in a file in the repository.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ensurepip is not available` / `No module named venv` | `sudo apt install -y python3-venv` (or `python3.12-venv`) |
| `error: externally-managed-environment` | You ran pip outside the venv. Activate `.venv` first or call `.venv/bin/python -m pip` |
| `Dataset verification failed: missing required file` | Run `download-data`, or place the files manually (`data/raw/README.md`) |
| `SHA-256 ... does not match` | File modified or a different version. Delete it and re-download; use `--allow-checksum-mismatch` only deliberately |
| `Could not download ...` | No internet or proxy. Check `curl -I https://gitlab.com`, or place the files manually |
| `Model file not found` / `show-best` exit code 4 | Run `train` first |
| Exit code 3 from `predict` | The input file failed validation; the message names the offending columns or cells |
| `Model was saved with scikit-learn X but Y is installed` | Retrain on this machine (`train`) instead of copying `.joblib` files between environments |
| `_tkinter` / display errors | Should not happen (Agg backend). If a third-party import triggers it, `export MPLBACKEND=Agg` |
| Training is slow | Reduce `cv.outer_repeats` in `configs/config.yaml`, or use `--quick` for testing |

Exit codes: 0 success, 1 unexpected error, 2 wrong command-line usage, 3 data/validation problem, 4 missing
prerequisite (file, model, results).
