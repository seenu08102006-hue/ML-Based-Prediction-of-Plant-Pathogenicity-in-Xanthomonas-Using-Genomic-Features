# Windows setup (PowerShell)

Tested on Windows 11 with Python 3.12.1 in the project virtual environment, using Windows PowerShell 5.1.

## 1. Prerequisites

* **Python 3.10–3.12** from <https://www.python.org/downloads/>. Python 3.12 is recommended because it is also
  the Ubuntu 24.04 default. During installation tick *Add python.exe to PATH*. The `py` launcher is installed
  automatically.
* **Git** from <https://git-scm.com/download/win>.

Check:

```powershell
py -0p          # lists installed Python versions
git --version
```

## 2. Get the code

```powershell
cd $HOME\Desktop
git clone https://github.com/seenu08102006-hue/ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features.git
cd ML-Based-Prediction-of-Plant-Pathogenicity-in-Xanthomonas-Using-Genomic-Features
```

## 3. One-command setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# or choose another installed version: ... -File scripts\setup_windows.ps1 -PythonVersion 3.11
```

The script finds Python, creates `.venv` (or reuses an existing one), installs `requirements.txt` and the package
in editable mode, and runs `check-env`. It does not change your PowerShell profile or execution policy;
`-ExecutionPolicy Bypass` applies only to that single command.

### Manual alternative

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## 4. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
# If PowerShell refuses ("running scripts is disabled"), allow it for this window only:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You can also skip activation and call `.\.venv\Scripts\python.exe -m xanthomonas_ml ...` directly.

## 5. Run

```powershell
python -m xanthomonas_ml check-env
python -m xanthomonas_ml download-data
python -m xanthomonas_ml validate-data
python -m xanthomonas_ml prepare-data
python -m xanthomonas_ml eda
python -m xanthomonas_ml train            # about 20-30 min with the default nested CV
python -m xanthomonas_ml interpret
python -m xanthomonas_ml report
python -m xanthomonas_ml show-best
python -m xanthomonas_ml test
# or everything at once:
python -m xanthomonas_ml run-all
```

## 6. VS Code

Open the folder, run *Python: Select Interpreter* and choose `.venv\Scripts\python.exe`. VS Code is not needed to
run the project.

## Troubleshooting

| Problem | Fix |
|---|---|
| `py` not found | Use `python` instead, or reinstall Python with the launcher option |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (current window only) |
| `pip` SSL / proxy errors | Configure your proxy (`pip config set global.proxy http://...`) or install on another network |
| `download-data` fails | Follow the manual steps in `data/raw/README.md`, then run `validate-data` |
| Excel changed the CSV | Never open and re-save the raw CSVs; download them again |
