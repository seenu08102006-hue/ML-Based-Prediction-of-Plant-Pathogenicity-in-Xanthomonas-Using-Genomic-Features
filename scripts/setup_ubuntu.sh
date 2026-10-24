#!/usr/bin/env bash
# Set up the project on Ubuntu (or another Debian-based Linux).
#
# Safe to run repeatedly: an existing .venv is reused, never deleted.
# This script never runs sudo. If an operating-system package is missing it
# prints the exact apt command and explains why the package is needed.
#
# Usage (from the project root):  bash scripts/setup_ubuntu.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON:-python3}"
MIN_MAJOR=3
MIN_MINOR=10

info() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. Python present and recent enough
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "python3 not found. Install it with:
    sudo apt update && sudo apt install -y python3 python3-venv python3-pip
(python3 runs the pipeline; python3-venv creates the isolated environment; python3-pip installs packages.)"
fi
PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "${PYTHON_BIN}" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_MAJOR}, ${MIN_MINOR}) else 1)"; then
    fail "Python ${PY_VERSION} found, but >= ${MIN_MAJOR}.${MIN_MINOR} is required.
Ubuntu 22.04 ships 3.10 and 24.04 ships 3.12, both supported. On older releases install a newer python3."
fi
info "Python ${PY_VERSION} found at $(command -v "${PYTHON_BIN}")"

# 2. Git (needed to clone and update the repository, not to run the pipeline)
if command -v git >/dev/null 2>&1; then
    info "$(git --version)"
else
    info "git not found (only needed for cloning/updating). Install with: sudo apt install -y git"
fi

# 3. venv support
if ! "${PYTHON_BIN}" -c "import venv, ensurepip" >/dev/null 2>&1; then
    fail "Python virtual-environment support is missing. Install it with:
    sudo apt update && sudo apt install -y python${PY_VERSION}-venv
(python${PY_VERSION}-venv provides the 'venv' and 'ensurepip' modules used to create .venv.)"
fi

# 4. Create the virtual environment only if it does not exist yet
if [ -x "${VENV_DIR}/bin/python" ]; then
    info "Reusing existing virtual environment ${VENV_DIR}"
else
    info "Creating virtual environment ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VPY="${VENV_DIR}/bin/python"

# 5-6. Install dependencies and the project (editable, so code updates apply after git pull)
info "Installing dependencies (requirements.txt) and the project package"
"${VPY}" -m pip install --upgrade pip
"${VPY}" -m pip install -r requirements.txt
"${VPY}" -m pip install -e .

# 7. Import check
"${VPY}" -c "import xanthomonas_ml, sklearn, pandas, matplotlib; print('imports OK: xanthomonas_ml', xanthomonas_ml.__version__)"

# 8. Environment check
"${VPY}" -m xanthomonas_ml check-env

cat <<EOF

Setup complete. Next steps:

  source .venv/bin/activate
  python -m xanthomonas_ml download-data      # fetch + verify the published dataset (~1 MB)
  python -m xanthomonas_ml run-all            # validate, prepare, EDA, train, interpret, report
  python -m xanthomonas_ml test               # automated tests

or simply:  bash scripts/run_pipeline.sh
EOF
