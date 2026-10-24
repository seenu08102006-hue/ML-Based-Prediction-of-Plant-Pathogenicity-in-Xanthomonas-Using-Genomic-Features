#!/usr/bin/env bash
# Run the complete analysis on Linux/macOS: download -> validate -> prepare ->
# EDA -> train (nested CV) -> interpret -> report, then the automated tests.
#
# Usage (from anywhere):  bash scripts/run_pipeline.sh [--quick]
#   --quick  reduced CV/forest settings for a smoke test (results not reportable)
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    echo "No .venv found; run: bash scripts/setup_ubuntu.sh" >&2
    exit 4
fi

# Headless plotting is enforced in the code (Agg backend); MPLBACKEND is set as
# well so any third-party import also stays non-interactive.
export MPLBACKEND=Agg

"${PY}" -m xanthomonas_ml run-all "$@"
"${PY}" -m xanthomonas_ml test -q

echo
echo "Done. Report: reports/project_report.md   Figures: outputs/figures/   Models: models/"
