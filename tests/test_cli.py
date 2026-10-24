"""CLI tests: help, exit codes, offline end-to-end run on SYNTHETIC data."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from xanthomonas_ml.cli import EXIT_DATA, EXIT_MISSING, EXIT_OK, build_parser, main

COMMANDS = ["check-env", "download-data", "inspect-data", "validate-data", "prepare-data", "eda", "train", "evaluate",
            "interpret", "show-best", "model-info", "predict", "build-features", "report", "run-all", "test"]


def test_module_help_subprocess():
    result = subprocess.run([sys.executable, "-m", "xanthomonas_ml", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for command in COMMANDS:
        assert command in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--help"])
    assert exc.value.code == 0


def test_no_command_prints_help(capsys):
    assert main([]) == EXIT_OK
    assert "usage" in capsys.readouterr().out


def test_unknown_command_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["fly"])
    assert exc.value.code == 2


def test_check_env(synthetic_project, capsys):
    assert main(["--config", str(synthetic_project.source), "check-env"]) == EXIT_OK
    assert "Environment OK" in capsys.readouterr().out


def test_missing_config_exit_code(tmp_path):
    assert main(["--config", str(tmp_path / "nope.yaml"), "check-env"]) == EXIT_MISSING


def test_validate_missing_data_exit_code(synthetic_project):
    (synthetic_project.raw_dir / "domain_matrix.csv").unlink()
    assert main(["--config", str(synthetic_project.source), "validate-data"]) == EXIT_DATA


def test_commands_needing_training_fail_cleanly(synthetic_project):
    cfg = str(synthetic_project.source)
    assert main(["--config", cfg, "show-best"]) == EXIT_MISSING
    assert main(["--config", cfg, "interpret"]) == EXIT_MISSING
    assert main(["--config", cfg, "model-info", "--model", "random_forest"]) == EXIT_MISSING


def test_offline_end_to_end(synthetic_project, no_network, tmp_path, capsys):
    """Full workflow with local data only; any network access would raise."""
    cfg = str(synthetic_project.source)
    for command in (["inspect-data"], ["validate-data"], ["prepare-data"], ["eda"],
                    ["train", "--models", "decision_tree", "l1_logistic_regression"], ["evaluate"], ["interpret"],
                    ["show-best"], ["model-info"], ["report"]):
        assert main(["--config", cfg, *command]) == EXIT_OK, command
    # Output directories were created on demand.
    for path in (synthetic_project.processed_dir, synthetic_project.models_dir, synthetic_project.figures_dir,
                 synthetic_project.tables_dir, synthetic_project.reports_dir):
        assert path.is_dir()
    data = pd.read_csv(synthetic_project.processed_dir / "features.csv.gz").head(3)
    new = tmp_path / "new_strains.csv"
    data.to_csv(new, index=False)
    out = tmp_path / "pred.csv"
    assert main(["--config", cfg, "predict", "--input", str(new), "--output", str(out)]) == EXIT_OK
    assert len(pd.read_csv(out)) == 3
    assert "not a diagnosis" in capsys.readouterr().out.lower()


def test_predict_bad_input_exit_code(trained_project, tmp_path):
    cfg, data = trained_project
    bad = data.X.head(2).reset_index().astype(str)
    bad.iloc[0, 1:] = "7"
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    code = main(["--config", str(cfg.source), "predict", "--model", "random_forest", "--input", str(path)])
    assert code == EXIT_DATA


def test_external_evaluation(trained_project, tmp_path, capsys):
    cfg, data = trained_project
    table = data.X.reset_index()
    table["pathogenicity"] = data.y.map({1: "pathogenic", 0: "non-pathogenic"}).to_numpy()
    path = tmp_path / "labelled.csv"
    table.to_csv(path, index=False)
    assert main(["--config", str(cfg.source), "evaluate", "--input", str(path), "--model", "decision_tree"]) == EXIT_OK
    assert "balanced_accuracy" in capsys.readouterr().out
