"""Command-line interface: ``python -m xanthomonas_ml <command> [options]``.

Exit codes: 0 success, 1 unexpected error, 2 usage error (argparse),
3 data/validation failure, 4 missing prerequisite (file, model or results).
"""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from xanthomonas_ml import RESEARCH_DISCLAIMER, __version__
from xanthomonas_ml.config import Config, ConfigError, load_config
from xanthomonas_ml.utils import get_logger, setup_logging

log = get_logger("cli")

EXIT_OK, EXIT_ERROR, EXIT_DATA, EXIT_MISSING = 0, 1, 3, 4
# Settings applied by --quick: fast enough for a smoke test, but the numbers must not be reported.
QUICK_OVERRIDES = {
    "cv": {"outer_repeats": 2, "species_grouped_folds": 3},
    "models": {"random_forest": {"n_estimators": 100}},
    "interpretation": {"permutation_top_k": 3, "permutation_repeats": 2},
}


# ----------------------------------------------------------------------------- commands
def cmd_check_env(cfg: Config, args: argparse.Namespace) -> int:
    print(f"xanthomonas_ml {__version__}")
    print(f"Python {platform.python_version()} ({sys.executable}) on {platform.platform()}")
    # Each check prints one line; any failure makes the overall result (and exit code) an error.
    ok = True
    if sys.version_info < (3, 10):
        print("  [FAIL] Python >= 3.10 is required")
        ok = False
    for module in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "yaml", "joblib", "requests"):
        try:
            mod = importlib.import_module(module)
            print(f"  [ OK ] {module:<11} {getattr(mod, '__version__', '')}")
        except ImportError as exc:
            print(f"  [FAIL] {module:<11} not importable: {exc}")
            ok = False
    try:
        import matplotlib

        from xanthomonas_ml import plotting  # noqa: F401  (selects the headless Agg backend)

        print(f"  [ OK ] plotting backend: {matplotlib.get_backend()} (headless-safe)")
    except ImportError as exc:
        print(f"  [FAIL] plotting unavailable: {exc}")
        ok = False
    print(f"Config: {cfg.source}")
    print(f"Project root: {cfg.root}")
    for label, path in (("raw data", cfg.raw_dir), ("processed", cfg.processed_dir), ("models", cfg.models_dir),
                        ("outputs", cfg.outputs_dir), ("reports", cfg.reports_dir)):
        print(f"  {label:<10} {path}  ({'exists' if path.exists() else 'will be created'})")
    from xanthomonas_ml.dataset_download import check_local_files

    present = [s for s in check_local_files(cfg) if s.exists]
    print(f"Dataset files present: {len(present)}/{len(check_local_files(cfg))} "
          "(run 'download-data' or 'validate-data' for details)")
    print("Environment OK" if ok else "Environment has problems (see above)")
    return EXIT_OK if ok else EXIT_ERROR


def cmd_download(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.dataset_download import download_dataset

    manifest = download_dataset(cfg, force=args.force, timeout=args.timeout, retries=args.retries)
    for f in manifest["files"]:
        print(f"  {f['action']:<10} {f['filename']:<26} {f['size_bytes']:>9} bytes  sha256 OK")
    print(f"Manifest: {cfg.raw_dir / 'MANIFEST.json'}")
    return EXIT_OK


def cmd_inspect(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.data_loader import discover_files, load_feature_matrix, load_metadata
    from xanthomonas_ml.dataset_download import check_local_files

    print(f"Raw data directory: {cfg.raw_dir}")
    for st in check_local_files(cfg):
        state = "missing" if not st.exists else ("checksum OK" if st.checksum_ok else "CHECKSUM MISMATCH")
        print(f"  [{'req' if st.required else 'opt'}] {st.filename:<26} {state}")
    others = [p.name for p in discover_files(cfg.raw_dir)
              if p.name not in {s.filename for s in check_local_files(cfg)}]
    if others:
        print(f"  Other tabular files (not used): {others}")
    files = cfg["dataset"]["files"]
    fm, gi = cfg.raw_dir / files["domain_matrix"]["filename"], cfg.raw_dir / files["genome_information"]["filename"]
    if fm.is_file():
        df = load_feature_matrix(fm)
        print(f"\n{fm.name}: {df.shape[0]} rows x {df.shape[1]} columns; id column '{df.columns[0]}'; "
              f"first features {list(df.columns[1:6])}")
    if gi.is_file():
        meta = load_metadata(gi)
        print(f"\n{gi.name}: {meta.shape[0]} rows x {meta.shape[1]} columns")
        for col in meta.columns:
            print(f"  {col:<16} missing={int(meta[col].isna().sum()):>3}  e.g. {meta[col].dropna().iloc[0] if meta[col].notna().any() else ''}")
        label = cfg["dataset"]["label_column"]
        if label in meta.columns:
            print(f"\nLabel column '{label}': {meta[label].fillna('<missing>').value_counts().to_dict()}")
    if not fm.is_file() and not gi.is_file():
        print("\nNo dataset files found. Run 'python -m xanthomonas_ml download-data'.")
        return EXIT_MISSING
    return EXIT_OK


def cmd_validate(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.data_loader import load_raw_dataset
    from xanthomonas_ml.dataset_download import check_local_files, require_valid_files
    from xanthomonas_ml.validation import validate_raw_dataset

    paths = require_valid_files(cfg, allow_checksum_mismatch=args.allow_checksum_mismatch)
    report = validate_raw_dataset(load_raw_dataset(paths), cfg, check_local_files(cfg))
    json_path, md_path = report.save(cfg.reports_dir)
    for issue in report.issues:
        print(f"  {issue.level:<7} {issue.code}: {issue.message}")
    print(f"\nValidation {'PASSED' if report.ok else 'FAILED'}: {len(report.errors)} errors, "
          f"{len(report.warnings)} warnings\nReport: {md_path}")
    return EXIT_OK if report.ok else EXIT_DATA


def cmd_prepare(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.preprocessing import prepare_or_load

    data = prepare_or_load(cfg, force=args.force, allow_checksum_mismatch=args.allow_checksum_mismatch)
    counts = data.y.value_counts()
    print(f"Analysis set: {data.X.shape[0]} strains x {data.X.shape[1]} features; "
          f"pathogenic = {counts.get(1, 0)}, non-pathogenic = {counts.get(0, 0)}")
    print(f"Removed samples: {len(data.removed_samples)}; removed constant features: {len(data.removed_features)}")
    print(f"Saved in {cfg.processed_dir}")
    return EXIT_OK


def cmd_eda(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.eda import run_eda
    from xanthomonas_ml.preprocessing import prepare_or_load

    summary = run_eda(prepare_or_load(cfg), cfg)
    print(f"EDA complete for {summary['n_strains']} strains; figures in {cfg.figures_dir}, tables in {cfg.tables_dir}")
    return EXIT_OK


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.evaluate import generate_evaluation_outputs, headline_table
    from xanthomonas_ml.preprocessing import prepare_or_load
    from xanthomonas_ml.train import run_training

    # Pipeline: prepare the data, run nested CV and fit the final models, then draw the figures from the saved predictions.
    data = prepare_or_load(cfg)
    run_training(data, cfg, models=args.models, skip_species_cv=args.skip_species_cv)
    generate_evaluation_outputs(cfg)
    print("\nOut-of-fold performance (mean ± sd over CV repeats):")
    print(headline_table(cfg).to_string(index=False))
    print(f"\nModels saved in {cfg.models_dir}; results in {cfg.outputs_dir / 'results'}")
    return EXIT_OK


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.evaluate import generate_evaluation_outputs, headline_table

    if args.input:
        return _evaluate_external(cfg, args)
    out = generate_evaluation_outputs(cfg)
    print(headline_table(cfg).to_string(index=False))
    print(f"\n{len(out['figures'])} evaluation figures written to {cfg.figures_dir}")
    return EXIT_OK


def _evaluate_external(cfg: Config, args: argparse.Namespace) -> int:
    """Evaluate a saved model on an independent labelled feature file."""
    import pandas as pd

    from xanthomonas_ml.evaluate import compute_metrics
    from xanthomonas_ml.predict import predict_dataframe, read_feature_file, validate_input
    from xanthomonas_ml.train import load_model_bundle

    bundle = load_model_bundle(_model_path(cfg, args.model))
    df = read_feature_file(Path(args.input))
    if args.label_column not in df.columns:
        print(f"Label column '{args.label_column}' not found in {args.input}", file=sys.stderr)
        return EXIT_DATA
    # Remove the label column from the features so the true answer cannot reach the model.
    labels = df.pop(args.label_column).str.strip()
    mapping = bundle["label_mapping"]
    keep = labels.isin(mapping)
    if not keep.all():
        print(f"Excluding {int((~keep).sum())} rows whose label is not one of {list(mapping)}")
    X, report = validate_input(df[keep.to_numpy()], bundle, id_column=args.id_column,
                               fill_missing_as_absent=args.fill_missing_as_absent)
    pred = predict_dataframe(bundle, X, report)
    y_true = labels[keep].map(mapping).to_numpy()
    metrics = compute_metrics(y_true, pred["predicted_class"].map(mapping).to_numpy(), pred["probability_pathogenic"])
    print(f"External evaluation of {bundle['model_name']} on {len(y_true)} labelled strains:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    if args.output:
        pd.DataFrame([{k: v for k, v in metrics.items() if k != "undefined"}]).to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    return EXIT_OK


def cmd_interpret(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.interpretability import CAVEAT, run_interpretation

    top = run_interpretation(cfg)
    for model, feats in top.items():
        print(f"{model}: top domains {feats[:8]}")
    print(f"\nTables: {cfg.tables_dir}/feature_importance_<model>.csv\n{CAVEAT}")
    return EXIT_OK


def cmd_show_best(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.evaluate import best_model_info, headline_table

    best = best_model_info(cfg)
    print(f"Ranking by mean {best['selection_metric']} ({best['design']}, {best['level']}):")
    for i, r in enumerate(best["ranking"], 1):
        # sd is undefined (null) when only one CV repeat was run.
        sd = f" ± {r['std']:.3f}" if r.get("std") is not None else " (sd undefined: 1 repeat)"
        mean = f"{r['mean']:.3f}" if r.get("mean") is not None else "undefined"
        print(f"  {i}. {r['model']:<24} {mean}{sd} (n={r['n_repeats']})")
    if best.get("note"):
        print(best["note"])
    print("\n" + headline_table(cfg).to_string(index=False))
    return EXIT_OK


def _model_path(cfg: Config, model: str | None) -> Path:
    """Resolve a model name ('random_forest'), 'best', or a file path."""
    if model and (model.endswith(".joblib") or Path(model).is_file()):
        return Path(model)
    if not model or model == "best":
        from xanthomonas_ml.evaluate import best_model_info

        model = best_model_info(cfg)["best_model"]
    return cfg.models_dir / f"{model}.joblib"


def cmd_model_info(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.train import load_model_bundle

    path = _model_path(cfg, args.model)
    bundle = load_model_bundle(path)
    print(f"Model file: {path}")
    for key in ("model_name", "model_label", "trained_utc", "best_params", "inner_cv_best_score",
                "n_training_samples", "training_class_counts", "n_features_after_filters", "label_mapping",
                "feature_encoding", "versions"):
        print(f"  {key}: {bundle.get(key)}")
    print(f"  input schema: {len(bundle['feature_schema'])} Pfam columns; required (used after filters): "
          f"{len(bundle['selected_features'])}")
    print("  NOTE: resubstitution metrics in the JSON sidecar are training-set scores, not generalisation.")
    return EXIT_OK


def cmd_predict(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.predict import predict_file

    output = Path(args.output) if args.output else cfg.outputs_dir / "predictions" / f"predictions_{Path(args.input).stem}.csv"
    result = predict_file(_model_path(cfg, args.model), Path(args.input), output, id_column=args.id_column,
                          fill_missing_as_absent=args.fill_missing_as_absent)
    # The note column is dropped from the printed table; the disclaimer is printed once below instead.
    with __import__("pandas").option_context("display.width", 140, "display.max_columns", 10):
        print(result.drop(columns=["note"]).to_string(index=False))
    print(f"\nSaved to {output}\n\nWARNING: {RESEARCH_DISCLAIMER}")
    return EXIT_OK


def cmd_build_features(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.predict import build_feature_table

    schema = None
    if args.model:
        from xanthomonas_ml.train import load_model_bundle

        schema = load_model_bundle(_model_path(cfg, args.model))["feature_schema"]
    table = build_feature_table([Path(p) for p in args.interproscan], args.sample_ids, args.id_column, schema)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} sample(s) x {table.shape[1] - 1} Pfam columns to {out}")
    return EXIT_OK


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    from xanthomonas_ml.reporting import generate_report

    path = generate_report(cfg, Path(args.output) if args.output else None)
    print(f"Report written to {path}")
    return EXIT_OK


def cmd_run_all(cfg: Config, args: argparse.Namespace) -> int:
    steps: list[tuple[str, Callable[[Config, argparse.Namespace], int]]] = [
        ("download-data", cmd_download), ("validate-data", cmd_validate), ("prepare-data", cmd_prepare),
        ("eda", cmd_eda), ("train", cmd_train), ("interpret", cmd_interpret), ("report", cmd_report),
    ]
    for name, func in steps:
        print(f"\n=== {name} ===")
        code = func(cfg, args)
        if code != EXIT_OK:
            print(f"Stopped: step '{name}' failed with exit code {code}", file=sys.stderr)
            return code
    return EXIT_OK


def cmd_test(cfg: Config, args: argparse.Namespace) -> int:
    extra = args.pytest_args or []
    # Default to the whole tests/ folder unless the user named specific test paths.
    has_path = any(not a.startswith("-") and Path(cfg.root, a.split("::")[0]).exists() for a in extra)
    command = [sys.executable, "-m", "pytest"] + ([] if has_path else [str(cfg.root / "tests")]) + extra
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=cfg.root)


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m xanthomonas_ml",
        description="ML prediction of the plant-pathogenic phenotype of Xanthomonas strains from Pfam domain "
                    "presence/absence (data: te Molder et al. 2021, BMC Genomics 22:848).",
        epilog="Typical workflow: download-data -> validate-data -> prepare-data -> eda -> train -> interpret -> "
               "report (or simply: run-all).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="path to config YAML (default: configs/config.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Helper: register a subcommand and store its handler, so main() can call args.func.
    def add(name: str, func: Callable, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    def checksum_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--allow-checksum-mismatch", action="store_true",
                       help="accept raw files that differ from the pinned published version (logged)")

    def quick_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--quick", action="store_true",
                       help="smoke-test settings (2 CV repeats, 100 trees); NOT for reported results")

    add("check-env", cmd_check_env, "check Python, dependencies and project paths")
    p = add("download-data", cmd_download, "download (if needed) and verify the published dataset")
    p.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
    p.add_argument("--timeout", type=float, default=60.0, help="seconds per request (default 60)")
    p.add_argument("--retries", type=int, default=3, help="attempts per file (default 3)")
    add("inspect-data", cmd_inspect, "show files, shapes, columns and label values of the raw data")
    checksum_flag(add("validate-data", cmd_validate, "validate local raw data without downloading"))
    p = add("prepare-data", cmd_prepare, "build and save the aligned analysis dataset")
    p.add_argument("--force", action="store_true", help="rebuild even if processed data are current")
    checksum_flag(p)
    add("eda", cmd_eda, "exploratory data analysis (figures + tables)")
    p = add("train", cmd_train, "nested CV evaluation, final model fitting and saving")
    p.add_argument("--models", nargs="+", choices=["decision_tree", "l1_logistic_regression", "random_forest"],
                   help="subset of models (default: all enabled in config)")
    p.add_argument("--skip-species-cv", action="store_true", help="skip the species-grouped CV design")
    p.add_argument("--outer-repeats", type=int, help="override cv.outer_repeats")
    quick_flag(p)
    p = add("evaluate", cmd_evaluate,
            "recompute metrics and evaluation figures from saved CV results, or evaluate a saved model on an "
            "independent labelled file (--input)")
    p.add_argument("--input", help="labelled feature file for external evaluation")
    p.add_argument("--label-column", default="pathogenicity", help="label column in --input")
    p.add_argument("--model", default="best", help="model name, 'best' or path (external evaluation)")
    p.add_argument("--id-column", default="gca")
    p.add_argument("--fill-missing-as-absent", action="store_true")
    p.add_argument("--output", help="CSV for external metrics")
    add("interpret", cmd_interpret, "feature-importance tables and figures")
    add("show-best", cmd_show_best, "show the model ranking from the last training run")
    p = add("model-info", cmd_model_info, "load a saved model and print its metadata")
    p.add_argument("--model", default="best", help="model name, 'best' or path to a .joblib file")
    p = add("predict", cmd_predict, "predict the phenotype of strains from a Pfam presence/absence file")
    p.add_argument("--input", required=True, help="CSV/TSV: id column + Pfam columns with 0/1")
    p.add_argument("--model", default="best", help="model name, 'best' (default) or path to a .joblib file")
    p.add_argument("--output", help="output CSV (default: outputs/predictions/predictions_<input>.csv)")
    p.add_argument("--id-column", default="gca", help="identifier column name (default: gca)")
    p.add_argument("--fill-missing-as-absent", action="store_true",
                   help="treat required Pfam columns missing from the input as absent (0) - use with care")
    p = add("build-features", cmd_build_features,
            "convert InterProScan TSV output (one file per genome) into a presence/absence input file")
    p.add_argument("--interproscan", nargs="+", required=True, help="InterProScan TSV file(s)")
    p.add_argument("--sample-ids", nargs="+", help="identifiers (default: file names)")
    p.add_argument("--output", required=True, help="output CSV")
    p.add_argument("--id-column", default="gca")
    p.add_argument("--model", help="include every column of this model's schema (0 = not detected)")
    p = add("report", cmd_report, "generate the Markdown project report from saved outputs")
    p.add_argument("--output", help="output path (default: reports/project_report.md)")
    p = add("run-all", cmd_run_all, "download, validate, prepare, eda, train, interpret and report")
    for flag in ("--force",):
        p.add_argument(flag, action="store_true", help="force re-download and re-preparation")
    p.set_defaults(timeout=60.0, retries=3, allow_checksum_mismatch=False, models=None, skip_species_cv=False,
                   outer_repeats=None, output=None)
    quick_flag(p)
    p = add("test", cmd_test, "run the automated test suite (pytest)")
    p.add_argument("pytest_args", nargs=argparse.REMAINDER, help="extra arguments passed to pytest")
    return parser


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    if getattr(args, "quick", False):
        overrides = {k: dict(v) for k, v in QUICK_OVERRIDES.items()}
        log.warning("--quick: reduced CV/forest settings for a smoke test; do not report these results")
    if getattr(args, "outer_repeats", None):
        overrides.setdefault("cv", {})["outer_repeats"] = args.outer_repeats
    return cfg.with_overrides(overrides) if overrides else cfg


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = build_parser()
    # Unknown options are allowed only for 'test', where they are passed to pytest (e.g. -q, -k NAME).
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        if args.command != "test":
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        args.pytest_args = unknown + (args.pytest_args or [])
    if not args.command:
        parser.print_help()
        return EXIT_OK
    setup_logging(args.verbose)
    from xanthomonas_ml.dataset_download import DatasetError
    from xanthomonas_ml.predict import PredictionInputError
    from xanthomonas_ml.preprocessing import PreparationError

    try:
        cfg = _apply_overrides(load_config(args.config), args)
        return args.func(cfg, args)
    # Known failure types get their own exit codes (3 = data, 4 = missing prerequisite), so scripts can tell them apart.
    except (ConfigError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return EXIT_MISSING
    except (DatasetError, PreparationError, PredictionInputError) as exc:
        log.error("%s", exc)
        return EXIT_DATA
    except KeyboardInterrupt:
        log.error("Interrupted")
        return EXIT_ERROR
    except Exception as exc:  # last-resort handler: show the message, keep a traceback in verbose mode
        log.error("Unexpected error: %s: %s", type(exc).__name__, exc, exc_info=args.verbose)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
