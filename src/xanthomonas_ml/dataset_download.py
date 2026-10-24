"""Acquisition and integrity verification of the published dataset.

The authentic data are the CSV files published by te Molder et al. (2021) in the
authors' GitLab repository. Each file is pinned to a fixed commit and verified
against a SHA-256 checksum, so we never silently analyse a different file.

Files are written to ``data/raw`` (configurable). A valid local copy is never
re-downloaded unless ``force=True``. Users without network access can place the
files manually in ``data/raw``; they are then verified exactly the same way.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xanthomonas_ml.config import Config
from xanthomonas_ml.utils import ensure_dir, get_logger, sha256_file, utc_now_iso, write_json

log = get_logger("download")

MANIFEST_NAME = "MANIFEST.json"
# Stored in every manifest, so the redistribution limits travel with the data.
LICENSE_NOTE = (
    "Data originate from the public GitLab repository accompanying te Molder et al. (2021), "
    "BMC Genomics 22:848 (open-access article). When checked on 2026-10-24 the GitLab "
    "repository declared no explicit license, so the data files are NOT redistributed in "
    "this project's Git repository; they are downloaded from the original source at run "
    "time. Cite the original publication when using them."
)


class DatasetError(RuntimeError):
    """Raised for missing, corrupt or unverifiable dataset files."""


@dataclass
class FileSpec:
    """Description of one published input file."""

    key: str
    filename: str
    url: str
    sha256: str | None
    role: str
    required: bool


@dataclass
class FileStatus:
    """Integrity status of one local file."""

    key: str
    filename: str
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    expected_sha256: str | None
    checksum_ok: bool | None
    required: bool

    @property
    def usable(self) -> bool:
        return self.exists and self.checksum_ok is not False


def dataset_file_specs(cfg: Config) -> list[FileSpec]:
    """Build the list of expected input files from the configuration."""
    ds = cfg["dataset"]
    specs: list[FileSpec] = []
    for key, entry in ds["files"].items():
        url = ds["base_url"].format(commit=ds["commit"], filename=entry["filename"])
        specs.append(
            FileSpec(
                key=key,
                filename=entry["filename"],
                url=url,
                sha256=entry.get("sha256"),
                role=entry.get("role", ""),
                required=bool(entry.get("required", True)),
            )
        )
    return specs


def check_local_files(cfg: Config) -> list[FileStatus]:
    """Inspect local raw files without any network access.

    Returns:
        One :class:`FileStatus` per configured file.
    """
    statuses: list[FileStatus] = []
    for spec in dataset_file_specs(cfg):
        path = cfg.raw_dir / spec.filename
        exists = path.is_file()
        digest = sha256_file(path) if exists else None
        # None = not checked (file missing or no pinned checksum); only False counts as a failure.
        ok = None if not exists or spec.sha256 is None else digest == spec.sha256
        statuses.append(
            FileStatus(
                key=spec.key,
                filename=spec.filename,
                path=path.as_posix(),
                exists=exists,
                size_bytes=path.stat().st_size if exists else None,
                sha256=digest,
                expected_sha256=spec.sha256,
                checksum_ok=ok,
                required=spec.required,
            )
        )
    return statuses


def require_valid_files(cfg: Config, allow_checksum_mismatch: bool = False) -> dict[str, Path]:
    """Ensure all required files are present and verified; return their paths.

    Args:
        cfg: Loaded configuration.
        allow_checksum_mismatch: Accept files whose checksum differs from the
            pinned published version (e.g. a manually curated update). A warning
            is logged, because results are then no longer tied to the publication.

    Raises:
        DatasetError: if a required file is missing or fails verification.
    """
    paths: dict[str, Path] = {}
    problems: list[str] = []
    for status in check_local_files(cfg):
        if not status.exists:
            if status.required:
                problems.append(f"missing required file {status.path}")
            continue
        if status.checksum_ok is False:
            msg = (
                f"{status.filename}: SHA-256 {status.sha256} does not match the pinned "
                f"published version {status.expected_sha256}"
            )
            if allow_checksum_mismatch:
                log.warning("%s (accepted because --allow-checksum-mismatch was given)", msg)
            else:
                problems.append(msg)
                continue
        paths[status.key] = Path(status.path)
    if problems:
        raise DatasetError(
            "Dataset verification failed:\n  - "
            + "\n  - ".join(problems)
            + "\nRun 'python -m xanthomonas_ml download-data' or place the authentic files in "
            + cfg.raw_dir.as_posix()
            + " (see data/raw/README.md)."
        )
    return paths


def _download_one(url: str, dest: Path, timeout: float, retries: int) -> None:
    """Stream ``url`` to ``dest`` via a temporary ``.part`` file (atomic rename)."""
    import requests  # local import keeps offline commands free of network modules

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                written = 0
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        handle.write(chunk)
                        written += len(chunk)
            # Detect truncated transfers when the server reports a length.
            if expected is not None and int(expected) != written:
                raise DatasetError(f"incomplete download: {written} of {expected} bytes")
            os.replace(tmp, dest)
            return
        except Exception as exc:  # network errors, HTTP errors, truncation
            last_error = exc
            log.warning("Download attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            tmp.unlink(missing_ok=True)
    raise DatasetError(f"Could not download {url}: {last_error}")


def download_dataset(
    cfg: Config, force: bool = False, timeout: float = 60.0, retries: int = 3
) -> dict[str, Any]:
    """Download (if needed) and verify every configured file, then write a manifest.

    Args:
        cfg: Loaded configuration.
        force: Re-download even if a verified local copy exists.
        timeout: Per-request timeout in seconds.
        retries: Attempts per file before giving up.

    Returns:
        The manifest dictionary that was written to ``data/raw/MANIFEST.json``.

    Raises:
        DatasetError: if a required file cannot be obtained or verified.
    """
    raw_dir = ensure_dir(cfg.raw_dir)
    ds = cfg["dataset"]
    entries: list[dict[str, Any]] = []
    for spec in dataset_file_specs(cfg):
        dest = raw_dir / spec.filename
        action = "reused"
        # Download only when forced, when the file is missing, or when the local copy fails its checksum.
        if force or not dest.is_file() or (spec.sha256 and sha256_file(dest) != spec.sha256):
            if dest.is_file() and not force:
                log.warning("%s exists but fails checksum verification; re-downloading", dest.name)
            log.info("Downloading %s", spec.url)
            try:
                _download_one(spec.url, dest, timeout=timeout, retries=retries)
            except DatasetError:
                # A failed optional file is skipped; a failed required file stops the run.
                if spec.required:
                    raise
                log.warning("Optional file %s could not be downloaded; continuing", spec.filename)
                continue
            action = "downloaded"
        else:
            log.info("Verified local copy of %s found; not downloading again", spec.filename)

        # Verify the file whether it was downloaded or reused; a mismatch is quarantined and stops the run.
        digest = sha256_file(dest)
        if spec.sha256 and digest != spec.sha256:
            # Keep the file for inspection but rename it so it can never be used by mistake.
            bad = dest.with_suffix(dest.suffix + ".checksum-mismatch")
            os.replace(dest, bad)
            raise DatasetError(
                f"{spec.filename}: checksum {digest} != expected {spec.sha256}. "
                f"The file was moved to {bad.name} for inspection."
            )
        entries.append(
            {
                **asdict(spec),
                "sha256_verified": digest,
                "size_bytes": dest.stat().st_size,
                "local_path": f"{cfg['paths']['raw_data']}/{spec.filename}",
                "action": action,
            }
        )

    manifest = {
        "dataset_name": ds.get("name"),
        "source_repository": ds.get("repository"),
        "source_commit": ds.get("commit"),
        "publication": "te Molder D, Poncheewin W, Schaap PJ, Koehorst JJ. Machine learning approaches "
        "to predict the plant-associated phenotype of Xanthomonas strains. BMC Genomics. "
        "2021;22:848. doi:10.1186/s12864-021-08093-0",
        "related_dataset_doi": "10.4121/14546625 (annotated genomes in RDF/HDT; not required here)",
        "retrieved_utc": utc_now_iso(),
        "license_note": LICENSE_NOTE,
        "files": entries,
    }
    write_json(raw_dir / MANIFEST_NAME, manifest)
    log.info("Dataset verified; manifest written to %s", (raw_dir / MANIFEST_NAME).as_posix())
    return manifest
