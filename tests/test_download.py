"""Tests for dataset acquisition and integrity checks (network is always mocked)."""

from __future__ import annotations

import hashlib

import pytest

from xanthomonas_ml import dataset_download as dd


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200, declared_length: int | None = None):
        self.payload, self.status = payload, status
        self.headers = {"Content-Length": str(declared_length if declared_length is not None else len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status}")

    def iter_content(self, chunk_size):
        yield self.payload


def _config_with_checksums(cfg, payloads):
    files = {k: dict(v) for k, v in cfg["dataset"]["files"].items()}
    for key, entry in files.items():
        entry["sha256"] = hashlib.sha256(payloads[entry["filename"]]).hexdigest()
    return cfg.with_overrides({"dataset": {"files": files}})


@pytest.fixture
def payloads(synthetic_project):
    return {p.name: p.read_bytes() for p in synthetic_project.raw_dir.glob("*.csv")}


@pytest.fixture
def empty_raw(synthetic_project, payloads):
    # Depends on `payloads` so the file contents are captured before deletion.
    for p in synthetic_project.raw_dir.glob("*.csv"):
        p.unlink()
    return synthetic_project


def _fake_get(payloads, calls):
    def get(url, stream, timeout):
        calls.append(url)
        return FakeResponse(payloads[url.rsplit("/", 1)[-1]])

    return get


def test_download_verifies_and_writes_manifest(monkeypatch, empty_raw, payloads):
    import requests

    cfg = _config_with_checksums(empty_raw, payloads)
    calls: list[str] = []
    monkeypatch.setattr(requests, "get", _fake_get(payloads, calls))
    manifest = dd.download_dataset(cfg)
    assert len(calls) == 3 and all(f["action"] == "downloaded" for f in manifest["files"])
    assert (cfg.raw_dir / dd.MANIFEST_NAME).is_file()
    assert "license_note" in manifest and manifest["source_commit"]
    # Second call reuses the verified copies without any request.
    calls.clear()
    manifest = dd.download_dataset(cfg)
    assert calls == [] and all(f["action"] == "reused" for f in manifest["files"])


def test_checksum_mismatch_is_rejected(monkeypatch, empty_raw, payloads):
    import requests

    cfg = _config_with_checksums(empty_raw, payloads)
    corrupted = dict(payloads, **{"domain_matrix.csv": b"gca,PF1\nA,1\n"})
    monkeypatch.setattr(requests, "get", _fake_get(corrupted, []))
    with pytest.raises(dd.DatasetError, match="checksum"):
        dd.download_dataset(cfg)
    assert not (cfg.raw_dir / "domain_matrix.csv").exists()
    assert (cfg.raw_dir / "domain_matrix.csv.checksum-mismatch").exists()


def test_truncated_download_detected(monkeypatch, empty_raw, payloads):
    import requests

    cfg = _config_with_checksums(empty_raw, payloads)
    monkeypatch.setattr(requests, "get", lambda url, stream, timeout: FakeResponse(b"abc", declared_length=999))
    with pytest.raises(dd.DatasetError, match="Could not download"):
        dd.download_dataset(cfg, retries=2)
    assert not list(cfg.raw_dir.glob("*.part"))


def test_network_failure_gives_clear_error(monkeypatch, empty_raw):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(dd.DatasetError, match="no route to host"):
        dd.download_dataset(empty_raw, retries=1)


def test_local_mismatch_blocks_unless_allowed(synthetic_project, payloads):
    cfg = _config_with_checksums(synthetic_project, payloads)
    (cfg.raw_dir / "genome_information.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(dd.DatasetError, match="does not match"):
        dd.require_valid_files(cfg)
    paths = dd.require_valid_files(cfg, allow_checksum_mismatch=True)
    assert "genome_information" in paths


def test_check_local_files_offline(synthetic_project, no_network):
    statuses = dd.check_local_files(synthetic_project)
    assert all(s.exists for s in statuses)
    assert all(s.checksum_ok is None for s in statuses)  # synthetic config has no pinned checksums


def test_real_config_pins_checksums():
    from xanthomonas_ml.config import load_config

    cfg = load_config()
    specs = dd.dataset_file_specs(cfg)
    assert all(s.sha256 and len(s.sha256) == 64 for s in specs)
    assert all(cfg["dataset"]["commit"] in s.url for s in specs)
