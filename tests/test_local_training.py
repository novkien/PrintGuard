"""Disposable local review storage and retry/cleanup checks."""

from __future__ import annotations

import pytest
from printguard.server.local_training import LocalTrainingStore


def test_save_retry_dedup_and_consume(tmp_path):
    store = LocalTrainingStore(tmp_path)
    store.save("review-a", [(b"image-one", "good"), (b"image-one", "good"), (b"image-two", "failure")])
    store.save("review-a", [(b"image-one", "good")])
    store.save("review-b", [(b"image-one", "good")])
    assert store.status()["pending"] == 2
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 2
    rows = store.claim()
    assert len(rows) == 2
    assert store.claim() == []
    store.finish(rows)
    assert store.status()["pending"] == 0
    assert set(store.status()["completed_reviews"]) == {"review-a", "review-b"}
    assert list((tmp_path / "images").glob("*.jpg")) == []
    store.save("review-a", [(b"image-one", "good")])
    assert store.status()["pending"] == 0
    assert list((tmp_path / "images").glob("*.jpg")) == []


def test_labels_conflict_and_interrupted_batch(tmp_path):
    store = LocalTrainingStore(tmp_path)
    with pytest.raises(ValueError, match="conflicting"):
        store.save("review-a", [(b"one", "good"), (b"one", "failure")])
    assert store.status()["pending"] == 0
    store.save("review-a", [(b"one", "good")])
    store.claim()
    with pytest.raises(ValueError, match="already training"):
        store.save("review-b", [(b"one", "failure")])
    store.release()
    store.save("review-a", [(b"one", "failure")])
    assert store.status()["labels"] == {"failure": 1}


def test_queue_cap_rejects_without_partial_save(tmp_path):
    store = LocalTrainingStore(tmp_path)
    with pytest.raises(ValueError, match="full"):
        store.save("review-a", [(str(i).encode(), "good") for i in range(501)])
    assert store.status()["pending"] == 0
    assert not list((tmp_path / "images").glob("*.jpg"))


def test_interrupted_file_cleanup_preserves_pending_samples(tmp_path):
    store = LocalTrainingStore(tmp_path)
    store.save("review", [(b"one", "good")])
    (tmp_path / "images" / "orphan.jpg").write_bytes(b"interrupted write")
    (tmp_path / "images" / "partial.tmp").write_bytes(b"partial")
    store.clean_orphans()
    assert len(list((tmp_path / "images").iterdir())) == 1
    store.finish(store.claim())
    (tmp_path / "images" / "leftover.jpg").write_bytes(b"interrupted deletion")
    store.clean_orphans()
    assert not list((tmp_path / "images").iterdir())


def test_rollback_does_not_overwrite_a_newer_published_model(tmp_path):
    store = LocalTrainingStore(tmp_path)
    store.write_json("config.json", {"mode": "auto", "benchmark_seconds": 30})
    store.write_json("active.json", {"version": "model-2", "previous": "model-1"})
    store.reject_active("model-1", "stale error")
    assert store.status()["active_version"] == "model-2"
    store.reject_active("model-2", "bad ONNX")
    assert store.status()["active_version"] == "model-1"
    assert store.status()["mode"] == "store"
    assert store.status()["benchmark_seconds"] == 30
