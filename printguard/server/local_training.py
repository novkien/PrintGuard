"""Bounded, content-addressed local review storage shared with the training worker."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any


class LocalTrainingStore:
    """Keeps labeled images once, with small receipts after their pixels are consumed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS samples (
                    sha TEXT PRIMARY KEY, label TEXT NOT NULL, status TEXT NOT NULL,
                    created REAL NOT NULL, review_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_items (
                    review_id TEXT NOT NULL, sha TEXT NOT NULL, PRIMARY KEY(review_id,sha));
                CREATE TABLE IF NOT EXISTS receipts (review_id TEXT PRIMARY KEY, created REAL NOT NULL);
            """)

    def connect(self) -> sqlite3.Connection:
        """Opens a bounded-wait database connection."""
        db = sqlite3.connect(self.root / "queue.sqlite", timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def read_json(self, name: str, default: dict[str, Any]) -> dict[str, Any]:
        """Reads one worker-owned status document."""
        path = self.root / name
        return json.loads(path.read_text()) if path.exists() else default

    def write_json(self, name: str, value: dict[str, Any]) -> None:
        """Atomically replaces one worker-owned document."""
        path = self.root / name
        temporary = path.with_suffix(".tmp-" + uuid.uuid4().hex)
        temporary.write_text(json.dumps(value))
        temporary.replace(path)

    def status(self) -> dict[str, Any]:
        """Reports queue size, measured mode and the last job without exposing images."""
        config = self.read_json("config.json", {"mode": "store"})
        job = self.read_json("status.json", {"state": "idle", "message": "CPU benchmark pending"})
        with self.connect() as db:
            count = db.execute("SELECT count(*) FROM samples WHERE status!='done'").fetchone()[0]
            labels = dict(db.execute("SELECT label,count(*) FROM samples WHERE status!='done' GROUP BY label").fetchall())
            completed = [r[0] for r in db.execute("SELECT review_id FROM receipts ORDER BY created DESC LIMIT 2000")]
        return {**job, "mode": config["mode"], "pending": count, "labels": labels,
                "benchmark_seconds": config.get("benchmark_seconds"), "completed_reviews": completed,
                "active_version": self.read_json("active.json", {}).get("version")}

    def save(self, review_id: str, frames: list[tuple[bytes, str]]) -> dict[str, Any]:
        """Commits a reviewed batch locally without contacting the upstream inbox.

        Args:
            review_id: The native review identity, used for retry deduplication.
            frames: JPEG bytes and human labels, good or failure.

        Raises:
            ValueError: A label conflicts with an active or consumed sample, or storage is full.
        """
        if not frames or any(label not in ("good", "failure") for _, label in frames):
            raise ValueError("Choose at least one labeled frame")
        unique: dict[str, tuple[bytes, str]] = {}
        for jpeg, label in frames:
            sha = hashlib.sha256(jpeg).hexdigest()
            if sha in unique and unique[sha][1] != label:
                raise ValueError("The same image has conflicting labels")
            unique[sha] = (jpeg, label)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = {r["sha"]: r for r in db.execute("SELECT * FROM samples")}
            pending = sum(r["status"] != "done" for r in existing.values())
            new = [sha for sha in unique if sha not in existing]
            current_bytes = 0
            for path in (self.root / "images").glob("*.jpg"):
                try:
                    current_bytes += path.stat().st_size
                except FileNotFoundError:
                    pass  # A committed finish can still be deleting its pixels.
            if pending + len(new) > 500 or current_bytes + sum(len(unique[s][0]) for s in new) > 200 * 1024 * 1024:
                raise ValueError("Local queue is full (500 images / 200 MB); train or clear the pending batch first")
            for sha, (_, label) in unique.items():
                row = existing.get(sha)
                if row and row["label"] != label and row["status"] != "pending":
                    raise ValueError("This image is already training or consumed with another label")
            for sha, (jpeg, label) in unique.items():
                row = existing.get(sha)
                if row is None:
                    path = self.root / "images" / (sha + ".jpg")
                    if not path.exists():
                        temporary = path.with_suffix(".tmp")
                        temporary.write_bytes(jpeg)
                        temporary.replace(path)
                    db.execute("INSERT INTO samples VALUES (?,?,?,?,?)", (sha, label, "pending", time.time(), review_id))
                elif row["status"] == "pending":
                    db.execute("UPDATE samples SET label=? WHERE sha=?", (label, sha))
                db.execute("INSERT OR IGNORE INTO review_items VALUES (?,?)", (review_id, sha))
            self._receipts(db)
            self._prune_receipts(db)
        return self.status()

    def request(self) -> dict[str, Any]:
        """Requests a serialized worker run; repeated clicks do not create more jobs."""
        self.write_json("request.json", {"requested": True})
        return self.status()

    def claim(self) -> list[dict[str, Any]]:
        """Claims the immutable pending snapshot for the single worker."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = [dict(r) for r in db.execute("SELECT * FROM samples WHERE status='pending' ORDER BY created")]
            db.execute("UPDATE samples SET status='training' WHERE status='pending'")
        return rows

    def finish(self, rows: list[dict[str, Any]]) -> None:
        """Deletes consumed JPEGs and keeps bounded hash receipts to suppress retries."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in rows:
                db.execute("UPDATE samples SET status='done' WHERE sha=?", (row["sha"],))
            self._receipts(db)
            self._prune_receipts(db)
            db.execute("DELETE FROM samples WHERE sha IN (SELECT sha FROM samples WHERE status='done' ORDER BY created DESC LIMIT -1 OFFSET 50000)")
        for row in rows:
            (self.root / "images" / (row["sha"] + ".jpg")).unlink(missing_ok=True)

    def release(self) -> None:
        """Returns an interrupted job to the pending queue for an explicit retry."""
        with self.connect() as db:
            db.execute("UPDATE samples SET status='pending' WHERE status='training'")

    def clean_orphans(self) -> None:
        """Recovers interrupted pixel writes/deletes under the same lock as save."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            keep = {r[0] + ".jpg" for r in db.execute("SELECT sha FROM samples WHERE status!='done'")}
            for path in (self.root / "images").iterdir():
                if path.is_file() and path.name not in keep:
                    path.unlink(missing_ok=True)

    def reject_active(self, version: str, reason: str) -> None:
        """Reverts a bundle the hub could not load, without overwriting a newer one."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            pointer = self.read_json("active.json", {})
            if pointer.get("version") != version:
                return
            previous = pointer.get("previous")
            if not isinstance(previous, str) or not re.fullmatch(r"model-[0-9]+", previous):
                previous = None
            self.write_json("active.json", {"version": previous, "previous": None})
            config = self.read_json("config.json", {})
            self.write_json("config.json", {**config, "mode": "store"})
            self.write_json("status.json", {"state": "error", "message": "Model load failed; rolled back: " + reason[:300]})

    def _receipts(self, db: sqlite3.Connection) -> None:
        for row in db.execute("""SELECT review_id FROM review_items GROUP BY review_id
            HAVING NOT EXISTS (SELECT 1 FROM review_items i JOIN samples s ON i.sha=s.sha
            WHERE i.review_id=review_items.review_id AND s.status!='done')"""):
            db.execute("INSERT OR IGNORE INTO receipts VALUES (?,?)", (row[0], time.time()))

    def _prune_receipts(self, db: sqlite3.Connection) -> None:
        db.execute("DELETE FROM review_items WHERE review_id IN (SELECT review_id FROM receipts)")
        db.execute("DELETE FROM receipts WHERE review_id NOT IN (SELECT review_id FROM receipts ORDER BY created DESC LIMIT 2000)")
