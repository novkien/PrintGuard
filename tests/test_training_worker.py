"""Disposable worker publication tests, runnable with the optional CPU training venv."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from printguard.server.local_training import LocalTrainingStore


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional CPU training dependencies are not installed")
class WorkerLifecycle(unittest.TestCase):
    def setUp(self):
        from printguard.training import worker
        self.worker = worker
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = LocalTrainingStore(self.root / "local-training")
        self.store.save("review", [(str(i).encode(), "good" if i < 3 else "failure") for i in range(6)])

    def tearDown(self):
        self.directory.cleanup()

    def candidate(self, *args, accepted=True, **kwargs):
        args[3].mkdir(parents=True)
        return {"accepted": accepted, "base_version": None}

    def test_publish_consumes_pixels_and_keeps_one_rollback(self):
        versions = self.root / "local-models"
        (versions / "model-1").mkdir(parents=True)
        with patch.object(self.worker, "candidate", self.candidate):
            self.worker.run(self.store, self.root, 10, "cpu")
        active = self.store.status()["active_version"]
        self.assertTrue((versions / active).is_dir())
        self.assertFalse((versions / "model-1").exists())
        self.assertEqual(self.store.status()["pending"], 0)
        self.assertEqual(list((self.store.root / "images").iterdir()), [])
        self.assertEqual(self.store.status()["completed_reviews"], ["review"])

    def test_rejected_model_does_not_replace_current_and_deletes_pixels(self):
        def rejected(*args, **kwargs):
            return self.candidate(*args, accepted=False, **kwargs)
        with patch.object(self.worker, "candidate", rejected):
            self.worker.run(self.store, self.root, 10, "cpu")
        self.assertIsNone(self.store.status()["active_version"])
        self.assertEqual(self.store.status()["pending"], 0)
        self.assertEqual(list((self.root / "local-models").iterdir()), [])

    def test_error_preserves_pixels_for_retry(self):
        with patch.object(self.worker, "candidate", side_effect=TimeoutError("budget")):
            self.worker.run(self.store, self.root, 10, "cpu")
        self.assertEqual(self.store.status()["state"], "error")
        self.assertEqual(self.store.status()["pending"], 6)
        self.assertEqual(len(list((self.store.root / "images").iterdir())), 6)

    def test_one_class_waits_for_future_review_without_discarding_labels(self):
        self.store = LocalTrainingStore(self.root / "one-class")
        self.store.save("good-print", [(str(i).encode(), "good") for i in range(6)])
        with patch.object(self.worker, "candidate") as train:
            self.worker.run(self.store, self.root, 10, "cpu")
        train.assert_not_called()
        self.assertEqual(self.store.status()["state"], "waiting")
        self.assertEqual(self.store.status()["pending"], 6)

    def test_cleanup_error_never_deletes_the_published_bundle(self):
        with patch.object(self.worker, "candidate", self.candidate), patch.object(self.store, "finish", side_effect=OSError("disk")):
            self.worker.run(self.store, self.root, 10, "cpu")
        active = self.store.status()["active_version"]
        self.assertTrue((self.root / "local-models" / active).is_dir())
        self.assertEqual(self.store.status()["state"], "error")
        self.worker.recover(self.store)
        self.assertEqual(self.store.status()["pending"], 0)
        self.assertEqual(list((self.store.root / "images").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
