"""Fine-tunes reviewed batches on CPU and publishes validated model bundles."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import re
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torchvision.models.shufflenetv2 import InvertedResidual, ShuffleNetV2

from printguard.engine import vision
from printguard.server.local_training import LocalTrainingStore


class CheckpointContainer(nn.Module):
    """Loads only the reviewed upstream checkpoint's container state."""


def load_encoder(base: Path, active: Path | None) -> nn.Module:
    """Loads the verified original model and optional tensor-only trained weights."""
    checkpoint = base / "base.pt"
    expected = "c5df7b534ec38e19914a0e6af3e23735a5734b3c051467a40cf8cd3f1a957817"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("Training checkpoint checksum does not match the verified deployed baseline")
    safe = [(CheckpointContainer, "protonets.models.few_shot.Protonet"), ShuffleNetV2, InvertedResidual,
            nn.ReLU, nn.MaxPool2d, nn.Conv2d, nn.BatchNorm2d, nn.Dropout2d, nn.Flatten, nn.Sequential]
    with torch.serialization.safe_globals(safe):
        encoder = torch.load(checkpoint, map_location="cpu", weights_only=True).encoder
    if active is not None:
        encoder.load_state_dict(torch.load(active / "encoder_state.pt", map_location="cpu", weights_only=True))
    return encoder.eval()


def candidate(store: LocalTrainingStore, rows: list[dict[str, Any]], base: Path,
              output: Path, epochs: int, benchmark: bool = False, device: str = "cpu") -> dict[str, Any]:
    """Measures or fine-tunes a labeled batch without changing the active model.

    Validation requires both human classes outside the optimization subset.
    Benchmark labels are synthetic and are never eligible for publication.
    """
    started = time.monotonic()
    torch.manual_seed(47)
    torch.set_num_threads(1)
    pointer = store.read_json("active.json", {})
    if pointer.get("version") and not re.fullmatch(r"model-[0-9]+", pointer["version"]):
        raise ValueError("Invalid active model version")
    active = store.root.parent / "local-models" / pointer["version"] if pointer.get("version") else None
    source = active or base
    model = load_encoder(base, active).to(device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.conv5.parameters():
        parameter.requires_grad_(True)
    prototypes = json.loads((source / "prototypes.json").read_text())["prototypes"]
    metadata = json.loads((base / "metadata.json").read_text())
    assets = vision.assets_from_dicts(metadata, prototypes)
    tensors = [vision.preprocess(np.asarray(Image.open(store.root / "images" / (r["sha"] + ".jpg")).convert("RGB")), assets)[0] for r in rows]
    x = torch.from_numpy(np.stack(tensors)).to(device)
    y = torch.tensor([r["label"] == "failure" for r in rows], dtype=torch.long, device=device)
    proto = torch.tensor(np.stack([prototypes["success"], prototypes["failure"]]), dtype=torch.float32, device=device)
    validation: list[int] = []
    for label in (0, 1):
        indices = [i for i in range(len(rows)) if int(y[i]) == label]
        if len(indices) >= 3:
            validation.extend(indices[-max(1, len(indices) // 4):])
    training = [i for i in range(len(rows)) if i not in validation]
    if not training:
        raise ValueError("No training frames remain")
    with torch.no_grad():
        reference = teacher(x)
        old_logits = -torch.cdist(reference, proto).square()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4, momentum=0.9)
    for epoch in range(epochs):
        if time.monotonic() - started > 1200:
            raise TimeoutError("Pi training exceeded the owner's 20-minute budget")
        for begin in range(0, len(training), 4):
            if time.monotonic() - started > 1200:
                raise TimeoutError("CPU training exceeded the owner's 20-minute budget")
            ids = training[begin:begin + 4]
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(x[ids])
            logits = -torch.cdist(embeddings, proto).square()
            loss = nn.functional.cross_entropy(logits, y[ids]) + 10 * nn.functional.mse_loss(embeddings, reference[ids])
            if not bool(torch.isfinite(loss)):
                raise ValueError("Training produced a non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        store.write_json("status.json", {"state": "training", "message": f"Epoch {epoch + 1}/{epochs}", "images": len(rows)})
    with torch.no_grad():
        features = model(x)
        updated = proto.clone()
        for label in (0, 1):
            ids = [i for i in training if int(y[i]) == label]
            if ids:
                updated[label] = .9 * proto[label] + .1 * features[ids].mean(dim=0)
        new_logits = -torch.cdist(features, updated).square()
    output.mkdir(parents=True, exist_ok=True)
    model.cpu()
    exported = output / "encoder_float32.onnx"
    torch.onnx.export(model, x[:1].cpu(), str(exported), input_names=["input"], output_names=["embedding"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(exported))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(exported), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": x[:1].cpu().numpy()})[0]
    with torch.no_grad():
        expected = model(x[:1].cpu()).numpy()
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-4 or not np.isfinite(actual).all():
        raise ValueError("Exported model failed numerical validation")
    checks: dict[str, Any] = {"export_error": error, "holdout_images": len(validation), "holdout_both_classes": len({int(y[i]) for i in validation}) == 2}
    accepted = False
    if checks["holdout_both_classes"] and not benchmark:
        old_loss = float(nn.functional.cross_entropy(old_logits[validation], y[validation]))
        new_loss = float(nn.functional.cross_entropy(new_logits[validation], y[validation]))
        old_predictions = old_logits.argmax(dim=1)
        new_predictions = new_logits.argmax(dim=1)
        class_passes = []
        for label in (0, 1):
            ids = [i for i in validation if int(y[i]) == label]
            class_passes.append(int((new_predictions[ids] == y[ids]).sum()) >= int((old_predictions[ids] == y[ids]).sum()))
        accepted = new_loss <= old_loss and all(class_passes)
        checks.update(old_loss=old_loss, new_loss=new_loss, class_accuracy_not_worse=all(class_passes))
    torch.save(model.state_dict(), output / "encoder_state.pt")
    (output / "prototypes.json").write_text(json.dumps({"embedding_dim": 1024, "prototypes": {name: updated[i].cpu().tolist() for i, name in enumerate(("success", "failure"))}}))
    (output / "metadata.json").write_text(json.dumps(metadata))
    result = {"seconds": time.monotonic() - started, "images": len(rows), "epochs": epochs, "device": device,
              "accepted": accepted, "benchmark": benchmark, "checks": checks,
              "base_version": pointer.get("version"), "checkpoint": "verified std-cnn"}
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result


def run(store: LocalTrainingStore, base: Path, epochs: int, device: str) -> None:
    """Trains one queued snapshot, retains rollback, then deletes consumed images."""
    rows = store.claim()
    if not rows:
        return
    if any(sum(row["label"] == label for row in rows) < 3 for label in ("good", "failure")):
        store.release()
        store.write_json("status.json", {"state": "waiting", "message": "Saved locally; need at least 3 good and 3 failure images for validation"})
        return
    versions = store.root.parent / "local-models"
    version = "model-" + str(time.time_ns())
    output = versions / version
    published = False
    try:
        report = candidate(store, rows, base, output, epochs, device=device)
        if report["accepted"]:
            current = store.read_json("active.json", {})
            if current.get("version") != report["base_version"]:
                raise ValueError("Active model changed during training; refusing a stale replacement")
            (output / "consumed.json").write_text(json.dumps([row["sha"] for row in rows]))
            store.write_json("active.json", {"version": version, "previous": current.get("version")})
            published = True
            keep = {version, current.get("version")}
            for path in versions.glob("model-*"):
                if path.name not in keep:
                    shutil.rmtree(path)
            message = "Validated model activated; training images deleted"
        else:
            shutil.rmtree(output)
            message = "Model kept: validation needs both labels or did not improve; consumed images deleted"
        store.finish(rows)
        store.write_json("status.json", {"state": "completed", "message": message, "last_run": report})
        print(json.dumps({"event": "training_completed", "images": len(rows), "accepted": report["accepted"], "seconds": report.get("seconds")}), flush=True)
    except Exception as exc:
        if not published:
            shutil.rmtree(output, ignore_errors=True)
            store.release()
        else:
            # Publication is irreversible here: never delete the active bundle.
            # Finish is idempotent and orphan cleanup recovers interrupted deletes.
            try:
                store.finish(rows)
            except Exception:
                pass
        store.write_json("status.json", {"state": "error", "message": str(exc)[:400]})


def recover(store: LocalTrainingStore) -> None:
    """Finishes published batches after a crash and removes unreferenced candidates."""
    pointer = store.read_json("active.json", {})
    keep = {name for name in (pointer.get("version"), pointer.get("previous"))
            if isinstance(name, str) and re.fullmatch(r"model-[0-9]+", name)}
    versions = store.root.parent / "local-models"
    if pointer.get("version") in keep:
        manifest = versions / pointer["version"] / "consumed.json"
        if manifest.exists():
            consumed = set(json.loads(manifest.read_text()))
            with store.connect() as db:
                rows = [dict(row) for row in db.execute("SELECT * FROM samples WHERE status='training'") if row["sha"] in consumed]
            if rows:
                store.finish(rows)
    store.release()
    store.clean_orphans()
    for path in versions.glob("model-*"):
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path)


def main() -> None:
    """Runs the worker, an explicit queue run, or a non-publishing timing benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    store = LocalTrainingStore(args.root)
    with (store.root / "worker.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover(store)
        if args.benchmark:
            with store.connect() as db:
                rows = [dict(r) for r in db.execute("SELECT * FROM samples WHERE status='pending' ORDER BY created LIMIT 20")]
            if len(rows) != 20:
                raise ValueError("The benchmark requires exactly 20 disposable images")
            output = store.root / "benchmark-output"
            try:
                report = candidate(store, rows, args.base, output, args.epochs, benchmark=True, device=args.device)
                store.write_json("benchmark.json", report)
                print(json.dumps(report), flush=True)
            finally:
                shutil.rmtree(output, ignore_errors=True)
            return
        if args.once:
            run(store, args.base, args.epochs, args.device)
            return
        print("Local CPU training worker ready", flush=True)
        while True:
            store.clean_orphans()
            config = store.read_json("config.json", {"mode": "store"})
            request = store.root / "request.json"
            wanted = request.exists()
            if wanted:
                request.unlink()
            if config["mode"] == "auto" or wanted:
                status = store.status()
                enough_labels = all(status["labels"].get(label, 0) >= 3 for label in ("good", "failure"))
                if status["pending"] and (status["state"] != "waiting" or enough_labels or wanted):
                    run(store, args.base, args.epochs, args.device)
                    if store.status()["state"] == "error":
                        store.write_json("config.json", {**config, "mode": "store"})
            time.sleep(5)


if __name__ == "__main__":
    main()
