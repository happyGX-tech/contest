"""Content-addressed storage for complete-simulator local-avoid trajectories.

The dataset deliberately keeps the exact gzip bytes emitted by the simulator.
Usability is recorded separately: completed workers with MissionOutcomeTracker
labels may be used for outcome-conditioned training, while timeouts and proxy-
labelled traces are retained but quarantined.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping


DATASET_FORMAT = "local-avoid-full-sim-dataset-v1"
ENTRY_FORMAT = "local-avoid-full-sim-dataset-entry-v1"
SUPPORTED_TRACE_FORMATS = {
    "local-avoid-runtime-trajectory-v1",
    "local-avoid-runtime-trajectory-v2",
    "local-avoid-runtime-trajectory-v3",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def dataset_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".archive.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def inspect_trace(path: str | Path) -> dict[str, Any]:
    """Inspect a trace without changing it and return fail-closed metadata."""

    trace_path = Path(path)
    terminals: Counter[str] = Counter()
    missile_types: Counter[str] = Counter()
    trace_formats: Counter[str] = Counter()
    errors: list[str] = []
    trajectory_count = 0
    step_count = 0
    real_outcome_count = 0
    try:
        with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"invalid_json_line_{line_number}:{error.msg}")
                    continue
                trajectory_count += 1
                trace_format = str(payload.get("format", "missing"))
                trace_formats[trace_format] += 1
                if trace_format not in SUPPORTED_TRACE_FORMATS:
                    errors.append(
                        f"unsupported_format_line_{line_number}:{trace_format}"
                    )
                steps = payload.get("steps")
                if not isinstance(steps, list):
                    errors.append(f"missing_steps_line_{line_number}")
                else:
                    step_count += len(steps)
                terminal = str(payload.get("terminal", "missing"))
                terminals[terminal] += 1
                missile_types[str(payload.get("missile_type_slot", "missing"))] += 1
                if payload.get("terminal_source") == "mission_outcome_tracker":
                    real_outcome_count += 1
    except (OSError, EOFError, UnicodeDecodeError) as error:
        errors.append(f"unreadable_gzip:{type(error).__name__}:{error}")

    if trajectory_count == 0:
        errors.append("empty_trace")
    all_real_outcomes = (
        trajectory_count > 0 and real_outcome_count == trajectory_count
    )
    if not all_real_outcomes:
        errors.append(
            f"missing_real_outcomes:{trajectory_count - real_outcome_count}"
        )
    return {
        "valid_gzip_jsonl": not any(
            value.startswith(("invalid_json", "unreadable_gzip", "empty_trace"))
            for value in errors
        ),
        "all_real_outcomes": all_real_outcomes,
        "trajectory_count": trajectory_count,
        "step_count": step_count,
        "terminal_counts": dict(sorted(terminals.items())),
        "missile_type_counts": dict(sorted(missile_types.items())),
        "trace_formats": dict(sorted(trace_formats.items())),
        "errors": errors,
    }


def initialize_dataset(root: str | Path) -> Path:
    dataset_root = Path(root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    descriptor = dataset_root / "dataset.json"
    if not descriptor.exists():
        atomic_json(
            descriptor,
            {
                "format": DATASET_FORMAT,
                "created_at": utc_now(),
                "storage": "content_addressed_exact_gzip",
                "training_rule": (
                    "Only records with usable_for_outcome_training=true may be "
                    "used as real-outcome training data."
                ),
            },
        )
    else:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        if payload.get("format") != DATASET_FORMAT:
            raise ValueError(f"unsupported dataset format in {descriptor}")
    return dataset_root


def _copy_content_addressed(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)


def _rebuild_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted((root / "records").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "format": DATASET_FORMAT,
        "updated_at": utc_now(),
        "objects": len(records),
        "source_references": sum(len(item.get("sources", [])) for item in records),
        "usable_objects": sum(
            bool(item.get("usable_for_outcome_training")) for item in records
        ),
        "quarantined_objects": sum(
            not bool(item.get("usable_for_outcome_training")) for item in records
        ),
        "trajectories": sum(
            int(item.get("inspection", {}).get("trajectory_count", 0))
            for item in records
        ),
        "steps": sum(
            int(item.get("inspection", {}).get("step_count", 0))
            for item in records
        ),
        "record_files": [
            str(Path("records") / f"{item['trace_sha256']}.json")
            for item in records
        ],
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


def archive_policy(root: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    dataset_root = initialize_dataset(root)
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    destination = dataset_root / "policies" / f"{digest}.pt"
    with dataset_lock(dataset_root):
        _copy_content_addressed(source, destination)
        if sha256_file(destination) != digest:
            raise IOError(f"archived checkpoint hash mismatch: {destination}")
    return {
        "policy_sha256": digest,
        "policy_object": str(destination.relative_to(dataset_root)),
        "source_checkpoint": str(source),
    }


def archive_trace(
    root: str | Path,
    trace: str | Path,
    *,
    metadata: Mapping[str, Any],
    source_error: str | None = None,
) -> dict[str, Any]:
    """Archive exact trace bytes and an independently validated record."""

    dataset_root = initialize_dataset(root)
    source = Path(trace).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    inspection = inspect_trace(source)
    usable = (
        source_error is None
        and inspection["valid_gzip_jsonl"]
        and inspection["all_real_outcomes"]
    )
    object_path = Path("objects") / "sha256" / digest[:2] / f"{digest}.jsonl.gz"
    destination = dataset_root / object_path
    source_record = {
        **dict(metadata),
        "source_path": str(source),
        "source_error": source_error,
        "archived_at": utc_now(),
    }
    record_path = dataset_root / "records" / f"{digest}.json"
    with dataset_lock(dataset_root):
        _copy_content_addressed(source, destination)
        if sha256_file(destination) != digest:
            raise IOError(f"archived trace hash mismatch: {destination}")
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            identity = (
                source_record.get("source_path"),
                source_record.get("run_id"),
                source_record.get("seed"),
            )
            known = {
                (item.get("source_path"), item.get("run_id"), item.get("seed"))
                for item in record.get("sources", [])
            }
            if identity not in known:
                record.setdefault("sources", []).append(source_record)
        else:
            record = {
                "format": ENTRY_FORMAT,
                "trace_sha256": digest,
                "trace_bytes": source.stat().st_size,
                "object_path": str(object_path),
                "usable_for_outcome_training": usable,
                "inspection": inspection,
                "sources": [source_record],
            }
        # A content object is usable if at least one clean, fully labelled source
        # proves its provenance.  A later duplicate timeout cannot quarantine it.
        record["usable_for_outcome_training"] = bool(
            record.get("usable_for_outcome_training") or usable
        )
        atomic_json(record_path, record)
        manifest = _rebuild_manifest(dataset_root)
    return {**record, "manifest": manifest}


def _workers_from_run(run_dir: Path) -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("workers_iteration_*.json")):
        workers.extend(json.loads(path.read_text(encoding="utf-8")).get("workers", []))
    if workers:
        return workers
    history_path = run_dir / "training_history.jsonl"
    if not history_path.is_file():
        return []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if payload.get("event") == "worker":
                workers.append(payload)
    return workers


def archive_training_run(root: str | Path, run: str | Path) -> dict[str, Any]:
    """Import every extant trace from a full-simulator training run."""

    dataset_root = initialize_dataset(root)
    run_dir = Path(run).resolve()
    protocol_path = run_dir / "protocol.json"
    if not protocol_path.is_file():
        protocol_path = run_dir / "run_status.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"run protocol not found under {run_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    policy_archive: dict[str, Any] = {}
    input_model = protocol.get("input_model")
    if input_model and Path(input_model).is_file():
        policy_archive = archive_policy(dataset_root, input_model)

    imported = []
    missing = []
    for worker in _workers_from_run(run_dir):
        trace_path = Path(str(worker.get("trace_path", "")))
        if not trace_path.is_file():
            missing.append(str(trace_path))
            continue
        policy_sha256 = worker.get("checkpoint_sha256")
        if not policy_sha256 and int(worker.get("iteration", 0)) == 1:
            policy_sha256 = protocol.get("input_sha256")
        metadata = {
            "source_kind": "training_run_import",
            "source_run": str(run_dir),
            "pipeline": protocol.get("pipeline"),
            "run_id": worker.get("run_id"),
            "iteration": worker.get("iteration"),
            "scenario": worker.get("scenario"),
            "seed": worker.get("seed"),
            "blue_policy": protocol.get("blue_policy"),
            "objective_motion": protocol.get("objective_motion"),
            "gate_mode": protocol.get("gate_mode"),
            "policy_sha256": policy_sha256,
            "policy_object": (
                policy_archive.get("policy_object")
                if policy_sha256 == policy_archive.get("policy_sha256")
                else None
            ),
        }
        imported.append(
            archive_trace(
                dataset_root,
                trace_path,
                metadata=metadata,
                source_error=worker.get("error"),
            )
        )
    return {
        "dataset_root": str(dataset_root),
        "source_run": str(run_dir),
        "imported_objects": len(imported),
        "usable_objects": sum(
            bool(item.get("usable_for_outcome_training")) for item in imported
        ),
        "quarantined_objects": sum(
            not bool(item.get("usable_for_outcome_training")) for item in imported
        ),
        "missing_traces": missing,
        "manifest": json.loads(
            (dataset_root / "manifest.json").read_text(encoding="utf-8")
        ),
    }


def verify_dataset(root: str | Path) -> dict[str, Any]:
    """Verify every content-addressed trajectory and policy object."""

    dataset_root = initialize_dataset(root)
    errors: list[str] = []
    verified_records = 0
    for record_path in sorted((dataset_root / "records").glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        digest = str(record.get("trace_sha256", ""))
        object_path = dataset_root / str(record.get("object_path", ""))
        if not object_path.is_file():
            errors.append(f"missing_object:{record_path.name}:{object_path}")
            continue
        actual = sha256_file(object_path)
        if actual != digest:
            errors.append(
                f"trace_hash_mismatch:{record_path.name}:{digest}:{actual}"
            )
            continue
        current_inspection = inspect_trace(object_path)
        if current_inspection != record.get("inspection"):
            errors.append(f"inspection_mismatch:{record_path.name}")
            continue
        verified_records += 1
    verified_policies = 0
    for policy_path in sorted((dataset_root / "policies").glob("*.pt")):
        if sha256_file(policy_path) != policy_path.stem:
            errors.append(f"policy_hash_mismatch:{policy_path.name}")
        else:
            verified_policies += 1
    return {
        "dataset_root": str(dataset_root),
        "valid": not errors,
        "verified_records": verified_records,
        "verified_policies": verified_policies,
        "errors": errors,
    }
