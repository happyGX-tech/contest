from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from policies.red.skills.evasion_ppo.trajectory_dataset import (
    archive_trace,
    inspect_trace,
)


def write_trace(path: Path, *, real_outcome: bool) -> None:
    payload = {
        "format": "local-avoid-runtime-trajectory-v3",
        "entity_id": 101,
        "missile_type_slot": 1,
        "terminal": "arrived" if real_outcome else "arrived_proxy",
        "steps": [
            {
                "observation": [0.0, 1.0],
                "action": 1,
                "action_mask": [1, 1, 1],
            }
        ],
    }
    if real_outcome:
        payload["terminal_source"] = "mission_outcome_tracker"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


class FullSimTrajectoryDatasetTests(unittest.TestCase):
    def test_real_outcome_trace_is_content_addressed_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            trace = Path(directory) / "trace.jsonl.gz"
            write_trace(trace, real_outcome=True)

            first = archive_trace(
                root,
                trace,
                metadata={"run_id": "run-1", "seed": 1},
            )
            second = archive_trace(
                root,
                trace,
                metadata={"run_id": "run-1", "seed": 1},
            )

            self.assertTrue(first["usable_for_outcome_training"])
            self.assertEqual(first["trace_sha256"], second["trace_sha256"])
            self.assertTrue((root / first["object_path"]).is_file())
            record = json.loads(
                (root / "records" / f"{first['trace_sha256']}.json").read_text()
            )
            self.assertEqual(len(record["sources"]), 1)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["objects"], 1)
            self.assertEqual(manifest["usable_objects"], 1)
            self.assertEqual(manifest["trajectories"], 1)

    def test_proxy_or_timeout_trace_is_retained_but_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            trace = Path(directory) / "timeout.jsonl.gz"
            write_trace(trace, real_outcome=False)

            inspection = inspect_trace(trace)
            archived = archive_trace(
                root,
                trace,
                metadata={"run_id": "run-timeout", "seed": 2},
                source_error="timeout",
            )

            self.assertFalse(inspection["all_real_outcomes"])
            self.assertFalse(archived["usable_for_outcome_training"])
            self.assertTrue((root / archived["object_path"]).is_file())
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["quarantined_objects"], 1)


if __name__ == "__main__":
    unittest.main()
