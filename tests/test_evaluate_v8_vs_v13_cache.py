from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any, Callable
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_v8_vs_v13_full_sim as evaluator


CHECKPOINT_SHA256 = "a" * 64
SCENARIO_SHA256 = "b" * 64
CASE_INFO_SHA256 = "c" * 64
CODE_IDENTITY = {
    "name": "test-code",
    "algorithm": "sha256",
    "sha256": "d" * 64,
}
OBSERVATION_IDENTITY = {
    "name": "test-observation",
    "algorithm": "sha256",
    "sha256": "e" * 64,
}
RUNTIME_ENVIRONMENT_IDENTITY = {
    "runtime": "docker",
    "docker_image": "registry.example/simulator:test",
    "docker_image_id": "sha256:" + "f" * 64,
    "docker_run_reference": "sha256:" + "f" * 64,
    "docker_repo_digests": [
        "registry.example/simulator@sha256:" + "1" * 64,
    ],
    "mounted_competition_python": {
        "configured_path": "/host/competition/bin/python",
        "resolved_path": "/host/competition/bin/python3.11",
        "container_path": "/opt/competition-env/bin/python",
        "sha256": "2" * 64,
        "size_bytes": 12345,
    },
    "competition_environment_tree": {
        "name": "mounted_competition_conda_environment",
        "algorithm": "sha256-content-and-relative-path",
        "sha256": "3" * 64,
        "root": "/host/competition",
        "file_count": 10,
        "symlink_count": 2,
        "total_file_bytes": 999,
    },
}
HOST_RUNTIME_ENVIRONMENT_IDENTITY = {
    "runtime": "host",
    "python_executable": {
        "configured_path": "/usr/bin/python",
        "resolved_path": "/usr/bin/python3.11",
        "sha256": "4" * 64,
        "size_bytes": 54321,
    },
    "platform": {
        "system": "Linux",
        "release": "test-kernel",
        "machine": "x86_64",
    },
}


def make_task(
    max_steps: int,
    *,
    condition: str = "v8",
) -> evaluator.EvaluationTask:
    return evaluator.EvaluationTask(
        condition=condition,
        checkpoint=f"/checkpoints/{condition}.pt",
        checkpoint_sha256=CHECKPOINT_SHA256,
        gate_mode="none",
        map_id="E01",
        tier="easy",
        scenario="final20/easy/E01",
        blue_policy="b0_fixed_ratio_random",
        seed=11902,
        scenario_sha256=SCENARIO_SHA256,
        scenario_case_info_sha256=CASE_INFO_SHA256,
        max_steps=max_steps,
    )


def identity(
    task: evaluator.EvaluationTask,
    *,
    workers: int = 4,
    timeout_seconds: float = 1800.0,
    kill_grace_seconds: float = 10.0,
) -> dict[str, object]:
    return evaluator.task_protocol_identity(
        task,
        objective_motion="static",
        runtime="docker",
        docker_image="registry.example/simulator:test",
        workers=workers,
        timeout_seconds=timeout_seconds,
        kill_grace_seconds=kill_grace_seconds,
        runtime_environment_identity=RUNTIME_ENVIRONMENT_IDENTITY,
        code_identity=CODE_IDENTITY,
        observation_encoding_identity=OBSERVATION_IDENTITY,
    )


def write_summary(
    path: Path,
    *,
    steps: int = 17,
    termination_reason: str = "environment_done",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    zero_type = {
        "launched": 0,
        "arrived": 0,
        "arrival_rate": 0.0,
        "effective_hits": 0,
        "hit_rate_per_launch": 0.0,
    }
    path.write_text(
        json.dumps(
            {
                "scenario": "E01",
                "policies": {
                    "red": "r9_hierarchical_learning",
                    "red_motion": "local_avoid_s3",
                    "blue": "b0_fixed_ratio_random",
                },
                "outcomes": {
                    "definitions": {},
                    "launched": 0,
                    "engaged": 0,
                    "engaged_resolved": 0,
                    "engaged_censored": 0,
                    "evaded": 0,
                    "intercepted": 0,
                    "evasion_rate": 0.0,
                    "arrived": 0,
                    "arrival_rate": 0.0,
                    "effective_hits": 0,
                    "hit_rate": 0.0,
                    "hit_rate_per_launch": 0.0,
                    "alive_at_end": 0,
                    "arrival_tolerance_m": 500.0,
                    "per_platform": {},
                    "by_type": {
                        "21000": dict(zero_type),
                        "21001": dict(zero_type),
                        "21002": dict(zero_type),
                    },
                },
                "score": {
                    "score": 12.5,
                    "raw_score": 0.125,
                    "K": 0.125,
                    "T": 0.0,
                    "completed": False,
                    "destroyed_ids": [],
                    "destroyed_weight": 0.0,
                    "total_objective_weight": 1.0,
                    "objective_weights": {"1": 1.0},
                    "destruction_steps": {},
                },
                "red": {
                    "platforms_total": 0,
                    "launched": 0,
                    "alive": 0,
                    "lost": 0,
                },
                "blue": {
                    "entities_total": 0,
                    "alive": 0,
                    "destroyed": 0,
                    "initial_health_total": 0.0,
                    "remaining_health_total": 0.0,
                },
                "objectives": [],
                "steps_executed": steps,
                "termination_reason": termination_reason,
            }
        ),
        encoding="utf-8",
    )


def update_summary(
    path: Path, update: Callable[[dict[str, Any]], object]
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "objective_motion": "static",
        "runtime": "docker",
        "docker_image": "registry.example/simulator:test",
        "workers": 4,
        "timeout_seconds": 1800.0,
        "kill_grace_seconds": 10.0,
        "run_prefix": "test",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class EvaluatorProtocolCacheTest(unittest.TestCase):
    def test_matrix_axes_are_nonempty_and_unique(self) -> None:
        self.assertEqual(
            evaluator.parse_unique_csv("E06,H07", label="map"),
            ("E06", "H07"),
        )
        with self.assertRaisesRegex(ValueError, "At least one map"):
            evaluator.parse_unique_csv(",,", label="map")
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            evaluator.parse_unique_csv("11902,11902", label="seed")

    def test_seed_axis_rejects_integer_aliases_and_negative_values(self) -> None:
        self.assertEqual(
            evaluator.parse_unique_nonnegative_int_csv("11902,11903", label="seed"),
            (11902, 11903),
        )
        for raw in ("11902,011902", "11902,+11902"):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError, "Duplicate seed"
            ):
                evaluator.parse_unique_nonnegative_int_csv(raw, label="seed")
        with self.assertRaisesRegex(ValueError, "Negative seed"):
            evaluator.parse_unique_nonnegative_int_csv("-1", label="seed")

    def test_docker_command_uses_inspected_immutable_image_id(self) -> None:
        task = replace(
            make_task(1200),
            checkpoint=str(ROOT / "models" / "test-checkpoint.pt"),
        )
        args = argparse.Namespace(
            runtime="docker",
            objective_motion="static",
            docker_image="registry.example/simulator:test",
            docker_run_reference="sha256:" + "f" * 64,
        )
        command, _container_name = evaluator.simulator_command(args, task, "test-run")
        self.assertIn(args.docker_run_reference, command)
        self.assertNotIn(args.docker_image, command)
        self.assertIn("RED_LEARNING_TRAIN=0", command)

    def test_host_environment_forces_evaluation_mode(self) -> None:
        args = argparse.Namespace(objective_motion="static")
        task = make_task(1200)
        contaminated = {
            "RED_LEARNING_TRAIN": "1",
            "LOCAL_AVOID_TRACE_PATH": "/tmp/trace",
            "BLUE_WAVE_SIZE": "999",
        }
        with mock.patch.dict(os.environ, contaminated, clear=False):
            env = evaluator.simulator_environment(args, task)
        self.assertEqual(env["RED_LEARNING_TRAIN"], "0")
        self.assertEqual(env["BLUE_OBJECTIVE_MOTION"], "static")
        self.assertEqual(env["LOCAL_AVOID_GATE_MODE"], "none")
        self.assertNotIn("LOCAL_AVOID_TRACE_PATH", env)
        self.assertNotIn("BLUE_WAVE_SIZE", env)

    def test_cancelled_matrix_cannot_launch_a_new_process(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        with tempfile.TemporaryFile(mode="w+") as log_file, mock.patch.object(
            evaluator.subprocess, "Popen"
        ) as popen:
            with self.assertRaises(evaluator.EvaluationCancelled):
                evaluator.start_process_unless_cancelled(
                    ["simulator"],
                    cancel_event=cancel_event,
                    container_name=None,
                    log_file=log_file,
                    env={},
                )
        popen.assert_not_called()

    def test_cancel_signal_is_linearized_with_active_process_snapshot(self) -> None:
        cancel_event = threading.Event()
        evaluator.terminate_all_active(0.0, cancel_event=cancel_event)
        self.assertTrue(cancel_event.is_set())

    def test_new_matrix_invalidates_stale_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            marker = output_dir / "COMPLETED.json"
            marker.write_text("{}", encoding="utf-8")
            self.assertTrue(evaluator.invalidate_completion_marker(output_dir))
            self.assertFalse(marker.exists())
            self.assertFalse(evaluator.invalidate_completion_marker(output_dir))

    def test_main_invalidates_marker_before_expensive_runtime_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            marker = output_dir / "COMPLETED.json"
            marker.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                evaluator,
                "resolve_runtime_environment_identity",
                side_effect=RuntimeError("preflight stopped"),
            ), self.assertRaisesRegex(RuntimeError, "preflight stopped"):
                evaluator.main(
                    [
                        "--runtime",
                        "docker",
                        "--maps",
                        "E01",
                        "--blue-policies",
                        "b0_fixed_ratio_random",
                        "--seeds",
                        "11902",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertFalse(marker.exists())
            status = json.loads(
                (output_dir / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "preflight")

    def test_scenario_limit_resolves_to_3000_and_explicit_1200(self) -> None:
        default_protocol = evaluator.resolve_scenario_protocol(
            "final20/easy/E01", None
        )
        explicit_protocol = evaluator.resolve_scenario_protocol(
            "final20/easy/E01", 1200
        )

        self.assertEqual(default_protocol[2], 3000)
        self.assertEqual(explicit_protocol[2], 1200)
        self.assertEqual(default_protocol[:2], explicit_protocol[:2])

    def test_1200_and_3000_have_distinct_run_ids_and_fingerprints(self) -> None:
        task_1200 = make_task(1200)
        task_3000 = make_task(3000)
        identity_1200 = identity(task_1200)
        identity_3000 = identity(task_3000)

        run_id_1200 = evaluator.task_run_id("test", task_1200, identity_1200)
        run_id_3000 = evaluator.task_run_id("test", task_3000, identity_3000)

        self.assertIn("_ms1200_", run_id_1200)
        self.assertIn("_ms3000_", run_id_3000)
        self.assertNotEqual(run_id_1200, run_id_3000)
        self.assertNotEqual(
            evaluator.task_protocol_fingerprint(identity_1200),
            evaluator.task_protocol_fingerprint(identity_3000),
        )

    def test_identity_contains_complete_task_protocol(self) -> None:
        protocol = identity(make_task(1200))

        self.assertEqual(protocol["checkpoint"]["sha256"], CHECKPOINT_SHA256)
        self.assertEqual(protocol["scenario"]["sha256"], SCENARIO_SHA256)
        self.assertEqual(protocol["objective_motion"], "static")
        self.assertEqual(protocol["red_upper_policy"], "r9_hierarchical_learning")
        self.assertEqual(protocol["red_motion_policy"], "local_avoid_s3")
        self.assertEqual(protocol["gate_mode"], "none")
        self.assertEqual(protocol["blue_policy"], "b0_fixed_ratio_random")
        self.assertEqual(protocol["seed"], 11902)
        self.assertEqual(protocol["max_steps"], 1200)
        self.assertEqual(protocol["execution"]["workers"], 4)
        self.assertEqual(
            protocol["execution"]["wall_clock_timeout_seconds"], 1800.0
        )
        self.assertEqual(protocol["execution"]["kill_grace_seconds"], 10.0)
        self.assertEqual(
            protocol["execution"]["accepted_termination_reasons"],
            ["environment_done", "time_limit"],
        )
        self.assertEqual(protocol["runtime"], "docker")
        self.assertEqual(protocol["docker_image"], "registry.example/simulator:test")
        self.assertEqual(
            protocol["runtime_environment_identity"],
            RUNTIME_ENVIRONMENT_IDENTITY,
        )
        self.assertEqual(protocol["code_identity"], CODE_IDENTITY)
        self.assertEqual(protocol["observation_encoding_identity"], OBSERVATION_IDENTITY)

    def test_execution_and_termination_settings_participate_in_fingerprint(self) -> None:
        task = make_task(1200)
        baseline = evaluator.task_protocol_fingerprint(identity(task))
        variants = (
            identity(task, workers=2),
            identity(task, timeout_seconds=900.0),
            identity(task, kill_grace_seconds=1.0),
        )
        for variant in variants:
            with self.subTest(execution=variant["execution"]):
                self.assertNotEqual(
                    baseline, evaluator.task_protocol_fingerprint(variant)
                )

    def test_runtime_environment_identity_participates_in_fingerprint(self) -> None:
        task = make_task(1200)
        original = identity(task)
        original_fingerprint = evaluator.task_protocol_fingerprint(original)
        variants = []

        changed_image_id = json.loads(json.dumps(RUNTIME_ENVIRONMENT_IDENTITY))
        changed_image_id["docker_image_id"] = "sha256:" + "3" * 64
        changed_image_id["docker_run_reference"] = "sha256:" + "3" * 64
        variants.append(changed_image_id)

        changed_repo_digest = json.loads(json.dumps(RUNTIME_ENVIRONMENT_IDENTITY))
        changed_repo_digest["docker_repo_digests"] = [
            "registry.example/simulator@sha256:" + "5" * 64,
        ]
        variants.append(changed_repo_digest)

        changed_python = json.loads(json.dumps(RUNTIME_ENVIRONMENT_IDENTITY))
        changed_python["mounted_competition_python"]["sha256"] = "6" * 64
        variants.append(changed_python)

        changed_environment = json.loads(json.dumps(RUNTIME_ENVIRONMENT_IDENTITY))
        changed_environment["competition_environment_tree"]["sha256"] = "7" * 64
        variants.append(changed_environment)

        for changed_runtime in variants:
            with self.subTest(changed_runtime=changed_runtime):
                changed = evaluator.task_protocol_identity(
                    task,
                    objective_motion="static",
                    runtime="docker",
                    docker_image="registry.example/simulator:test",
                    workers=4,
                    timeout_seconds=1800.0,
                    kill_grace_seconds=10.0,
                    runtime_environment_identity=changed_runtime,
                    code_identity=CODE_IDENTITY,
                    observation_encoding_identity=OBSERVATION_IDENTITY,
                )
                self.assertNotEqual(
                    original_fingerprint,
                    evaluator.task_protocol_fingerprint(changed),
                )

    def test_host_runtime_identity_records_executable_and_platform(self) -> None:
        protocol = evaluator.task_protocol_identity(
            make_task(1200),
            objective_motion="static",
            runtime="host",
            docker_image="unused:test",
            workers=4,
            timeout_seconds=1800.0,
            kill_grace_seconds=10.0,
            runtime_environment_identity=HOST_RUNTIME_ENVIRONMENT_IDENTITY,
            code_identity=CODE_IDENTITY,
            observation_encoding_identity=OBSERVATION_IDENTITY,
        )

        self.assertEqual(
            protocol["runtime_environment_identity"],
            HOST_RUNTIME_ENVIRONMENT_IDENTITY,
        )

    def test_exact_sidecar_match_hits_and_force_skips_cache(self) -> None:
        task = make_task(1200)
        protocol = identity(task)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            summary_path = run_dir / "stamp" / "summary.json"
            write_summary(summary_path)
            evaluator.write_protocol_sidecar(run_dir, summary_path, protocol)

            self.assertEqual(
                evaluator.reusable_summary(run_dir, protocol, task),
                summary_path.resolve(),
            )
            self.assertIsNone(
                evaluator.reusable_summary(run_dir, protocol, task, force=True)
            )
            self.assertIsNone(
                evaluator.reusable_summary(
                    run_dir,
                    identity(task, timeout_seconds=900.0),
                    task,
                )
            )

    def test_missing_or_mismatched_sidecar_is_a_cache_miss(self) -> None:
        protocol_1200 = identity(make_task(1200))
        protocol_3000 = identity(make_task(3000))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            summary_path = run_dir / "stamp" / "summary.json"
            write_summary(summary_path)

            self.assertIsNone(
                evaluator.reusable_summary(run_dir, protocol_1200, make_task(1200))
            )

            evaluator.write_protocol_sidecar(run_dir, summary_path, protocol_1200)
            self.assertIsNone(
                evaluator.reusable_summary(run_dir, protocol_3000, make_task(3000))
            )

            sidecar_path = run_dir / evaluator.CACHE_PROTOCOL_FILENAME
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["protocol_fingerprint"] = "0" * 64
            evaluator.atomic_json(sidecar_path, sidecar)
            self.assertIsNone(
                evaluator.reusable_summary(run_dir, protocol_1200, make_task(1200))
            )

    def test_fresh_sidecar_is_committed_only_by_final_commit_stage(self) -> None:
        task = make_task(1200)
        args = make_args()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            evaluator, "ROOT", Path(directory)
        ):
            protocol = evaluator.task_identity_from_args(
                args,
                task,
                RUNTIME_ENVIRONMENT_IDENTITY,
                CODE_IDENTITY,
                OBSERVATION_IDENTITY,
            )
            run_id = evaluator.task_run_id("test", task, protocol)
            run_dir = Path(directory) / "results" / "runs" / run_id
            summary_path = run_dir / "stamp" / "summary.json"
            write_summary(summary_path)
            record = evaluator.summary_record(
                task,
                summary_path,
                run_id=run_id,
                protocol_fingerprint=evaluator.task_protocol_fingerprint(protocol),
                cached=False,
                elapsed_seconds=1.0,
            )
            sidecar = run_dir / evaluator.CACHE_PROTOCOL_FILENAME
            self.assertFalse(sidecar.exists())
            committed = evaluator.commit_completed_task_sidecars(
                [record],
                args,
                RUNTIME_ENVIRONMENT_IDENTITY,
                CODE_IDENTITY,
                OBSERVATION_IDENTITY,
            )
            self.assertEqual(committed, [sidecar])
            self.assertTrue(sidecar.is_file())

    def test_final_cache_commit_rejects_changed_summary(self) -> None:
        task = make_task(1200)
        args = make_args()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            evaluator, "ROOT", Path(directory)
        ):
            protocol = evaluator.task_identity_from_args(
                args,
                task,
                RUNTIME_ENVIRONMENT_IDENTITY,
                CODE_IDENTITY,
                OBSERVATION_IDENTITY,
            )
            run_id = evaluator.task_run_id("test", task, protocol)
            run_dir = Path(directory) / "results" / "runs" / run_id
            summary_path = run_dir / "stamp" / "summary.json"
            write_summary(summary_path)
            record = evaluator.summary_record(
                task,
                summary_path,
                run_id=run_id,
                protocol_fingerprint=evaluator.task_protocol_fingerprint(protocol),
                cached=False,
                elapsed_seconds=1.0,
            )
            update_summary(
                summary_path,
                lambda payload: payload["score"].__setitem__("score", 13.0),
            )
            with self.assertRaisesRegex(RuntimeError, "Summary changed"):
                evaluator.commit_completed_task_sidecars(
                    [record],
                    args,
                    RUNTIME_ENVIRONMENT_IDENTITY,
                    CODE_IDENTITY,
                    OBSERVATION_IDENTITY,
                )
            self.assertFalse((run_dir / evaluator.CACHE_PROTOCOL_FILENAME).exists())

    def test_summary_validation_rejects_identity_numeric_and_count_failures(self) -> None:
        task = make_task(1200)
        mutations = {
            "scenario": lambda payload: payload.__setitem__("scenario", "H07"),
            "policy": lambda payload: payload["policies"].__setitem__(
                "blue", "b3_min_cost_assignment"
            ),
            "nan": lambda payload: payload["score"].__setitem__(
                "score", math.nan
            ),
            "negative": lambda payload: payload["outcomes"].__setitem__(
                "launched", -1
            ),
            "counts": lambda payload: payload["outcomes"].__setitem__(
                "engaged", 1
            ),
            "types": lambda payload: payload["outcomes"]["by_type"].pop(
                "21002"
            ),
            "type_totals": lambda payload: payload["outcomes"]["by_type"][
                "21000"
            ].__setitem__("launched", 1),
            "platform_totals": lambda payload: payload["outcomes"][
                "per_platform"
            ].__setitem__(
                "1",
                {
                    "target_id": 1,
                    "engaged": False,
                    "resolved": False,
                    "arrived": False,
                    "effective_hit": False,
                    "alive_at_end": True,
                    "minimum_target_distance_m": None,
                },
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                summary_path = Path(directory) / "summary.json"
                write_summary(summary_path)
                update_summary(summary_path, mutation)
                with self.assertRaisesRegex(ValueError, "Invalid simulator summary"):
                    evaluator.summary_record(
                        task,
                        summary_path,
                        run_id="test",
                        protocol_fingerprint="a" * 64,
                        cached=False,
                        elapsed_seconds=1.0,
                    )

    def test_cache_rejects_semantically_invalid_summary_even_with_matching_hash(self) -> None:
        task = make_task(1200)
        protocol = identity(task)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            summary_path = run_dir / "stamp" / "summary.json"
            write_summary(summary_path)
            update_summary(
                summary_path,
                lambda payload: payload["outcomes"].__setitem__("arrived", 1),
            )
            evaluator.write_protocol_sidecar(run_dir, summary_path, protocol)
            self.assertIsNone(
                evaluator.reusable_summary(run_dir, protocol, task)
            )

    def test_preview_report_and_record_include_resolved_max_steps(self) -> None:
        v13 = make_task(1200, condition="v13")
        v8 = make_task(1200, condition="v8")
        tasks = [v13, v8]
        conditions = (
            evaluator.Condition("v13", Path(v13.checkpoint), "none", CHECKPOINT_SHA256),
            evaluator.Condition("v8", Path(v8.checkpoint), "none", CHECKPOINT_SHA256),
        )
        args = argparse.Namespace(
            objective_motion="static",
            runtime="docker",
            docker_image="registry.example/simulator:test",
            workers=1,
            timeout_seconds=10.0,
            kill_grace_seconds=2.0,
        )

        preview = evaluator.protocol_preview(
            tasks,
            conditions,
            args,
            RUNTIME_ENVIRONMENT_IDENTITY,
            CODE_IDENTITY,
            OBSERVATION_IDENTITY,
        )
        records = [{**task.__dict__, "error": "not run"} for task in tasks]
        report = evaluator.build_report(
            tasks,
            records,
            conditions,
            args,
            RUNTIME_ENVIRONMENT_IDENTITY,
            CODE_IDENTITY,
            OBSERVATION_IDENTITY,
        )

        self.assertEqual(preview["max_steps"], 1200)
        self.assertEqual(preview["max_steps_by_map"], {"E01": 1200})
        self.assertEqual(report["protocol"]["max_steps"], 1200)
        self.assertEqual(report["protocol"]["max_steps_by_map"], {"E01": 1200})
        self.assertEqual(
            preview["runtime_environment_identity"],
            RUNTIME_ENVIRONMENT_IDENTITY,
        )
        self.assertEqual(
            report["protocol"]["runtime_environment_identity"],
            RUNTIME_ENVIRONMENT_IDENTITY,
        )

        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            write_summary(summary_path)
            record = evaluator.summary_record(
                v8,
                summary_path,
                run_id="test_ms1200",
                protocol_fingerprint=evaluator.task_protocol_fingerprint(identity(v8)),
                cached=False,
                elapsed_seconds=1.0,
            )
        self.assertEqual(record["max_steps"], 1200)
        self.assertEqual(record["termination_reason"], "environment_done")

    def test_summary_rejects_inconsistent_time_limit(self) -> None:
        task = make_task(1200)
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            write_summary(
                summary_path,
                steps=1199,
                termination_reason="time_limit",
            )
            with self.assertRaisesRegex(ValueError, "expected 1200"):
                evaluator.summary_record(
                    task,
                    summary_path,
                    run_id="test_ms1200",
                    protocol_fingerprint=evaluator.task_protocol_fingerprint(
                        identity(task)
                    ),
                    cached=False,
                    elapsed_seconds=1.0,
                )


if __name__ == "__main__":
    unittest.main()
