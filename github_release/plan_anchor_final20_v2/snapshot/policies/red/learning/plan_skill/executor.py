"""Deterministic compiler/runtime for :mod:`plan_skill.spec`.

The runtime deliberately has no trainable parameters.  It binds every plan
token to concrete entity IDs before an episode, exposes the small policy API
used by ``UnifiedAttackMissileAgent``, and produces a plan-versus-command
ledger.  It never reads hidden target truth: a hidden response is armed only
by ``InfoCenter.discovery_events`` after legal track fusion accepts a 9500.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import SimpleNamespace
from typing import Iterable, Mapping

import numpy as np

from policies.red.learning.unified.info_center import InfoCenter
from policies.red.learning.unified.unified_encoder import (
    GOAL_SLOTS,
    eligible_targets,
    engagement_efficiency,
    target_sort_key,
)
from policies.red.learning.unified.unified_networks import (
    KEEP_INDEX,
    SECTOR_INDEX_0,
    TARGET_DIM,
)

from .spec import (
    DeploymentTemplate,
    FormationTemplate,
    GoalKind,
    GoalRef,
    HiddenResponseMode,
    PlanSpec,
    ROLE_ORDER,
    Role,
)


ROLE_TO_ENTITY_TYPE = {Role.H: 21000, Role.M: 21001, Role.L: 21002}
ENTITY_TYPE_TO_ROLE = {value: key for key, value in ROLE_TO_ENTITY_TYPE.items()}
SUPPORTED_SEARCH_POLICIES = frozenset(("legal_grid_search_v1",))


def load_plan_spec(path) -> PlanSpec:
    """Load a canonical token-list JSON accepted by PlanSpec."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("plan spec JSON must be a canonical token list")
    return PlanSpec.from_tokens(raw)


@dataclass(frozen=True)
class PlanBinding:
    entity_id: int
    role: Role
    wave_id: int | None
    formation_id: str | None
    planned_release_step: int | None
    goal: GoalRef | None
    search_cell: int | None = None
    reserve_response: bool = False
    response_group_index: int | None = None
    formation_rank: int = 0

    def as_dict(self) -> dict:
        goal = None
        if self.goal is not None:
            goal = {
                "kind": self.goal.kind.value,
                "target_id": (
                    self.goal.target_id
                    if self.goal.kind == GoalKind.PUBLIC_TARGET else None
                ),
                "target_type": (
                    self.goal.target_type
                    if self.goal.kind == GoalKind.PUBLIC_TARGET else None
                ),
                "search_policy_id": (
                    self.goal.search_policy_id
                    if self.goal.kind == GoalKind.HIDDEN_SEARCH else None
                ),
            }
        return {
            "entity_id": self.entity_id,
            "role": self.role.value,
            "wave_id": self.wave_id,
            "formation_id": self.formation_id,
            "planned_release_step": self.planned_release_step,
            "goal": goal,
            "search_cell": self.search_cell,
            "reserve_response": self.reserve_response,
            "response_group_index": self.response_group_index,
            "formation_rank": self.formation_rank,
        }


def _rectangle(coordinates) -> tuple[float, float, float, float]:
    if len(coordinates) != 1 or len(coordinates[0]) != 4:
        raise ValueError("deployment area must contain one four-corner rectangle")
    points = coordinates[0]
    west = min(float(point[0]) for point in points)
    east = max(float(point[0]) for point in points)
    south = min(float(point[1]) for point in points)
    north = max(float(point[1]) for point in points)
    expected = {(west, south), (west, north), (east, south), (east, north)}
    if west >= east or south >= north or {
        (float(point[0]), float(point[1])) for point in points
    } != expected:
        raise ValueError("deployment area must be a non-degenerate axis-aligned rectangle")
    return west, east, south, north


def build_grid_deployment_plan(
    spec: PlanSpec,
    *,
    entity_types: Mapping[int, int],
    coordinates,
    coordinates_hm=None,
    public_target_positions: Iterable[tuple[float, float]],
) -> dict:
    """Compile normalized deployment parameters to the legal 3 x 8 x 4 grid.

    Grid cells are scored in the frame from the deployment centre to the
    public-target centroid.  ``frontage`` controls the number of lateral grid
    lanes and ``depth`` the number of forward rows.  The closest deterministic
    cells receive counts as evenly as possible.  No hidden target contributes
    to this transform.
    """

    targets = tuple((float(x), float(y)) for x, y in public_target_positions)

    def framed_cells(zone):
        west, east, south, north = _rectangle(zone)
        centre_x = 0.5 * (west + east)
        centre_y = 0.5 * (south + north)
        if targets:
            target_x = sum(item[0] for item in targets) / len(targets)
            target_y = sum(item[1] for item in targets) / len(targets)
        else:
            target_x, target_y = centre_x, north
        tx = (target_x - centre_x) / max(0.5 * (east - west), 1e-12)
        ty = (target_y - centre_y) / max(0.5 * (north - south), 1e-12)
        norm = math.hypot(tx, ty)
        forward_axis = (0.0, 1.0) if norm <= 1e-12 else (tx / norm, ty / norm)
        lateral_axis = (-forward_axis[1], forward_axis[0])
        result = []
        for row in range(8):
            for col in range(4):
                nx = -1.0 + 2.0 * (col + 0.5) / 4.0
                ny = 1.0 - 2.0 * (row + 0.5) / 8.0
                forward = nx * forward_axis[0] + ny * forward_axis[1]
                lateral = nx * lateral_axis[0] + ny * lateral_axis[1]
                result.append((row, col, forward, lateral))
        return result

    l_cells = framed_cells(coordinates)
    hm_cells = framed_cells(coordinates_hm or coordinates)

    deployment = spec.deployment
    lateral_lanes = 1 + round(float(deployment.frontage) * 3.0)
    forward_rows = 1 + round(float(deployment.depth) * 7.0)
    cell_budget = min(32, max(1, lateral_lanes * forward_rows))
    role_offsets = {
        Role.H: deployment.h_depth_offset,
        Role.M: deployment.m_depth_offset,
        Role.L: deployment.l_depth_offset,
    }
    counts: dict[str, list[list[int]]] = {}
    for role in ROLE_ORDER:
        entity_type = ROLE_TO_ENTITY_TYPE[role]
        total = sum(int(value) == entity_type for value in entity_types.values())
        matrix = [[0 for _ in range(4)] for _ in range(8)]
        desired_forward = max(-1.0, min(
            1.0,
            float(deployment.anchor_forward)
            + float(role_offsets[role]) * max(float(deployment.depth), 0.125),
        ))
        desired_lateral = float(deployment.anchor_lateral)

        if deployment.template == DeploymentTemplate.LATERAL_LINE:
            forward_weight, lateral_weight = 4.0, 0.7
        elif deployment.template == DeploymentTemplate.COMPACT:
            forward_weight, lateral_weight = 1.5, 1.5
        else:
            forward_weight, lateral_weight = 1.0, 1.0
        role_cells = hm_cells if role in (Role.H, Role.M) else l_cells
        ordered = sorted(
            role_cells,
            key=lambda cell: (
                forward_weight * (cell[2] - desired_forward) ** 2
                + lateral_weight * (cell[3] - desired_lateral) ** 2,
                cell[0],
                cell[1],
            ),
        )[:cell_budget]
        quotient, remainder = divmod(total, len(ordered))
        for index, (row, col, _forward, _lateral) in enumerate(ordered):
            matrix[row][col] = quotient + (1 if index < remainder else 0)
        counts[str(entity_type)] = matrix
    return {"version": 1, "counts": counts}


class DeterministicPlanPolicy:
    """Non-learning, fail-closed runtime for one immutable :class:`PlanSpec`."""

    goal_slots = GOAL_SLOTS
    sector_index_0 = SECTOR_INDEX_0
    keep_index = KEEP_INDEX
    target_dim = TARGET_DIM
    training = False
    update_count = 0
    _rounds_done = 0

    def __init__(
        self,
        spec: PlanSpec,
        *,
        initial_targets,
        entity_types: Mapping[int, int],
        max_steps: int,
        search_bounds: tuple[float, float, float, float],
        search_rows: int = 4,
        search_cols: int = 8,
    ):
        self.spec = spec
        self.initial_targets = tuple(initial_targets)
        self.entity_types = {
            int(entity_id): int(entity_type)
            for entity_id, entity_type in entity_types.items()
            if int(entity_type) in ENTITY_TYPE_TO_ROLE
        }
        self.total_platforms = len(self.entity_types)
        self.max_steps = int(max_steps)
        self.search_bounds = tuple(float(value) for value in search_bounds)
        self.search_rows = int(search_rows)
        self.search_cols = int(search_cols)
        # Match TopFormer's variable-candidate action layout rather than the
        # legacy UnifiedMAPPO layout, which exposes only two search sectors.
        self.goal_slots = 64
        self.search_count = self.search_rows * self.search_cols
        self.sector_index_0 = self.goal_slots
        self.keep_index = self.goal_slots + self.search_count
        self.target_dim = self.keep_index + 1
        self.coordinate_scale = None
        self.config = SimpleNamespace(
            max_steps=self.max_steps,
            train_stage=f"deterministic_plan_v{self.spec.schema_version}",
            upper_phase="plan_replay",
            sat_trainable=False,
            deploy_trainable=False,
            l_scout_enabled=False,
            l_land_targets=True,
        )

        counts = {
            role: sum(
                entity_type == ROLE_TO_ENTITY_TYPE[role]
                for entity_type in self.entity_types.values()
            )
            for role in ROLE_ORDER
        }
        public_targets = {
            int(target.entity_id): int(target.entity_type)
            for target in self.initial_targets
            if int(target.entity_type) in (9400, 9600)
        }
        self.spec.validate_for_context(
            available_counts=counts,
            public_targets=public_targets,
            max_steps=self.max_steps,
        )
        self._validate_executable_goals()
        self._bindings = self._bind_units()
        self._binding_by_entity = {
            binding.entity_id: binding for binding in self._bindings
        }
        self._entity_to_agent: dict[int, int] = {}
        self._agents: list = []
        self._agent_by_id: dict[int, object] = {}
        self._expected_deployment: dict[int, tuple[float, float, float]] = {}
        self._last_score = 0.0
        self.reset_episode()

    def _validate_executable_goals(self) -> None:
        for wave in self.spec.waves:
            for allocation in wave.allocations:
                goal = allocation.goal
                if goal.kind == GoalKind.HIDDEN_SEARCH:
                    if goal.search_policy_id not in SUPPORTED_SEARCH_POLICIES:
                        raise ValueError(
                            f"unsupported search policy {goal.search_policy_id!r}; "
                            f"supported={sorted(SUPPORTED_SEARCH_POLICIES)}"
                        )
                    if any(
                        cell >= self.search_count for cell in goal.search_cells
                    ):
                        raise ValueError(
                            "explicit search cell is outside the configured grid"
                        )
                    continue
                missile_type = ROLE_TO_ENTITY_TYPE[allocation.role]
                if engagement_efficiency(missile_type, goal.target_type) <= 0.0:
                    raise ValueError(
                        f"{allocation.role.value} cannot engage target type "
                        f"{goal.target_type} in the current simulator"
                    )
                visible = sorted(
                    eligible_targets(self.initial_targets, missile_type),
                    key=lambda target: target_sort_key(target, missile_type),
                )[:self.goal_slots]
                if goal.target_id not in {int(target.entity_id) for target in visible}:
                    raise ValueError(
                        f"target {goal.target_id} is outside the executable "
                        f"{self.goal_slots}-slot candidate set for {allocation.role.value}"
                    )

    @staticmethod
    def _formation_order(drafts, formation):
        role_rank = {role: index for index, role in enumerate(formation.role_order)}
        if formation.template == FormationTemplate.ROLE_LAYERED:
            return sorted(
                drafts,
                key=lambda item: (
                    role_rank[item[1]], item[0], item[2].canonical_key
                ),
            )
        if formation.template == FormationTemplate.LATERAL_LINE:
            # Centre-out entity order avoids an arbitrary persistent left edge.
            ids = sorted(item[0] for item in drafts)
            centre_order = sorted(
                ids,
                key=lambda entity_id: (
                    abs(ids.index(entity_id) - (len(ids) - 1) / 2.0), entity_id
                ),
            )
            rank = {entity_id: index for index, entity_id in enumerate(centre_order)}
            return sorted(drafts, key=lambda item: (rank[item[0]], role_rank[item[1]]))
        return sorted(drafts, key=lambda item: (item[0], role_rank[item[1]]))

    def _bind_units(self) -> tuple[PlanBinding, ...]:
        pools = {
            role: sorted(
                entity_id for entity_id, entity_type in self.entity_types.items()
                if entity_type == ROLE_TO_ENTITY_TYPE[role]
            )
            for role in ROLE_ORDER
        }
        cursors = {role: 0 for role in ROLE_ORDER}
        formations = {
            formation.formation_id: formation for formation in self.spec.formations
        }
        result: list[PlanBinding] = []
        search_binding_indexes: list[int] = []
        explicit_search_cell_by_entity: dict[int, int] = {}

        for wave in sorted(
            self.spec.waves, key=lambda item: (item.release_step, item.wave_id)
        ):
            formation = formations[wave.formation_id]
            allocation_order = sorted(
                wave.allocations,
                key=lambda allocation: (
                    formation.role_order.index(allocation.role),
                    allocation.goal.canonical_key,
                ),
            )
            drafts = []
            for allocation in allocation_order:
                start = cursors[allocation.role]
                stop = start + allocation.count
                selected = pools[allocation.role][start:stop]
                if len(selected) != allocation.count:
                    raise ValueError(
                        f"not enough {allocation.role.value} units for wave {wave.wave_id}"
                    )
                cursors[allocation.role] = stop
                if allocation.goal.kind == GoalKind.HIDDEN_SEARCH \
                        and allocation.goal.search_cells:
                    explicit_search_cell_by_entity.update(zip(
                        selected, allocation.goal.search_cells, strict=True
                    ))
                drafts.extend(
                    (entity_id, allocation.role, allocation.goal)
                    for entity_id in selected
                )

            ordered = self._formation_order(drafts, formation)
            member_count = len(ordered)
            lateral_lanes = min(
                max(member_count, 1),
                1 + round(
                    float(formation.lateral_width)
                    * (min(max(member_count, 1), 8) - 1)
                ),
            )
            depth_stride = 1 + round(float(formation.longitudinal_depth) * 3.0)
            for formation_index, (entity_id, role, goal) in enumerate(ordered):
                formation_rank = (formation_index // lateral_lanes) * depth_stride
                release_step = (
                    int(wave.release_step)
                    + formation_rank * int(formation.intra_wave_gap_steps)
                )
                if release_step >= self.max_steps:
                    raise ValueError(
                        f"compiled release step {release_step} for entity {entity_id} "
                        f"is outside max_steps={self.max_steps}"
                    )
                binding = PlanBinding(
                    entity_id=entity_id,
                    role=role,
                    wave_id=wave.wave_id,
                    formation_id=wave.formation_id,
                    planned_release_step=release_step,
                    goal=goal,
                    search_cell=explicit_search_cell_by_entity.get(entity_id),
                    formation_rank=formation_rank,
                )
                result.append(binding)
                if goal.kind == GoalKind.HIDDEN_SEARCH:
                    search_binding_indexes.append(len(result) - 1)

        automatic_search_indexes = [
            index for index in search_binding_indexes
            if result[index].search_cell is None
        ]
        search_count = len(automatic_search_indexes)
        search_cells = self.search_rows * self.search_cols
        for ordinal, result_index in enumerate(automatic_search_indexes):
            if search_count <= search_cells:
                cell = min(
                    search_cells - 1,
                    int((ordinal + 0.5) * search_cells / search_count),
                )
            else:
                cell = ordinal % search_cells
            previous = result[result_index]
            result[result_index] = PlanBinding(
                **{**asdict(previous), "role": previous.role, "goal": previous.goal,
                   "search_cell": cell}
            )

        reserve_ids = {
            role: pools[role][cursors[role]:] for role in ROLE_ORDER
        }
        response = self.spec.hidden_response
        if response.mode != HiddenResponseMode.NONE:
            selected_response_ids = reserve_ids[Role.L][:response.reserve_l_count]
            group_count = int(response.required_discoveries)
            quotient, remainder = divmod(len(selected_response_ids), group_count)
            group_indexes = [
                group_index
                for group_index in range(group_count)
                for _ in range(quotient + (1 if group_index < remainder else 0))
            ]
            for entity_id, group_index in zip(
                selected_response_ids, group_indexes, strict=True
            ):
                result.append(PlanBinding(
                    entity_id=entity_id,
                    role=Role.L,
                    wave_id=None,
                    formation_id=response.formation_id,
                    planned_release_step=None,
                    goal=None,
                    reserve_response=True,
                    response_group_index=group_index,
                ))
        response_ids = {item.entity_id for item in result if item.reserve_response}
        for role in ROLE_ORDER:
            for entity_id in reserve_ids[role]:
                if entity_id in response_ids:
                    continue
                result.append(PlanBinding(
                    entity_id=entity_id,
                    role=role,
                    wave_id=None,
                    formation_id=None,
                    planned_release_step=None,
                    goal=None,
                ))
        return tuple(sorted(result, key=lambda item: item.entity_id))

    @property
    def bindings(self) -> tuple[PlanBinding, ...]:
        return self._bindings

    def build_deployment_plan(self, *, coordinates, coordinates_hm=None) -> dict:
        return build_grid_deployment_plan(
            self.spec,
            entity_types=self.entity_types,
            coordinates=coordinates,
            coordinates_hm=coordinates_hm,
            public_target_positions=(
                (target.position.lon, target.position.lat)
                for target in self.initial_targets
                if int(target.entity_type) in (9400, 9600)
            ),
        )

    def register_deployment_actions(self, actions) -> None:
        expected = {}
        for action in actions:
            if int(action[0]) != 0:
                continue
            expected[int(action[1])] = (
                float(action[2]), float(action[3]), float(action[4])
            )
        if set(expected) != set(self.entity_types):
            raise ValueError("compiled deployment actions do not cover every plan unit")
        self._expected_deployment = expected

    def observe_deployment(self, simulator_factory) -> None:
        self._deployment_actual = {}
        for entity_id, expected in self._expected_deployment.items():
            simulator = simulator_factory.get_simulator_by_id(entity_id)
            if simulator is None:
                self._deployment_actual[entity_id] = {
                    "status": "missing_simulator", "expected": list(expected)
                }
                continue
            lla = simulator.entity_ext.entity.lla
            actual = (float(lla.x), float(lla.y), float(lla.z))
            error = math.sqrt(sum(
                (actual[index] - expected[index]) ** 2 for index in range(3)
            ))
            self._deployment_actual[entity_id] = {
                "status": "match" if error <= 1e-9 else "mismatch",
                "expected": list(expected),
                "actual": list(actual),
                "absolute_error": error,
            }

    # ---- shared policy lifecycle -------------------------------------------------

    def reset_episode(self) -> None:
        self.info_center = InfoCenter(
            self.initial_targets,
            self.total_platforms,
            coordinate_scale=self.coordinate_scale,
            search_bounds=self.search_bounds,
            search_rows=self.search_rows,
            search_cols=self.search_cols,
        )
        self._sat_requests: list[int] = []
        self._sat_granted: set[int] = set()
        self._current_step = 0
        self._hidden_trigger: dict | None = None
        self._launch_events: dict[int, dict] = {}
        self._blocked: dict[int, dict] = {}
        self._semantic_retarget_violations: list[dict] = []
        self._search_completions: dict[int, dict] = {}
        self._engine_launch_ids: set[int] = set()
        self._engine_reconciled = False
        self._satellite_events: list[dict] = []
        self._deployment_actual: dict[int, dict] = {}
        self._score_frozen_stop: dict | None = None

    def register_entity_agent(self, entity_id: int, agent_id: int) -> None:
        self._entity_to_agent[int(entity_id)] = int(agent_id)

    def register_agent_object(self, agent) -> None:
        self._agents.append(agent)
        self._agent_by_id[int(agent.agent_id)] = agent

    def begin_deploy_phase(self, _full_observation) -> None:
        return None

    def begin_observation_phase(self, full_observation):
        self._current_step = int((full_observation or {}).get("step", self._current_step))
        self.info_center.begin_step(self._current_step)
        self._arm_hidden_response()
        return None

    def begin_decision_phase(self, step: int, agent_ids) -> tuple[int, ...]:
        self._current_step = int(step)
        # Reports run before this hook, so a discovery accepted in the current
        # observation phase can arm the pre-committed response without an
        # unintended extra one-step delay.
        self._arm_hidden_response()
        return tuple(sorted(
            agent_ids,
            key=lambda agent_id: (
                self._resolved_release_step_for_agent(agent_id), int(agent_id)
            ),
        ))

    def end_decision_phase(self, _step: int) -> None:
        return None

    def _arm_hidden_response(self) -> None:
        if self._hidden_trigger is not None or not self.info_center.discovery_events:
            return
        response = self.spec.hidden_response
        if response.mode == HiddenResponseMode.NONE:
            return
        unique_events = []
        seen_target_ids = set()
        for event in self.info_center.discovery_events:
            target_id = int(event["target_id"])
            if target_id in seen_target_ids:
                continue
            seen_target_ids.add(target_id)
            unique_events.append(event)
        if len(unique_events) < response.required_discoveries:
            return
        selected_events = unique_events[:response.required_discoveries]
        final_event = selected_events[-1]
        target_ids = [int(event["target_id"]) for event in selected_events]
        self._hidden_trigger = {
            "discovery_step": int(final_event["step"]),
            "discovery_steps": [int(event["step"]) for event in selected_events],
            "target_ids": target_ids,
            "release_step": (
                int(final_event["step"])
                + int(response.release_delay_steps)
            ),
        }
        if len(target_ids) == 1:
            self._hidden_trigger["target_id"] = target_ids[0]

    def _resolved_binding(self, agent_id: int) -> tuple[PlanBinding | None, int | None, int | None]:
        agent = self._agent_by_id.get(int(agent_id))
        if agent is None:
            return None, None, None
        binding = self._binding_by_entity.get(int(agent.entity_id))
        if binding is None:
            return None, None, None
        if not binding.reserve_response:
            target_id = (
                binding.goal.target_id
                if binding.goal is not None
                and binding.goal.kind == GoalKind.PUBLIC_TARGET else None
            )
            return binding, binding.planned_release_step, target_id
        if self._hidden_trigger is None:
            return binding, None, None
        group_index = binding.response_group_index
        if group_index is None:
            raise RuntimeError("response binding is missing its target group")
        return (
            binding,
            int(self._hidden_trigger["release_step"]),
            int(self._hidden_trigger["target_ids"][group_index]),
        )

    def _resolved_release_step_for_agent(self, agent_id: int) -> int:
        _binding, release_step, _target_id = self._resolved_binding(agent_id)
        return self.max_steps + 1 if release_step is None else int(release_step)

    def is_planned_search_entity(self, entity_id: int) -> bool:
        binding = self._binding_by_entity.get(int(entity_id))
        return bool(
            binding is not None
            and binding.goal is not None
            and binding.goal.kind == GoalKind.HIDDEN_SEARCH
        )

    def record_planned_search_complete(
        self, entity_id: int, cell: int, step: int
    ) -> None:
        self._search_completions.setdefault(int(entity_id), {
            "cell": int(cell), "step": int(step)
        })

    def select_decision(
        self,
        *,
        agent_id: int,
        step: int,
        launch_active: bool,
        goal_event: bool,
        motion_active: bool,
        resource_active: bool,
        **_unused,
    ) -> dict:
        agent = self._agent_by_id[int(agent_id)]
        binding, release_step, target_id = self._resolved_binding(agent_id)
        goal_action = None
        launch_action = None

        if agent.launch_step < 0 and binding is not None and release_step is not None:
            if int(step) >= int(release_step):
                if binding.goal is not None and binding.goal.kind == GoalKind.HIDDEN_SEARCH:
                    goal_action = self.sector_index_0 + int(binding.search_cell)
                else:
                    goal_action = next((
                        index for index, target in enumerate(agent._sorted_targets)
                        if int(target.entity_id) == int(target_id)
                    ), None)
                if goal_action is None:
                    self._blocked.setdefault(int(agent.entity_id), {
                        "first_step": int(step),
                        "reason": "planned_target_not_in_legal_candidate_set",
                        "target_id": target_id,
                    })
                elif launch_active:
                    launch_action = 0
        elif agent.launch_step >= 0 and goal_event:
            # PlanSpec contains no RETARGET token.  KEEP is the only legal
            # semantic response; lower-level search waypoint changes are not
            # semantic retargets.
            goal_action = self.keep_index

        return {
            "launch": launch_action,
            "goal": goal_action,
            "motion": 1 if motion_active else None,
            "resource": 1 if resource_active else None,
        }

    def register_satellite_request(self, entity_id: int) -> None:
        self._sat_requests.append(int(entity_id))

    def is_satellite_decider(self, entity_id: int) -> bool:
        candidates = [
            int(agent.entity_id) for agent in self._agents
            if int(getattr(agent, "missile_type", -1)) == 21000
            and int(getattr(agent, "launch_step", -1)) >= 0
            and not bool(getattr(agent, "sat_used", False))
            and int(agent.entity_id) in self.info_center.alive_ids
            and int(agent.entity_id) not in self.info_center.lost_ids
            and getattr(agent, "latest_observation", None)
            and agent.latest_observation.get("self")
        ]
        return bool(candidates) and int(entity_id) == min(candidates)

    def settle_satellite_requests(self):
        requests, self._sat_requests = self._sat_requests, []
        commands = []
        if requests and self.info_center.sat_budget_remaining > 0:
            entity_id = min(requests)
            self.info_center.mark_satellite_used()
            self._sat_granted.add(entity_id)
            commands.append([3, entity_id, 0.0, 0.0])
            self._satellite_events.append({
                "step": self._current_step,
                "entity_id": entity_id,
                "controller": "fixed_lowest_live_unused_h",
            })
        return commands, {
            "granted_entity_ids": tuple(command[1] for command in commands),
            "budget_remaining": self.info_center.sat_budget_remaining,
        }

    @property
    def satellite_granted(self) -> set[int]:
        return self._sat_granted

    def observe_decision_batch(self, batch) -> None:
        for record in batch.records:
            agent = self._agent_by_id.get(int(record.agent_id))
            if agent is None or record.commands is None:
                continue
            commands = np.asarray(record.commands)
            if commands.size == 0:
                continue
            commands = commands.reshape((-1, commands.shape[-1]))
            for command in commands:
                action_type = int(command[0])
                entity_id = int(command[1])
                if action_type == 1:
                    binding, planned_step, planned_target = self._resolved_binding(
                        record.agent_id
                    )
                    actual_target = getattr(agent, "current_goal_id", None)
                    actual_cell = getattr(agent, "_current_sector_index", None)
                    self._launch_events.setdefault(entity_id, {
                        "actual_step": int(record.step),
                        "planned_step": planned_step,
                        "planned_target_id": planned_target,
                        "actual_target_id": actual_target,
                        "planned_search_cell": (
                            binding.search_cell if binding is not None else None
                        ),
                        "actual_search_cell": actual_cell,
                        "command_target": [float(command[2]), float(command[3])],
                    })
                elif action_type == 2:
                    binding = self._binding_by_entity.get(entity_id)
                    if binding is None or (
                        binding.goal is None
                        or binding.goal.kind != GoalKind.HIDDEN_SEARCH
                    ):
                        self._semantic_retarget_violations.append({
                            "step": int(record.step),
                            "entity_id": entity_id,
                            "command_target": [float(command[2]), float(command[3])],
                        })

    def reconcile_episode_telemetry(self, telemetry: Mapping) -> None:
        """Cross-check emitted launch commands against native lifecycle events."""

        self._engine_launch_ids = {
            int(event["entity_id"])
            for event in (telemetry or {}).get("engine_lifecycle_events", ())
            if event.get("event_type") == "red_launch"
            and event.get("entity_id") is not None
        }
        self._engine_reconciled = True

    def plan_score_frozen(
        self,
        *,
        step: int,
        destroyed_objective_ids: Iterable[int],
        objective_ids: Iterable[int],
    ) -> bool:
        """Return whether no remaining plan action can change official score.

        This is an evaluation-only acceleration predicate.  Blue interceptors
        cannot damage score objectives, and untriggered reserve L units cannot
        discover anything while they remain on the ground.  We may therefore
        stop after all objectives are destroyed, or after every still-visible
        plan unit that could discover or damage a remaining objective has
        become ineffective.  The predicate never uses hidden truth: hidden
        completion is inferred only from the legal all-discoveries trigger.
        """

        if self._score_frozen_stop is not None:
            return True
        destroyed = {int(value) for value in destroyed_objective_ids}
        objectives = {int(value) for value in objective_ids}
        if not objectives:
            return False
        if objectives <= destroyed:
            self._score_frozen_stop = {
                "step": int(step),
                "reason": "all_objectives_destroyed",
                "destroyed_objective_ids": sorted(destroyed),
            }
            return True

        all_hidden_discovered = self._hidden_trigger is not None
        legal_target_ids = {
            int(target.entity_id) for target in self.info_center.targets()
        }

        def visible_agent(entity_id: int):
            agent_id = self._entity_to_agent.get(int(entity_id))
            agent = self._agent_by_id.get(agent_id) if agent_id is not None else None
            if agent is None:
                return None
            observation = getattr(agent, "latest_observation", None) or {}
            return agent if observation.get("self") else None

        for binding in self._bindings:
            agent = visible_agent(binding.entity_id)
            if agent is None:
                continue
            launched = int(getattr(agent, "launch_step", -1)) >= 0

            if binding.reserve_response:
                if self._hidden_trigger is None:
                    # Ground reserve cannot discover; a live non-reserve unit
                    # later in this loop is the only possible trigger source.
                    continue
                group_index = binding.response_group_index
                if group_index is None:
                    raise RuntimeError("response binding has no target group")
                target_id = int(
                    self._hidden_trigger["target_ids"][group_index]
                )
                if target_id in destroyed:
                    continue
                if launched or target_id in legal_target_ids:
                    return False
                continue

            goal = binding.goal
            if goal is None:
                continue
            if goal.kind == GoalKind.HIDDEN_SEARCH:
                if not all_hidden_discovered:
                    return False
                continue

            target_id = int(goal.target_id)
            if launched:
                if target_id not in destroyed:
                    return False
                # A public-route unit can still legally discover a hidden
                # objective before the all-discoveries gate has fired.
                if not all_hidden_discovered:
                    return False
            elif target_id not in destroyed and target_id in legal_target_ids:
                return False

        self._score_frozen_stop = {
            "step": int(step),
            "reason": "no_remaining_score_capable_plan_unit",
            "destroyed_objective_ids": sorted(destroyed),
        }
        return True

    # These hooks keep TrainingEnv's unified transaction intact without
    # creating samples or gradients.
    def note_team_scalar(self, *_args, **_kwargs) -> None:
        return None

    def note_agent_shaping(self, *_args, **_kwargs) -> None:
        return None

    def commit_transition(self, **_kwargs) -> bool:
        return False

    def finish_commit_phase(self) -> None:
        return None

    def record_launch(self, *_args, **_kwargs) -> None:
        return None

    def record_goal_choice(self, *_args, **_kwargs) -> None:
        return None

    def record_search_action(self, *_args, **_kwargs) -> None:
        return None

    def record_retarget(self, *_args, **_kwargs) -> None:
        return None

    def record_motion_dwell(self, *_args, **_kwargs) -> None:
        return None

    def finish_episode(self, score: float) -> None:
        self._last_score = float(score)
        self._rounds_done += 1

    def episode_metrics(self, ships_total=None) -> dict:
        return {
            "train_phase": f"deterministic-plan-v{self.spec.schema_version}",
            "training": False,
            "plan_id": self.spec.plan_id,
            "launch": {"count": len(self._launch_events)},
            "retarget": {"count": len(self._semantic_retarget_violations)},
            "satellite_usage": len(self._satellite_events),
            "discovery": {
                "ships_total": int(ships_total or 0),
                "ships_found": len(self.info_center.known_ships),
            },
        }

    def execution_consistency(self) -> dict:
        unit_rows = []
        exact = 0
        expected_launches = 0
        for binding in self._bindings:
            event = self._launch_events.get(binding.entity_id)
            resolved_step = binding.planned_release_step
            planned_target = None
            if binding.goal is not None and binding.goal.kind == GoalKind.PUBLIC_TARGET:
                planned_target = binding.goal.target_id
            if binding.reserve_response and self._hidden_trigger is not None:
                resolved_step = int(self._hidden_trigger["release_step"])
                group_index = binding.response_group_index
                planned_target = int(
                    self._hidden_trigger["target_ids"][group_index]
                )

            if binding.goal is None and not binding.reserve_response:
                status = "held_reserve"
            elif binding.reserve_response and self._hidden_trigger is None:
                status = "not_triggered"
            elif event is None:
                if self._score_frozen_stop is not None:
                    status = "not_needed_score_frozen"
                else:
                    status = "not_launched"
                    expected_launches += 1
            else:
                expected_launches += 1
                target_match = (
                    event["actual_target_id"] == planned_target
                    if planned_target is not None
                    else event["actual_search_cell"] == binding.search_cell
                )
                step_error = int(event["actual_step"]) - int(resolved_step)
                engine_observed = (
                    not self._engine_reconciled
                    or binding.entity_id in self._engine_launch_ids
                )
                if not engine_observed:
                    status = "engine_rejected_or_unobserved"
                elif step_error == 0 and target_match:
                    status = "exact"
                    exact += 1
                elif not target_match:
                    status = "goal_mismatch"
                elif step_error < 0:
                    status = "early"
                else:
                    status = "late"
            unit_rows.append({
                **binding.as_dict(),
                "resolved_release_step": resolved_step,
                "resolved_target_id": planned_target,
                "actual": event,
                "engine_launch_observed": (
                    binding.entity_id in self._engine_launch_ids
                    if self._engine_reconciled else None
                ),
                "status": status,
                "blocked": self._blocked.get(binding.entity_id),
            })

        deployment_matches = sum(
            row.get("status") == "match" for row in self._deployment_actual.values()
        )
        deployment_total = len(self._expected_deployment)
        binding_payload = json.dumps(
            [binding.as_dict() for binding in self._bindings],
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "schema_version": self.spec.schema_version,
            "plan_id": self.spec.plan_id,
            "deterministic": True,
            "trainable_parameters": 0,
            "binding_sha256": hashlib.sha256(
                binding_payload.encode("utf-8")
            ).hexdigest(),
            "formation_semantics": (
                "lateral_width compiles to simultaneous launch lanes; "
                "longitudinal_depth compiles to temporal layer stride; "
                "the engine has no in-flight geometric formation command"
            ),
            "hidden_trigger": self._hidden_trigger,
            "score_frozen_stop": self._score_frozen_stop,
            "search_completions": {
                str(key): value
                for key, value in sorted(self._search_completions.items())
            },
            "deployment": {
                "expected_count": deployment_total,
                "match_count": deployment_matches,
                "all_match": (
                    deployment_total > 0 and deployment_matches == deployment_total
                ),
                "by_entity": {
                    str(key): value
                    for key, value in sorted(self._deployment_actual.items())
                },
            },
            "launch_goal": {
                "expected_launches": expected_launches,
                "emitted_launches": len(self._launch_events),
                "engine_confirmed_launches": (
                    len(self._engine_launch_ids) if self._engine_reconciled else None
                ),
                "exact_count": exact,
                "all_exact": expected_launches == exact,
                "units": unit_rows,
            },
            "semantic_retarget_violations": self._semantic_retarget_violations,
            "satellite": {
                "controller": "fixed_lowest_live_unused_h",
                "events": self._satellite_events,
            },
        }
