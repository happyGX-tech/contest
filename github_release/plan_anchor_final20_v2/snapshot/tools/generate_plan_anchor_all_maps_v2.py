#!/usr/bin/env python3
"""Generate map-adaptive PlanSpec-v2 anchors for every final20 scenario.

The experiment is deliberately paired.  Within one map every anchor shares
the exact same entity-to-goal allocation, scout count and all-hidden discovery
gate.  An anchor changes only the deployment, strike-wave timing, or strike
formation named by the anchor.  Hidden target IDs and coordinates never enter
the generated plan tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from policies.red.learning.plan_skill import (
    DeploymentSpec,
    DeploymentTemplate,
    FormationSpec,
    FormationTemplate,
    GoalAllocation,
    GoalRef,
    HiddenResponseMode,
    HiddenResponseSpec,
    HiddenTargetSelector,
    PlanSpec,
    Role,
    RoleCounts,
    WaveSpec,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = ROOT / "scenarios/cases/final20"
DEFAULT_OUTPUT = ROOT / "experiments/plan_skill/anchors_final20_all_v2"
COMMON_SEEDS = (1, 2, 3, 4, 5)
MAX_STEPS = 3000

# Scout budgets keep the successful E01 32-L setting and scale it by map
# difficulty.  All remaining L units are held for the all-hidden response.
SCOUTS_BY_DIFFICULTY = {"easy": 32, "medium": 48, "hard": 72}


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    label: str
    factor: str
    description: str
    release_steps: tuple[int, int, int] = (0, 120, 240)
    deployment: DeploymentSpec | None = None
    strike_formation: FormationSpec | None = None


BASE_DEPLOYMENT = DeploymentSpec(
    template=DeploymentTemplate.ROLE_LAYERED,
    anchor_forward=0.0,
    anchor_lateral=0.0,
    frontage=0.67,
    depth=0.43,
    h_depth_offset=0.0,
    m_depth_offset=0.0,
    l_depth_offset=0.0,
)
BASE_STRIKE_FORMATION = FormationSpec(
    formation_id="strike",
    template=FormationTemplate.ROLE_LAYERED,
    lateral_width=0.65,
    longitudinal_depth=0.35,
    intra_wave_gap_steps=5,
    role_order=(Role.H, Role.M, Role.L),
)
SCOUT_FORMATION = FormationSpec(
    formation_id="scout",
    template=FormationTemplate.LATERAL_LINE,
    lateral_width=1.0,
    longitudinal_depth=0.0,
    intra_wave_gap_steps=0,
    role_order=(Role.L, Role.M, Role.H),
)
RESPONSE_FORMATION = FormationSpec(
    formation_id="hidden_response",
    template=FormationTemplate.COMPACT,
    lateral_width=1.0,
    longitudinal_depth=0.0,
    intra_wave_gap_steps=0,
    role_order=(Role.L, Role.M, Role.H),
)


ANCHORS = (
    Anchor(
        "A00_balanced", "均衡基准", "baseline",
        "中部部署、0/120/240 三波、普通角色分层编组。",
    ),
    Anchor(
        "A01_deploy_forward", "整体前置部署", "deployment",
        "仅将三类单位的整体部署锚点前移。",
        deployment=replace(BASE_DEPLOYMENT, anchor_forward=0.75),
    ),
    Anchor(
        "A02_deploy_rear", "整体后置部署", "deployment",
        "仅将三类单位的整体部署锚点后移。",
        deployment=replace(BASE_DEPLOYMENT, anchor_forward=-0.75),
    ),
    Anchor(
        "A03_deploy_compact", "密集部署", "deployment",
        "仅改成小正面、小纵深的紧凑部署。",
        deployment=replace(
            BASE_DEPLOYMENT,
            template=DeploymentTemplate.COMPACT,
            frontage=0.20,
            depth=0.15,
        ),
    ),
    Anchor(
        "A04_deploy_wide", "横向展开部署", "deployment",
        "仅改成大正面、浅纵深的横向展开部署。",
        deployment=replace(
            BASE_DEPLOYMENT,
            template=DeploymentTemplate.LATERAL_LINE,
            frontage=1.0,
            depth=0.15,
        ),
    ),
    Anchor(
        "A05_wave_mass", "单波齐射", "wave",
        "三个逻辑主攻波在第 0 步同时释放。",
        release_steps=(0, 0, 0),
    ),
    Anchor(
        "A06_wave_rapid", "快速三波", "wave",
        "三个主攻波在 0/40/80 步释放。",
        release_steps=(0, 40, 80),
    ),
    Anchor(
        "A07_wave_slow", "慢速三波", "wave",
        "三个主攻波在 0/300/600 步释放。",
        release_steps=(0, 300, 600),
    ),
    Anchor(
        "A08_formation_dense", "波内密集编组", "formation",
        "仅取消波内时间层，形成紧凑同时发射。",
        strike_formation=replace(
            BASE_STRIKE_FORMATION,
            template=FormationTemplate.COMPACT,
            lateral_width=1.0,
            longitudinal_depth=0.0,
            intra_wave_gap_steps=0,
        ),
    ),
    Anchor(
        "A09_formation_deep", "窄深编组", "formation",
        "仅改成窄正面、大纵深和较大波内间隔。",
        strike_formation=replace(
            BASE_STRIKE_FORMATION,
            lateral_width=0.15,
            longitudinal_depth=1.0,
            intra_wave_gap_steps=8,
        ),
    ),
    Anchor(
        "A10_l_screen", "L 前沿屏障", "combined",
        "L 前沿部署并在主攻波内先行，H/M 相对后置。",
        deployment=replace(
            BASE_DEPLOYMENT,
            h_depth_offset=-0.35,
            m_depth_offset=-0.35,
            l_depth_offset=0.80,
        ),
        strike_formation=replace(
            BASE_STRIKE_FORMATION,
            intra_wave_gap_steps=8,
            role_order=(Role.L, Role.M, Role.H),
        ),
    ),
    Anchor(
        "A11_hm_spearhead", "H/M 前置", "combined_control",
        "H/M 前置先行、L 后置，作为 L 前沿屏障的反向对照。",
        deployment=replace(
            BASE_DEPLOYMENT,
            h_depth_offset=0.60,
            m_depth_offset=0.60,
            l_depth_offset=-0.80,
        ),
        strike_formation=replace(
            BASE_STRIKE_FORMATION,
            role_order=(Role.H, Role.M, Role.L),
        ),
    ),
)


def _scenario_dirs() -> tuple[Path, ...]:
    return tuple(sorted(
        path.parent
        for path in CASES_ROOT.glob("*/*/case_info.json")
    ))


def _public_catalogue(case_info: dict) -> dict[int, int]:
    kind_to_type = {"target": 9400, "site": 9600}
    result = {}
    for root in case_info["roots"]:
        if root.get("initial_information") != "public":
            continue
        kind = str(root["kind"])
        if kind not in kind_to_type:
            raise ValueError(f"unknown public root kind: {kind}")
        result[int(root["id"])] = kind_to_type[kind]
    return result


def _weighted_target_counts(
    total: int,
    public_targets: dict[int, int],
    preferred_type: int,
) -> dict[int, int]:
    """Allocate every role unit while giving every public target support."""

    targets = sorted(public_targets)
    if total < len(targets):
        raise ValueError(
            f"role count {total} cannot cover {len(targets)} public targets"
        )
    counts = {target_id: 1 for target_id in targets}
    weights = {
        target_id: (3 if public_targets[target_id] == preferred_type else 1)
        for target_id in targets
    }
    for _ in range(total - len(targets)):
        target_id = min(
            targets,
            key=lambda value: (
                counts[value] / weights[value],
                0 if public_targets[value] == preferred_type else 1,
                value,
            ),
        )
        counts[target_id] += 1
    return counts


def _split_count(count: int, phase: int) -> tuple[int, int, int]:
    quotient, remainder = divmod(int(count), 3)
    values = [quotient, quotient, quotient]
    for offset in range(remainder):
        values[(phase + offset) % 3] += 1
    return tuple(values)


def _strike_splits(
    counts: RoleCounts,
    public_targets: dict[int, int],
) -> dict[Role, dict[int, tuple[int, int, int]]]:
    by_role = {
        Role.H: _weighted_target_counts(counts.h, public_targets, 9400),
        Role.M: _weighted_target_counts(counts.m, public_targets, 9600),
    }
    result: dict[Role, dict[int, tuple[int, int, int]]] = {}
    for role_index, role in enumerate((Role.H, Role.M)):
        result[role] = {
            target_id: _split_count(count, target_index + role_index)
            for target_index, (target_id, count) in enumerate(
                sorted(by_role[role].items())
            )
        }
    return result


def _strike_allocations(
    wave_index: int,
    splits: dict[Role, dict[int, tuple[int, int, int]]],
    public_targets: dict[int, int],
) -> tuple[GoalAllocation, ...]:
    allocations = []
    for role in (Role.H, Role.M):
        for target_id, split in sorted(splits[role].items()):
            count = split[wave_index]
            if count:
                allocations.append(GoalAllocation(
                    role=role,
                    goal=GoalRef.public_target(
                        target_id, public_targets[target_id]
                    ),
                    count=count,
                ))
    return tuple(allocations)


def _build_plan(
    anchor: Anchor,
    *,
    counts: RoleCounts,
    public_targets: dict[int, int],
    hidden_count: int,
    scout_l: int,
) -> PlanSpec:
    if hidden_count < 2:
        raise ValueError("all-discoveries plans require at least two hidden targets")
    if not 0 < scout_l < counts.l:
        raise ValueError("scout count must leave a positive L response reserve")
    response_l = counts.l - scout_l
    if response_l < hidden_count:
        raise ValueError("response reserve must cover every hidden target")

    strike_splits = _strike_splits(counts, public_targets)
    waves = [WaveSpec(
        wave_id=0,
        release_step=0,
        formation_id="scout",
        allocations=(GoalAllocation(
            role=Role.L,
            goal=GoalRef.hidden_search("legal_grid_search_v1"),
            count=scout_l,
        ),),
    )]
    for wave_index, release_step in enumerate(anchor.release_steps):
        waves.append(WaveSpec(
            wave_id=wave_index + 1,
            release_step=release_step,
            formation_id="strike",
            allocations=_strike_allocations(
                wave_index, strike_splits, public_targets
            ),
        ))
    plan = PlanSpec(
        deployment=anchor.deployment or BASE_DEPLOYMENT,
        formations=(
            SCOUT_FORMATION,
            anchor.strike_formation or BASE_STRIKE_FORMATION,
            RESPONSE_FORMATION,
        ),
        waves=tuple(waves),
        reserves=RoleCounts(l=response_l),
        hidden_response=HiddenResponseSpec(
            mode=HiddenResponseMode.RESERVE_L_AFTER_ALL_DISCOVERIES,
            reserve_l_count=response_l,
            release_delay_steps=20,
            formation_id="hidden_response",
            selector=HiddenTargetSelector.ALL_LEGAL_DISCOVERIES,
            required_discoveries=hidden_count,
        ),
        schema_version=2,
    )
    plan.validate_for_context(
        available_counts=counts.as_dict(),
        public_targets=public_targets,
        max_steps=MAX_STEPS,
    )
    return plan


def _role_counts(case_info: dict) -> RoleCounts:
    raw = case_info["red_missile_counts"]
    return RoleCounts(
        h=int(raw["high"]),
        m=int(raw["medium"]),
        l=int(raw["low"]),
    )


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    map_rows = []
    for scenario_dir in _scenario_dirs():
        case_info_path = scenario_dir / "case_info.json"
        case_info = json.loads(case_info_path.read_text(encoding="utf-8"))
        scenario_id = str(case_info["scenario_id"])
        difficulty = str(case_info["difficulty"])
        scenario = f"final20/{difficulty}/{scenario_id}"
        counts = _role_counts(case_info)
        public_targets = _public_catalogue(case_info)
        hidden_count = len(case_info["hidden_objective_ids"])
        scout_l = SCOUTS_BY_DIFFICULTY[difficulty]
        map_dir = output / scenario_id
        map_dir.mkdir(parents=True, exist_ok=True)

        anchors = []
        for anchor in ANCHORS:
            plan = _build_plan(
                anchor,
                counts=counts,
                public_targets=public_targets,
                hidden_count=hidden_count,
                scout_l=scout_l,
            )
            filename = f"{anchor.anchor_id}.json"
            (map_dir / filename).write_text(
                plan.canonical_token_json() + "\n", encoding="utf-8"
            )
            anchors.append({
                **asdict(anchor),
                "release_steps": list(anchor.release_steps),
                "deployment": asdict(plan.deployment),
                "strike_formation": asdict(next(
                    item for item in plan.formations
                    if item.formation_id == "strike"
                )),
                "plan_file": filename,
                "plan_id": plan.plan_id,
                "accounted_counts": plan.accounted_counts.as_dict(),
            })

        manifest = {
            "schema_version": 2,
            "experiment": "plan_anchor_sensitivity_final20_all_v2",
            "scenario": scenario,
            "scenario_id": scenario_id,
            "difficulty": difficulty,
            "title": case_info.get("title", scenario_id),
            "max_steps": MAX_STEPS,
            "common_seeds": list(COMMON_SEEDS),
            "blue_policy": "b0_fixed_ratio_random",
            "blue_interceptor_ratio": 2,
            "red_motion_policy": "local_avoid_s3",
            "satellite_rule": "fixed_lowest_live_unused_h",
            "search_policy": "legal_grid_search_v1",
            "fixed_goal_contract": {
                "public_target_types": {
                    str(key): value for key, value in public_targets.items()
                },
                "public_attack_h": counts.h,
                "public_attack_m": counts.m,
                "public_attack_l": 0,
                "scout_l": scout_l,
                "all_hidden_response_l": counts.l - scout_l,
                "required_hidden_discoveries": hidden_count,
                "permanent_hold_l": 0,
                "retarget_during_episode": False,
            },
            "anchors": anchors,
        }
        (map_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        map_rows.append({
            "scenario": scenario,
            "scenario_id": scenario_id,
            "difficulty": difficulty,
            "title": case_info.get("title", scenario_id),
            "manifest": str((map_dir / "manifest.json").relative_to(output)),
            "red_counts": counts.as_dict(),
            "public_target_count": len(public_targets),
            "hidden_target_count": hidden_count,
            "scout_l": scout_l,
            "response_l": counts.l - scout_l,
        })

    root_manifest = {
        "schema_version": 2,
        "experiment": "plan_anchor_sensitivity_final20_all_v2",
        "common_seeds": list(COMMON_SEEDS),
        "anchor_count": len(ANCHORS),
        "map_count": len(map_rows),
        "expected_runs": len(ANCHORS) * len(map_rows) * len(COMMON_SEEDS),
        "maps": map_rows,
        "anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "label": anchor.label,
                "factor": anchor.factor,
                "description": anchor.description,
            }
            for anchor in ANCHORS
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = generate(args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "map_count": manifest["map_count"],
        "anchor_count": manifest["anchor_count"],
        "expected_runs": manifest["expected_runs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
