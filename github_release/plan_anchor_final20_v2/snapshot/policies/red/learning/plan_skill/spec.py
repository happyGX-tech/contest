"""Versioned PlanSpec and its canonical, human-readable token sentence.

The schema is intentionally independent of the simulator and Torch.  It can be
reviewed, generated and validated before any plan is allowed to control an
episode.  It also enforces the partial-observation boundary: pre-mission goal
tokens may name only public 9400/9600 targets; hidden 9500 coordinates or IDs
can never appear in a pre-mission plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Iterable, Mapping


PLAN_SCHEMA_VERSION = 2
SUPPORTED_PLAN_SCHEMA_VERSIONS = frozenset((1, 2))
PUBLIC_TARGET_TYPES = frozenset((9400, 9600))
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}$")


class Role(str, Enum):
    H = "H"
    M = "M"
    L = "L"


ROLE_ORDER = (Role.H, Role.M, Role.L)


class DeploymentTemplate(str, Enum):
    COMPACT = "compact"
    LATERAL_LINE = "lateral_line"
    ROLE_LAYERED = "role_layered"


class FormationTemplate(str, Enum):
    COMPACT = "compact"
    LATERAL_LINE = "lateral_line"
    ROLE_LAYERED = "role_layered"


class GoalKind(str, Enum):
    PUBLIC_TARGET = "public_target"
    HIDDEN_SEARCH = "hidden_search"


class HiddenResponseMode(str, Enum):
    NONE = "none"
    RESERVE_L_AFTER_DISCOVERY = "reserve_l_after_discovery"
    RESERVE_L_AFTER_ALL_DISCOVERIES = "reserve_l_after_all_discoveries"


class HiddenTargetSelector(str, Enum):
    FIRST_LEGAL_DISCOVERY = "first_legal_discovery"
    ALL_LEGAL_DISCOVERIES = "all_legal_discoveries"


class PlanTokenOp(str, Enum):
    PLAN = "PLAN"
    DEPLOY = "DEPLOY"
    FORMATION = "FORMATION"
    WAVE = "WAVE"
    ASSIGN = "ASSIGN"
    RESERVE = "RESERVE"
    ON_DISCOVERY = "ON_DISCOVERY"
    END = "END"


TokenScalar = str | int | float


def _finite_number(name: str, value: float, low: float, high: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    if not low <= float(value) <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")


def _non_negative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _identifier(name: str, value: str, *, allow_empty: bool = False) -> None:
    if allow_empty and value == "":
        return
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{name} must match {_IDENTIFIER.pattern!r}; got {value!r}"
        )


@dataclass(frozen=True)
class RoleCounts:
    """How many H/M/L units a plan accounts for."""

    h: int = 0
    m: int = 0
    l: int = 0

    def __post_init__(self) -> None:
        _non_negative_int("h", self.h)
        _non_negative_int("m", self.m)
        _non_negative_int("l", self.l)

    def for_role(self, role: Role) -> int:
        return {Role.H: self.h, Role.M: self.m, Role.L: self.l}[Role(role)]

    def plus(self, other: "RoleCounts") -> "RoleCounts":
        return RoleCounts(
            self.h + other.h,
            self.m + other.m,
            self.l + other.l,
        )

    def as_dict(self) -> dict[str, int]:
        return {"H": self.h, "M": self.m, "L": self.l}


@dataclass(frozen=True)
class DeploymentSpec:
    """Low-dimensional placement template in a public-target-relative frame.

    ``forward`` is the axis from the deployment-area centre towards the public
    target centroid and ``lateral`` is its perpendicular.  Anchors and role
    offsets use [-1, 1]; spans use [0, 1].  The later executor will map this
    normalized description to the existing 8 x 4 legal deployment grid.
    """

    template: DeploymentTemplate
    anchor_forward: float
    anchor_lateral: float
    frontage: float
    depth: float
    h_depth_offset: float = 0.0
    m_depth_offset: float = 0.0
    l_depth_offset: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "template", DeploymentTemplate(self.template))
        _finite_number("anchor_forward", self.anchor_forward, -1.0, 1.0)
        _finite_number("anchor_lateral", self.anchor_lateral, -1.0, 1.0)
        _finite_number("frontage", self.frontage, 0.0, 1.0)
        _finite_number("depth", self.depth, 0.0, 1.0)
        _finite_number("h_depth_offset", self.h_depth_offset, -1.0, 1.0)
        _finite_number("m_depth_offset", self.m_depth_offset, -1.0, 1.0)
        _finite_number("l_depth_offset", self.l_depth_offset, -1.0, 1.0)


@dataclass(frozen=True)
class FormationSpec:
    """Reusable formation referenced by one or more waves."""

    formation_id: str
    template: FormationTemplate
    lateral_width: float
    longitudinal_depth: float
    intra_wave_gap_steps: int = 0
    role_order: tuple[Role, Role, Role] = ROLE_ORDER

    def __post_init__(self) -> None:
        _identifier("formation_id", self.formation_id)
        object.__setattr__(self, "template", FormationTemplate(self.template))
        _finite_number("lateral_width", self.lateral_width, 0.0, 1.0)
        _finite_number(
            "longitudinal_depth", self.longitudinal_depth, 0.0, 1.0
        )
        _non_negative_int("intra_wave_gap_steps", self.intra_wave_gap_steps)
        order = tuple(Role(value) for value in self.role_order)
        if len(order) != 3 or set(order) != set(ROLE_ORDER):
            raise ValueError("role_order must be one permutation of H, M, L")
        object.__setattr__(self, "role_order", order)


@dataclass(frozen=True)
class GoalRef:
    """A legal pre-mission goal reference.

    Public targets preserve the real entity ID for exact replay.  A learned
    model must represent that ID through the current candidate embedding, not
    treat the integer itself as a transferable feature.  Hidden search names a
    frozen lower-level search policy and contains no hidden target ID/position.
    """

    kind: GoalKind
    target_id: int = -1
    target_type: int = -1
    search_policy_id: str = ""
    search_cells: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GoalKind(self.kind))
        cells = tuple(self.search_cells)
        if any(type(cell) is not int or not 0 <= cell < 32 for cell in cells):
            raise ValueError("search_cells must contain legal integer cells 0..31")
        if len(set(cells)) != len(cells):
            raise ValueError("search_cells must be unique")
        object.__setattr__(self, "search_cells", cells)
        if self.kind == GoalKind.PUBLIC_TARGET:
            _non_negative_int("target_id", self.target_id)
            if type(self.target_type) is not int \
                    or self.target_type not in PUBLIC_TARGET_TYPES:
                raise ValueError(
                    "pre-mission public targets must have type 9400 or 9600"
                )
            if self.search_policy_id:
                raise ValueError("public target goals cannot name a search policy")
            if self.search_cells:
                raise ValueError("public target goals cannot contain search cells")
        else:
            if self.target_id != -1 or self.target_type != -1:
                raise ValueError(
                    "hidden-search goals cannot contain a target ID or type"
                )
            _identifier("search_policy_id", self.search_policy_id)

    @classmethod
    def public_target(cls, target_id: int, target_type: int) -> "GoalRef":
        return cls(
            GoalKind.PUBLIC_TARGET,
            target_id=target_id,
            target_type=target_type,
        )

    @classmethod
    def hidden_search(
        cls, search_policy_id: str, search_cells: Iterable[int] = ()
    ) -> "GoalRef":
        return cls(
            GoalKind.HIDDEN_SEARCH,
            search_policy_id=search_policy_id,
            search_cells=tuple(search_cells),
        )

    @property
    def canonical_key(self) -> tuple[str, int, int, str, tuple[int, ...]]:
        return (
            self.kind.value,
            self.target_type,
            self.target_id,
            self.search_policy_id,
            self.search_cells,
        )


@dataclass(frozen=True)
class GoalAllocation:
    """Assign ``count`` units of one role to one immutable goal."""

    role: Role
    goal: GoalRef
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("allocation count must be a positive integer")
        if self.goal.kind == GoalKind.HIDDEN_SEARCH and self.role == Role.M:
            raise ValueError("hidden-search goals are legal only for H or L units")
        if self.goal.search_cells and len(self.goal.search_cells) != self.count:
            raise ValueError(
                "explicit search_cells length must equal allocation count"
            )


@dataclass(frozen=True)
class WaveSpec:
    """One release event with fixed, no-retarget goal allocations."""

    wave_id: int
    release_step: int
    formation_id: str
    allocations: tuple[GoalAllocation, ...]

    def __post_init__(self) -> None:
        _non_negative_int("wave_id", self.wave_id)
        _non_negative_int("release_step", self.release_step)
        _identifier("formation_id", self.formation_id)
        allocations = tuple(self.allocations)
        if not allocations:
            raise ValueError("a wave must contain at least one allocation")
        keys = [(item.role, item.goal.canonical_key) for item in allocations]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate role/goal allocation in one wave")
        object.__setattr__(self, "allocations", allocations)

    @property
    def counts(self) -> RoleCounts:
        values = {
            role: sum(item.count for item in self.allocations if item.role == role)
            for role in ROLE_ORDER
        }
        return RoleCounts(values[Role.H], values[Role.M], values[Role.L])


@dataclass(frozen=True)
class HiddenResponseSpec:
    """Pre-committed response using only still-unreleased reserve L units."""

    mode: HiddenResponseMode = HiddenResponseMode.NONE
    reserve_l_count: int = 0
    release_delay_steps: int = 0
    formation_id: str = ""
    selector: HiddenTargetSelector = HiddenTargetSelector.FIRST_LEGAL_DISCOVERY
    required_discoveries: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", HiddenResponseMode(self.mode))
        object.__setattr__(self, "selector", HiddenTargetSelector(self.selector))
        _non_negative_int("reserve_l_count", self.reserve_l_count)
        _non_negative_int("release_delay_steps", self.release_delay_steps)
        _non_negative_int("required_discoveries", self.required_discoveries)
        if self.mode == HiddenResponseMode.NONE:
            if (self.reserve_l_count or self.release_delay_steps
                    or self.formation_id or self.required_discoveries):
                raise ValueError("disabled hidden response must have zero/empty fields")
        elif self.mode == HiddenResponseMode.RESERVE_L_AFTER_DISCOVERY:
            if self.reserve_l_count <= 0:
                raise ValueError("enabled hidden response needs reserve_l_count > 0")
            _identifier("formation_id", self.formation_id)
            if self.selector != HiddenTargetSelector.FIRST_LEGAL_DISCOVERY \
                    or self.required_discoveries != 1:
                raise ValueError(
                    "first-discovery response requires selector "
                    "first_legal_discovery and required_discoveries=1"
                )
        else:
            if self.reserve_l_count <= 0:
                raise ValueError("enabled hidden response needs reserve_l_count > 0")
            _identifier("formation_id", self.formation_id)
            if self.selector != HiddenTargetSelector.ALL_LEGAL_DISCOVERIES \
                    or self.required_discoveries < 2:
                raise ValueError(
                    "all-discoveries response requires selector "
                    "all_legal_discoveries and required_discoveries>=2"
                )
            if self.reserve_l_count < self.required_discoveries:
                raise ValueError(
                    "all-discoveries response needs at least one reserve L "
                    "per required discovery"
                )


@dataclass(frozen=True)
class PlanToken:
    """One canonical symbolic token with primitive named fields."""

    op: PlanTokenOp
    fields: tuple[tuple[str, TokenScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", PlanTokenOp(self.op))
        fields = tuple(self.fields)
        names = [name for name, _ in fields]
        if len(set(names)) != len(names):
            raise ValueError("token fields must have unique names")
        for name, value in fields:
            if not isinstance(name, str) or not name:
                raise ValueError("token field names must be non-empty strings")
            if type(value) not in (str, int, float):
                raise ValueError("token values must be str, int or float")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("token float values must be finite")
        object.__setattr__(self, "fields", tuple(sorted(fields)))

    @classmethod
    def make(cls, op: PlanTokenOp, **fields: TokenScalar) -> "PlanToken":
        return cls(op, tuple(fields.items()))

    def to_dict(self) -> dict[str, TokenScalar]:
        return {"op": self.op.value, **dict(self.fields)}

    def require(self, *names: str) -> dict[str, TokenScalar]:
        data = dict(self.fields)
        if set(data) != set(names):
            raise ValueError(
                f"{self.op.value} fields must be {sorted(names)}; "
                f"got {sorted(data)}"
            )
        return data


@dataclass(frozen=True)
class PlanSpec:
    """A complete, immutable, pre-mission plan contract."""

    deployment: DeploymentSpec
    formations: tuple[FormationSpec, ...]
    waves: tuple[WaveSpec, ...]
    reserves: RoleCounts = RoleCounts()
    hidden_response: HiddenResponseSpec = HiddenResponseSpec()
    schema_version: int = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int \
                or self.schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema_version must be one of "
                f"{sorted(SUPPORTED_PLAN_SCHEMA_VERSIONS)}"
            )
        if self.schema_version == 1 and self.hidden_response.mode \
                == HiddenResponseMode.RESERVE_L_AFTER_ALL_DISCOVERIES:
            raise ValueError("all-discoveries response requires schema_version=2")
        if self.schema_version == 1 and any(
            allocation.goal.kind == GoalKind.HIDDEN_SEARCH
            and (
                allocation.role != Role.L
                or bool(allocation.goal.search_cells)
            )
            for wave in self.waves
            for allocation in wave.allocations
        ):
            raise ValueError(
                "H or explicit-cell hidden-search assignments require "
                "schema_version=2"
            )
        formations = tuple(self.formations)
        waves = tuple(self.waves)
        if not formations:
            raise ValueError("a plan must define at least one formation")
        if not waves:
            raise ValueError("a plan must define at least one wave")
        formation_ids = [item.formation_id for item in formations]
        if len(set(formation_ids)) != len(formation_ids):
            raise ValueError("formation IDs must be unique")
        wave_ids = [item.wave_id for item in waves]
        if len(set(wave_ids)) != len(wave_ids):
            raise ValueError("wave IDs must be unique")
        unknown = {
            item.formation_id for item in waves
            if item.formation_id not in formation_ids
        }
        if self.hidden_response.mode != HiddenResponseMode.NONE \
                and self.hidden_response.formation_id not in formation_ids:
            unknown.add(self.hidden_response.formation_id)
        if unknown:
            raise ValueError(f"unknown formation references: {sorted(unknown)}")
        if self.hidden_response.reserve_l_count > self.reserves.l:
            raise ValueError("hidden response exceeds explicitly held L reserves")
        # Store the same canonical order emitted by ``to_tokens`` so a token
        # round trip preserves dataclass equality as well as the content hash.
        object.__setattr__(
            self, "formations",
            tuple(sorted(formations, key=lambda item: item.formation_id)),
        )
        object.__setattr__(
            self, "waves",
            tuple(sorted(waves, key=lambda item: (item.release_step, item.wave_id))),
        )

    @property
    def allocated_counts(self) -> RoleCounts:
        result = RoleCounts()
        for wave in self.waves:
            result = result.plus(wave.counts)
        return result

    @property
    def accounted_counts(self) -> RoleCounts:
        return self.allocated_counts.plus(self.reserves)

    def validate_for_context(
        self,
        *,
        available_counts: Mapping[Role | str, int],
        public_targets: Mapping[int, int],
        max_steps: int,
    ) -> None:
        """Validate episode-dependent counts, targets and time bounds."""

        _non_negative_int("max_steps", max_steps)
        if max_steps == 0:
            raise ValueError("max_steps must be positive")
        normalized_counts: dict[Role, int] = {}
        for role in ROLE_ORDER:
            if role in available_counts:
                value = available_counts[role]
            elif role.value in available_counts:
                value = available_counts[role.value]
            else:
                raise ValueError(f"available_counts is missing {role.value}")
            _non_negative_int(f"available_counts[{role.value}]", value)
            normalized_counts[role] = value
        accounted = self.accounted_counts
        for role in ROLE_ORDER:
            if accounted.for_role(role) != normalized_counts[role]:
                raise ValueError(
                    f"plan accounts for {accounted.for_role(role)} {role.value} "
                    f"units but context provides {normalized_counts[role]}"
                )
        for wave in self.waves:
            if wave.release_step >= max_steps:
                raise ValueError(
                    f"wave {wave.wave_id} release_step must be below max_steps"
                )
            for allocation in wave.allocations:
                goal = allocation.goal
                if goal.kind != GoalKind.PUBLIC_TARGET:
                    continue
                actual_type = public_targets.get(goal.target_id)
                if actual_type is None:
                    raise ValueError(
                        f"target {goal.target_id} is absent from the public catalogue"
                    )
                if type(actual_type) is not int or actual_type != goal.target_type:
                    raise ValueError(
                        f"target {goal.target_id} type mismatch: "
                        f"plan={goal.target_type}, context={actual_type}"
                    )

    def to_tokens(self) -> tuple[PlanToken, ...]:
        """Return the canonical plan sentence; no simulator state is consulted."""

        tokens = [PlanToken.make(
            PlanTokenOp.PLAN, schema_version=self.schema_version)]
        d = self.deployment
        tokens.append(PlanToken.make(
            PlanTokenOp.DEPLOY,
            template=d.template.value,
            anchor_forward=float(d.anchor_forward),
            anchor_lateral=float(d.anchor_lateral),
            frontage=float(d.frontage),
            depth=float(d.depth),
            h_depth_offset=float(d.h_depth_offset),
            m_depth_offset=float(d.m_depth_offset),
            l_depth_offset=float(d.l_depth_offset),
        ))
        for formation in sorted(self.formations, key=lambda item: item.formation_id):
            tokens.append(PlanToken.make(
                PlanTokenOp.FORMATION,
                formation_id=formation.formation_id,
                template=formation.template.value,
                lateral_width=float(formation.lateral_width),
                longitudinal_depth=float(formation.longitudinal_depth),
                intra_wave_gap_steps=formation.intra_wave_gap_steps,
                role_order=",".join(role.value for role in formation.role_order),
            ))
        for wave in sorted(
            self.waves, key=lambda item: (item.release_step, item.wave_id)
        ):
            tokens.append(PlanToken.make(
                PlanTokenOp.WAVE,
                wave_id=wave.wave_id,
                release_step=wave.release_step,
                formation_id=wave.formation_id,
            ))
            for allocation in sorted(
                wave.allocations,
                key=lambda item: (
                    ROLE_ORDER.index(item.role), item.goal.canonical_key
                ),
            ):
                fields: dict[str, TokenScalar] = {
                    "wave_id": wave.wave_id,
                    "role": allocation.role.value,
                    "count": allocation.count,
                    "goal_kind": allocation.goal.kind.value,
                }
                if allocation.goal.kind == GoalKind.PUBLIC_TARGET:
                    fields.update(
                        target_id=allocation.goal.target_id,
                        target_type=allocation.goal.target_type,
                    )
                else:
                    fields["search_policy_id"] = allocation.goal.search_policy_id
                    if allocation.goal.search_cells:
                        fields["search_cells"] = ",".join(
                            str(cell) for cell in allocation.goal.search_cells
                        )
                tokens.append(PlanToken.make(PlanTokenOp.ASSIGN, **fields))
        tokens.append(PlanToken.make(
            PlanTokenOp.RESERVE,
            h=self.reserves.h,
            m=self.reserves.m,
            l=self.reserves.l,
        ))
        response = self.hidden_response
        response_fields: dict[str, TokenScalar] = {
            "mode": response.mode.value,
            "reserve_l_count": response.reserve_l_count,
            "release_delay_steps": response.release_delay_steps,
            "formation_id": response.formation_id,
            "selector": response.selector.value,
        }
        if self.schema_version >= 2:
            response_fields["required_discoveries"] = response.required_discoveries
        tokens.append(PlanToken.make(PlanTokenOp.ON_DISCOVERY, **response_fields))
        tokens.append(PlanToken.make(PlanTokenOp.END))
        return tuple(tokens)

    @classmethod
    def from_tokens(cls, tokens: Iterable[PlanToken | Mapping]) -> "PlanSpec":
        """Strictly parse a supported grammar and reject unknown tokens."""

        parsed = tuple(
            token if isinstance(token, PlanToken) else _token_from_mapping(token)
            for token in tokens
        )
        if len(parsed) < 7:
            raise ValueError("plan sentence is incomplete")
        if parsed[0].op != PlanTokenOp.PLAN or parsed[-1].op != PlanTokenOp.END:
            raise ValueError("plan sentence must start with PLAN and end with END")
        header = parsed[0].require("schema_version")
        schema_version = header["schema_version"]
        if type(schema_version) is not int \
                or schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
            raise ValueError("unsupported plan schema version")
        parsed[-1].require()
        if parsed[1].op != PlanTokenOp.DEPLOY:
            raise ValueError("DEPLOY must immediately follow PLAN")
        raw = parsed[1].require(
            "template", "anchor_forward", "anchor_lateral", "frontage", "depth",
            "h_depth_offset", "m_depth_offset", "l_depth_offset",
        )
        deployment = DeploymentSpec(
            template=DeploymentTemplate(str(raw["template"])),
            anchor_forward=float(raw["anchor_forward"]),
            anchor_lateral=float(raw["anchor_lateral"]),
            frontage=float(raw["frontage"]),
            depth=float(raw["depth"]),
            h_depth_offset=float(raw["h_depth_offset"]),
            m_depth_offset=float(raw["m_depth_offset"]),
            l_depth_offset=float(raw["l_depth_offset"]),
        )
        index = 2
        formations = []
        while index < len(parsed) and parsed[index].op == PlanTokenOp.FORMATION:
            raw = parsed[index].require(
                "formation_id", "template", "lateral_width",
                "longitudinal_depth", "intra_wave_gap_steps", "role_order",
            )
            formations.append(FormationSpec(
                formation_id=str(raw["formation_id"]),
                template=FormationTemplate(str(raw["template"])),
                lateral_width=float(raw["lateral_width"]),
                longitudinal_depth=float(raw["longitudinal_depth"]),
                intra_wave_gap_steps=int(raw["intra_wave_gap_steps"]),
                role_order=tuple(
                    Role(value) for value in str(raw["role_order"]).split(",")
                ),
            ))
            index += 1
        waves = []
        while index < len(parsed) and parsed[index].op == PlanTokenOp.WAVE:
            raw_wave = parsed[index].require(
                "wave_id", "release_step", "formation_id"
            )
            wave_id = int(raw_wave["wave_id"])
            index += 1
            allocations = []
            while index < len(parsed) and parsed[index].op == PlanTokenOp.ASSIGN:
                values = dict(parsed[index].fields)
                if int(values.get("wave_id", -1)) != wave_id:
                    raise ValueError("ASSIGN wave_id must match its preceding WAVE")
                kind = GoalKind(str(values.get("goal_kind", "")))
                if kind == GoalKind.PUBLIC_TARGET:
                    raw_assign = parsed[index].require(
                        "wave_id", "role", "count", "goal_kind",
                        "target_id", "target_type",
                    )
                    goal = GoalRef.public_target(
                        int(raw_assign["target_id"]),
                        int(raw_assign["target_type"]),
                    )
                else:
                    hidden_fields = [
                        "wave_id", "role", "count", "goal_kind",
                        "search_policy_id",
                    ]
                    if "search_cells" in values:
                        hidden_fields.append("search_cells")
                    raw_assign = parsed[index].require(*hidden_fields)
                    cells_raw = str(raw_assign.get("search_cells", ""))
                    goal = GoalRef.hidden_search(
                        str(raw_assign["search_policy_id"]),
                        (() if not cells_raw else tuple(
                            int(value) for value in cells_raw.split(",")
                        )),
                    )
                allocations.append(GoalAllocation(
                    Role(str(raw_assign["role"])),
                    goal,
                    int(raw_assign["count"]),
                ))
                index += 1
            waves.append(WaveSpec(
                wave_id=wave_id,
                release_step=int(raw_wave["release_step"]),
                formation_id=str(raw_wave["formation_id"]),
                allocations=tuple(allocations),
            ))
        if index >= len(parsed) or parsed[index].op != PlanTokenOp.RESERVE:
            raise ValueError("RESERVE must follow all waves")
        raw = parsed[index].require("h", "m", "l")
        reserves = RoleCounts(int(raw["h"]), int(raw["m"]), int(raw["l"]))
        index += 1
        if index >= len(parsed) or parsed[index].op != PlanTokenOp.ON_DISCOVERY:
            raise ValueError("ON_DISCOVERY must follow RESERVE")
        response_fields = [
            "mode", "reserve_l_count", "release_delay_steps",
            "formation_id", "selector",
        ]
        if schema_version >= 2:
            response_fields.append("required_discoveries")
        raw = parsed[index].require(*response_fields)
        mode = HiddenResponseMode(str(raw["mode"]))
        required_discoveries = (
            int(raw["required_discoveries"])
            if schema_version >= 2 else
            (1 if mode == HiddenResponseMode.RESERVE_L_AFTER_DISCOVERY else 0)
        )
        hidden_response = HiddenResponseSpec(
            mode=mode,
            reserve_l_count=int(raw["reserve_l_count"]),
            release_delay_steps=int(raw["release_delay_steps"]),
            formation_id=str(raw["formation_id"]),
            selector=HiddenTargetSelector(str(raw["selector"])),
            required_discoveries=required_discoveries,
        )
        index += 1
        if index != len(parsed) - 1:
            raise ValueError("unexpected token before END")
        return cls(
            deployment=deployment,
            formations=tuple(formations),
            waves=tuple(waves),
            reserves=reserves,
            hidden_response=hidden_response,
            schema_version=int(schema_version),
        )

    def canonical_token_json(self) -> str:
        return json.dumps(
            [token.to_dict() for token in self.to_tokens()],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def plan_id(self) -> str:
        """Content address for semantic plan fields (not run metadata)."""

        return hashlib.sha256(
            self.canonical_token_json().encode("utf-8")
        ).hexdigest()


def _token_from_mapping(raw: Mapping) -> PlanToken:
    if not isinstance(raw, Mapping) or "op" not in raw:
        raise ValueError("each token mapping needs an op field")
    return PlanToken(
        PlanTokenOp(str(raw["op"])),
        tuple((str(key), value) for key, value in raw.items() if key != "op"),
    )
