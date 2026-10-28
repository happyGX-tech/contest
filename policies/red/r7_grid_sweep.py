"""Eastbound grid sweep: every scanning L platform crosses one latitude lane.

The static-search layer in :mod:`policies.red.r7_static_search` sends three
platforms to a single fan destination.  That geometry is cheap but cannot cover
a map: nine searchers on diverging bearings leave tens of kilometres between
adjacent sweeps.  This layer uses the shape of the red launch area instead.  The
low-performance (L) launchers already occupy a narrow band that spans the whole
map latitude, so a *wall* of eastbound legs is both simpler and far denser:

* each scanning platform keeps its own latitude and flies due east;
* lanes are subsampled from the sorted launch latitudes so adjacent lanes stay
  inside two detection radii (``detection_radius_km`` 3 km -> 6 km spacing);
* when a platform legally detects a hidden ship it may retarget it in flight
  (core action ``2``), and the discovered ship enters the shared track fusion
  catalogue so later L waves can be assigned to it;
* the ordinary strike packages resume after a short lead, so the sweep does not
  consume the whole battle clock.

No hidden position is used to build the lanes: the geometry comes from the
public map polygon and the red launch band only.

Two policies use the layer: ``r7_grid_sweep`` (deterministic R7 strike packages
on straight motion, tested as a pure search experiment) and
``r9_hierarchical_learning`` when ``R9_GRID_SWEEP_PLATFORMS`` is set (same
sweep, but the learned evasion policy keeps the scanners alive underneath).
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from math import ceil

from .baselines import (
    Assignment,
    BaselineObservation,
    BaselineRules,
    PlatformState,
    R7StrikePackagePolicy,
    TargetPrior,
    distance_km,
)
from .contracts import Position


logger = logging.getLogger(__name__)

_DETECTION_RADIUS_KM = 3.0
_MAX_SWEEP_RANGE_KM = 200.0
# The 10-19 engine rejects search targets outside the scenario map area, so a
# sweep ends just inside the east edge instead of overshooting it.
_EAST_EDGE_INSET_DEGREES = 0.001
_DIRECT_RETARGET_RANGE_KM = 10.0
_LANE_SPACING_KM = 2.0 * _DETECTION_RADIUS_KM


class GridSweepLayer:
    """Lane geometry, launch decoration and in-flight ship retargeting.

    The owning policy keeps its own target scoring; the layer only decides which
    L platforms become scanners, where their lane ends, when the ordinary policy
    may resume, and which detected ship a scanner may claim.
    """

    def __init__(
        self,
        search_polygon: tuple[Position, ...],
        sweep_platform_target: int = 36,
        sweep_lead_steps: int = 240,
        max_retargets_per_platform: int = 2,
        max_claims_per_ship: int = 3,
    ) -> None:
        if len(search_polygon) < 3:
            raise ValueError("grid sweep requires a public map polygon")
        if sweep_platform_target < 1:
            raise ValueError("sweep_platform_target must be positive")
        if sweep_lead_steps < 1:
            raise ValueError("sweep_lead_steps must be positive")
        self.search_polygon = search_polygon
        self.sweep_platform_target = sweep_platform_target
        self.sweep_lead_steps = sweep_lead_steps
        self.max_retargets_per_platform = max_retargets_per_platform
        self.max_claims_per_ship = max_claims_per_ship
        # Which clock anchors the follow-on wave.  "deployment" (default) treats
        # the lead as time since step 0; "plan" keeps the historical
        # step-relative behaviour for A/B comparison against the old policy.
        self.follow_on_anchor = os.getenv("R7_FOLLOW_ON_ANCHOR", "deployment").strip().lower()
        if self.follow_on_anchor not in {"deployment", "plan"}:
            raise ValueError("R7_FOLLOW_ON_ANCHOR must be 'deployment' or 'plan'")
        self._planned = False
        self._follow_on_step = 0
        self._sweep_platform_ids: frozenset[int] = frozenset()
        self._retarget_counts: dict[int, int] = {}
        self._ship_claims: dict[int, int] = {}
        self._platform_ship_claims: dict[int, set[int]] = {}

    @property
    def planned(self) -> bool:
        return self._planned

    @property
    def follow_on_step(self) -> int:
        return self._follow_on_step

    @property
    def sweep_platform_ids(self) -> frozenset[int]:
        return self._sweep_platform_ids

    def lane_plan(
        self,
        observation: BaselineObservation,
    ) -> tuple[list[PlatformState], float]:
        """Return the scanning platforms and the shared eastbound longitude."""

        low_platforms = sorted(
            (item for item in observation.active_platforms() if item.kind == "L"),
            key=lambda item: (item.position.lat, item.entity_id),
        )
        if not low_platforms:
            return [], 0.0
        east_lon = max(item.lon for item in self.search_polygon) - _EAST_EDGE_INSET_DEGREES
        lanes = select_lane_platforms(low_platforms, self.sweep_platform_target, east_lon)
        return lanes, east_lon

    def decorate(
        self,
        assignments: tuple[Assignment, ...],
        lane_platforms: list[PlatformState],
        east_lon: float,
    ) -> tuple[Assignment, ...]:
        """Point every lane assignment due east at its own launch latitude."""

        platform_by_id = {item.entity_id: item for item in lane_platforms}
        decorated = tuple(
            replace(
                assignment,
                destination=Position(
                    lon=east_lon,
                    lat=platform_by_id[assignment.platform_id].position.lat,
                    alt=platform_by_id[assignment.platform_id].position.alt,
                ),
            )
            for assignment in assignments
            if assignment.platform_id in platform_by_id
        )
        self._sweep_platform_ids = frozenset(item.platform_id for item in decorated)
        return decorated

    def mark_planned(self, step: int, assignments: tuple[Assignment, ...]) -> None:
        self._planned = True
        # The 10-19 engine deploys every red platform at step 0, so the lead is a
        # deployment clock, not a planner clock.  A communication split replaces
        # the planner mid-flight: anchoring the follow-on to the replan step would
        # restart the wall clock and the released-wave strike could never happen.
        # The absolute anchor keeps the intended "sweep for N steps, then strike".
        if self.follow_on_anchor == "plan":
            self._follow_on_step = step + self.sweep_lead_steps
        else:
            self._follow_on_step = self.sweep_lead_steps if assignments else step
        if assignments:
            logger.info(
                "R7 grid sweep: %d L platforms on eastbound lanes (spacing target %.1f km); "
                "follow-on strike starts at step %d",
                len(assignments),
                _LANE_SPACING_KM,
                self._follow_on_step,
            )
        else:
            logger.warning("R7 grid sweep found no usable L lane; using ordinary policy immediately")

    def holding(self, observation: BaselineObservation) -> bool:
        return observation.step < self._follow_on_step

    def claim(self, platform: PlatformState, observed_targets: tuple[TargetPrior, ...]) -> TargetPrior | None:
        """Let a scanning platform attack a ship it has legally detected."""

        if platform.entity_id not in self._sweep_platform_ids:
            return None
        if self._retarget_counts.get(platform.entity_id, 0) >= self.max_retargets_per_platform:
            return None
        claimed_by_platform = self._platform_ship_claims.setdefault(platform.entity_id, set())
        candidates = [
            item
            for item in observed_targets
            if item.entity_type == 9500
            and item.entity_id not in claimed_by_platform
            and self._ship_claims.get(item.entity_id, 0) < self.max_claims_per_ship
            and distance_km(platform.position, item.position) <= _DIRECT_RETARGET_RANGE_KM
        ]
        if not candidates:
            return None
        target = min(
            candidates,
            key=lambda item: (
                self._ship_claims.get(item.entity_id, 0),
                distance_km(platform.position, item.position),
                item.entity_id,
            ),
        )
        self._retarget_counts[platform.entity_id] = (
            self._retarget_counts.get(platform.entity_id, 0) + 1
        )
        self._ship_claims[target.entity_id] = self._ship_claims.get(target.entity_id, 0) + 1
        claimed_by_platform.add(target.entity_id)
        logger.info(
            "R7 grid sweep platform %d detected and retargeted ship %d",
            platform.entity_id,
            target.entity_id,
        )
        return target

    def preferred_ship(
        self,
        platform: PlatformState,
        targets: tuple[TargetPrior, ...],
        counts: dict[int, int],
    ) -> TargetPrior | None:
        """A discovered ship outranks public targets for low-performance launchers."""

        if platform.kind != "L":
            return None
        discovered = [
            item
            for item in targets
            if item.entity_type == 9500 and counts.get(item.entity_id, 0) < self.max_claims_per_ship
        ]
        if not discovered:
            return None
        return min(discovered, key=lambda item: (distance_km(platform.position, item.position), item.entity_id))


class R7GridSweepPolicy(R7StrikePackagePolicy):
    """Run a dense eastbound L sweep before resuming ordinary R7 waves."""

    def __init__(
        self,
        rules: BaselineRules,
        search_polygon: tuple[Position, ...],
        sweep_platform_target: int | None = None,
        sweep_lead_steps: int | None = None,
        max_retargets_per_platform: int = 2,
        max_claims_per_ship: int | None = None,
    ) -> None:
        super().__init__(rules)
        self.layer = GridSweepLayer(
            search_polygon,
            sweep_platform_target=_env_int("R7_GRID_SWEEP_PLATFORMS", 36, sweep_platform_target),
            sweep_lead_steps=_env_int("R7_GRID_SWEEP_LEAD", 240, sweep_lead_steps),
            max_retargets_per_platform=max_retargets_per_platform,
            max_claims_per_ship=_env_int("R7_GRID_SWEEP_SHIP_CLAIMS", 3, max_claims_per_ship),
        )

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        if not self.layer.planned:
            lane_platforms, east_lon = self.layer.lane_plan(observation)
            base = tuple(
                Assignment(item.entity_id, -1, observation.step, 0.0)
                for item in lane_platforms
            )
            assignments = self.layer.decorate(base, lane_platforms, east_lon)
            self.layer.mark_planned(observation.step, assignments)
            return assignments
        if self.layer.holding(observation):
            return ()
        return super().decide(observation)

    def claim_retarget(self, platform, observed_targets):
        return self.layer.claim(platform, observed_targets)

    def _best_target(self, platform, targets, counts, damage=None):
        ship = self.layer.preferred_ship(platform, targets, counts)
        if ship is not None:
            return ship
        return super()._best_target(platform, targets, counts, damage)


def _env_int(name: str, fallback: int, explicit: int | None) -> int:
    """Environment overrides let an experiment widen the wall without code edits."""

    if explicit is not None:
        return explicit
    return int(os.environ.get(name, str(fallback)))


def select_lane_platforms(
    platforms: list[PlatformState],
    target: int,
    east_lon: float,
) -> list[PlatformState]:
    """Pick an evenly spaced latitude subset of the L launchers.

    Every selected lane is a straight eastbound leg.  Subsampling by rank keeps
    the lanes as close to uniform as the launch band allows, which is what makes
    the sweep dense without any hidden-target prior.
    """

    usable = [
        item
        for item in platforms
        if distance_km(item.position, Position(lon=east_lon, lat=item.position.lat, alt=item.position.alt))
        <= _MAX_SWEEP_RANGE_KM
    ]
    if not usable:
        return []
    count = min(target, len(usable))
    if count >= len(usable):
        return usable
    step = len(usable) / float(count)
    return [usable[min(len(usable) - 1, int(ceil(index * step)))] for index in range(count)]
