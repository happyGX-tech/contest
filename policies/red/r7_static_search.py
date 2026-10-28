"""Static-unknown-target search layered on top of the R7 strike policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import combinations, permutations
from math import ceil, cos, hypot, radians, sqrt

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
from .coverage_search import radial_fan_endpoints


logger = logging.getLogger(__name__)

_CANDIDATE_GROUP_CENTERS = 73
# Geometry follows the current 1:10 scenario coordinates; step timing does not.
_MINIMUM_SEARCH_ROUTE_KM = 12.0
_SEARCH_ROUTE_LENGTH_STEP_KM = 2.5
_SAFE_INITIAL_LAUNCH_DELAY_STEPS = 1
_DIRECT_RETARGET_RANGE_KM = 10.0
_SEARCH_DANGER_RADIUS_KM = 10.0
_KM_PER_DEGREE_LATITUDE = 111.32


@dataclass(frozen=True)
class SearchGroup:
    """Three low-performance platforms that remain mutually connected."""

    group_id: int
    platform_ids: tuple[int, int, int]
    destinations: tuple[Position, Position, Position]


class R7StaticSearchPolicy(R7StrikePackagePolicy):
    """Run safe three-platform searches before resuming ordinary R7 waves.

    Search geometry uses only the public map polygon and public 9600 sites.
    Hidden 9500 positions never enter route construction.  A search platform
    may retarget a 9500 only after that target appears in its legal observation.
    """

    def __init__(self, rules: BaselineRules, search_polygon: tuple[Position, ...]) -> None:
        super().__init__(rules)
        if rules.static_search_group_size != 3:
            raise ValueError("R7 static search currently requires three-platform groups")
        if len(search_polygon) < 3:
            raise ValueError("R7 static search requires a public map polygon")
        self.search_polygon = search_polygon
        self.search_groups: tuple[SearchGroup, ...] = ()
        self._search_planned = False
        self._follow_on_step = 0
        self._claimed_ship_ids: set[int] = set()
        self._retargeted_platform_ids: set[int] = set()

    @property
    def search_platform_ids(self) -> frozenset[int]:
        return frozenset(
            platform_id
            for group in self.search_groups
            for platform_id in group.platform_ids
        )

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        if not self._search_planned:
            self._search_planned = True
            initial_r7 = super().decide(observation)
            search_assignments = self._build_initial_search(observation)
            self._follow_on_step = (
                observation.step
                + _SAFE_INITIAL_LAUNCH_DELAY_STEPS
                + self.rules.static_search_lead_steps
                if search_assignments
                else observation.step
            )
            if search_assignments:
                logger.info(
                    "R7 static search launched %d groups (%d platforms); follow-on strike starts at step %d",
                    len(self.search_groups),
                    len(search_assignments),
                    self._follow_on_step,
                )
            else:
                logger.warning(
                    "R7 static search found no safe connected group; using ordinary R7 immediately"
                )
            return search_assignments or initial_r7

        if observation.step < self._follow_on_step:
            return ()

        # Preserve R7's low-performance probe before releasing H/M packages.
        if observation.step < self._follow_on_step + self.rules.probe_window:
            active_low = sorted(
                (item for item in observation.active_platforms() if item.kind == "L"),
                key=lambda item: item.entity_id,
            )
            return self._priority_assignments(
                observation,
                active_low,
                self._release_budget(observation, "probe"),
            )

        return super().decide(observation)

    def claim_retarget(
        self,
        platform: PlatformState,
        observed_targets: tuple[TargetPrior, ...],
    ) -> TargetPrior | None:
        """Let one searcher attack a ship only after legally observing it."""

        if (
            platform.entity_id not in self.search_platform_ids
            or platform.entity_id in self._retargeted_platform_ids
        ):
            return None
        target = next(
            (
                item
                for item in observed_targets
                if item.entity_type == 9500 and item.entity_id not in self._claimed_ship_ids
                and distance_km(platform.position, item.position) <= _DIRECT_RETARGET_RANGE_KM
            ),
            None,
        )
        if target is None:
            return None
        self._retargeted_platform_ids.add(platform.entity_id)
        self._claimed_ship_ids.add(target.entity_id)
        logger.info(
            "R7 static search platform %d legally detected and retargeted ship %d",
            platform.entity_id,
            target.entity_id,
        )
        return target

    def _best_target(
        self,
        platform: PlatformState,
        targets: tuple[TargetPrior, ...],
        counts: dict[int, int],
        damage: dict[int, float] | None = None,
    ) -> TargetPrior | None:
        if platform.kind == "L":
            detected_ships = [
                item
                for item in targets
                if item.entity_type == 9500 and counts.get(item.entity_id, 0) < 1
            ]
            if detected_ships:
                return min(
                    detected_ships,
                    key=lambda item: (distance_km(platform.position, item.position), item.entity_id),
                )
        return super()._best_target(platform, targets, counts, damage)

    def _build_initial_search(
        self,
        observation: BaselineObservation,
    ) -> tuple[Assignment, ...]:
        low_platforms = sorted(
            (item for item in observation.active_platforms() if item.kind == "L"),
            key=lambda item: item.entity_id,
        )
        if len(low_platforms) < self.rules.static_search_group_size:
            return ()

        low_count = len(low_platforms)
        desired = max(
            self.rules.static_search_min_platforms,
            ceil(
                low_count
                * self.rules.static_search_low_fraction
                / self.rules.static_search_group_size
            )
            * self.rules.static_search_group_size,
        )
        probe_budget = self._release_budget(observation, "probe")
        available = min(desired, probe_budget, len(low_platforms))
        search_count = available - available % self.rules.static_search_group_size
        if search_count == 0:
            return ()

        danger_sites = tuple(
            target.position
            for target in observation.targets
            if target.entity_type == 9600
        )
        group_count = search_count // self.rules.static_search_group_size
        eligible_platforms = [
            platform
            for platform in low_platforms
            if all(
                distance_km(platform.position, site)
                >= _SEARCH_DANGER_RADIUS_KM
                for site in danger_sites
            )
        ]
        platform_groups = _candidate_platform_groups(
            eligible_platforms,
            self.rules.static_search_group_spacing_km,
        )
        if not platform_groups:
            logger.info(
                "R7 static search has no communication-safe triple outside the %.0f km danger zones",
                _SEARCH_DANGER_RADIUS_KM,
            )
            return ()

        routed_platforms: list[PlatformState] = []
        routed_destinations: list[Position] = []
        groups: list[SearchGroup] = []
        selected_platform_ids: set[int] = set()
        for platforms in platform_groups:
            platform_ids = {item.entity_id for item in platforms}
            if platform_ids & selected_platform_ids:
                continue
            destinations = self._safe_destination_group(
                platforms,
                danger_sites,
                len(groups),
                group_count,
            )
            if destinations is None:
                continue
            group_id = len(groups) + 1
            group_platform_ids = tuple(item.entity_id for item in platforms)
            groups.append(SearchGroup(group_id, group_platform_ids, destinations))
            logger.info(
                "R7 static search group %d platforms=%s starts=%s destinations=%s",
                group_id,
                group_platform_ids,
                tuple(
                    (round(item.position.lon, 4), round(item.position.lat, 4))
                    for item in platforms
                ),
                tuple(
                    (round(item.lon, 4), round(item.lat, 4))
                    for item in destinations
                ),
            )
            routed_platforms.extend(platforms)
            routed_destinations.extend(destinations)
            selected_platform_ids.update(platform_ids)
            if len(groups) == group_count:
                break
        if not groups:
            logger.info(
                "R7 static search rejected all %d connected triples because their routes entered a danger zone",
                len(platform_groups),
            )
            return ()

        original_assignments = self._priority_assignments(
            observation,
            routed_platforms,
            len(routed_platforms),
        )
        assignments = tuple(
            replace(
                assignment,
                # Core dispatches step-0 actions before simulator clocks are
                # initialized.  Launching one step later avoids an immediate
                # false 1800-second timeout without changing core physics.
                launch_step=observation.step + _SAFE_INITIAL_LAUNCH_DELAY_STEPS,
                destination=destination,
                search_group_id=1 + index // self.rules.static_search_group_size,
            )
            for index, (assignment, destination) in enumerate(
                zip(original_assignments, routed_destinations)
            )
        )
        self.search_groups = tuple(groups)
        return assignments

    def _safe_destination_group(
        self,
        platforms: tuple[PlatformState, PlatformState, PlatformState],
        danger_sites: tuple[Position, ...],
        group_index: int,
        group_count: int,
    ) -> tuple[Position, Position, Position] | None:
        origin = Position(
            lon=sum(item.position.lon for item in platforms) / 3.0,
            lat=sum(item.position.lat for item in platforms) / 3.0,
            alt=sum(item.position.alt for item in platforms) / 3.0,
        )
        centers = radial_fan_endpoints(
            origin,
            self.search_polygon,
            count=_CANDIDATE_GROUP_CENTERS,
            max_range_km=self.rules.static_search_max_range_km,
        )
        candidates: list[tuple[Position, Position, Position]] = []
        for boundary_center in centers:
            maximum_distance = distance_km(origin, boundary_center)
            for route_distance in _descending_route_distances(maximum_distance):
                center = _interpolate_position(
                    origin,
                    boundary_center,
                    route_distance / maximum_distance,
                )
                formation = _equilateral_destinations(
                    origin,
                    center,
                    self.rules.static_search_group_spacing_km,
                )
                if not all(_inside_convex_polygon(item, self.search_polygon) for item in formation):
                    continue
                destinations = _safe_destination_order(
                    platforms,
                    formation,
                    danger_sites,
                    _SEARCH_DANGER_RADIUS_KM,
                )
                if destinations is None:
                    continue
                if any(
                    destination.lon <= platform.position.lon
                    for platform, destination in zip(platforms, destinations)
                ):
                    continue
                # Keep one longest safe eastbound route per bearing.  Shortening
                # a boundary route lets the scaled formation remain inside the
                # public search area near map edges.
                candidates.append(destinations)
                break

        if not candidates:
            return None
        preferred_index = (
            len(candidates) // 2
            if group_count == 1
            else round(group_index * (len(candidates) - 1) / (group_count - 1))
        )
        candidate_indexes = sorted(
            range(len(candidates)),
            key=lambda index: (abs(index - preferred_index), index),
        )
        return candidates[candidate_indexes[0]]


def _descending_route_distances(maximum_distance_km: float) -> tuple[float, ...]:
    if maximum_distance_km < _MINIMUM_SEARCH_ROUTE_KM:
        return ()
    count = int(
        (maximum_distance_km - _MINIMUM_SEARCH_ROUTE_KM)
        // _SEARCH_ROUTE_LENGTH_STEP_KM
    )
    distances = tuple(
        maximum_distance_km - index * _SEARCH_ROUTE_LENGTH_STEP_KM
        for index in range(count + 1)
    )
    if distances[-1] > _MINIMUM_SEARCH_ROUTE_KM + 1e-6:
        return distances + (_MINIMUM_SEARCH_ROUTE_KM,)
    return distances


def _interpolate_position(start: Position, end: Position, fraction: float) -> Position:
    return Position(
        lon=start.lon + fraction * (end.lon - start.lon),
        lat=start.lat + fraction * (end.lat - start.lat),
        alt=start.alt + fraction * (end.alt - start.alt),
    )


def _candidate_platform_groups(
    platforms: list[PlatformState],
    preferred_spacing_km: float,
) -> tuple[tuple[PlatformState, PlatformState, PlatformState], ...]:
    """Return all useful connected triples, best spacing first.

    The caller owns route-safety and disjoint-group selection.  Keeping those
    decisions together prevents a geometrically ideal but unsafe first choice
    from forcing the whole search layer to fall back to ordinary R7.
    """

    candidates: dict[
        tuple[int, int, int],
        tuple[
            tuple[float, float, tuple[int, int, int]],
            tuple[PlatformState, PlatformState, PlatformState],
        ],
    ] = {}
    communication_limit = 2.0 * preferred_spacing_km
    for anchor in platforms:
        neighbors = sorted(
            (item for item in platforms if item.entity_id != anchor.entity_id),
            key=lambda item: (
                abs(distance_km(anchor.position, item.position) - preferred_spacing_km),
                item.entity_id,
            ),
        )[:16]
        for left, right in combinations(neighbors, 2):
            group = tuple(sorted((anchor, left, right), key=lambda item: item.entity_id))
            group_ids = tuple(item.entity_id for item in group)
            distances = tuple(
                distance_km(first.position, second.position)
                for first, second in combinations(group, 2)
            )
            # Launchers may be co-located.  The preferred separation requirement
            # applies to the in-flight destination formation; initially we
            # only need every member inside twice the preferred separation.
            if max(distances) > communication_limit:
                continue
            score = (
                max(abs(value - preferred_spacing_km) for value in distances),
                sum(abs(value - preferred_spacing_km) for value in distances),
                group_ids,
            )
            existing = candidates.get(group_ids)
            if existing is None or score < existing[0]:
                candidates[group_ids] = (score, group)
    return tuple(
        group
        for _, group in sorted(candidates.values(), key=lambda item: item[0])
    )


def _safe_destination_order(
    platforms: tuple[PlatformState, PlatformState, PlatformState],
    destinations: tuple[Position, Position, Position],
    danger_sites: tuple[Position, ...],
    danger_radius_km: float,
) -> tuple[Position, Position, Position] | None:
    safe_orders: list[tuple[float, tuple[Position, Position, Position]]] = []
    for destination_order in permutations(destinations):
        if any(
            _point_to_segment_distance_km(site, platform.position, destination)
            < danger_radius_km
            for site in danger_sites
            for platform, destination in zip(platforms, destination_order)
        ):
            continue
        safe_orders.append(
            (
                sum(
                    distance_km(platform.position, destination)
                    for platform, destination in zip(platforms, destination_order)
                ),
                destination_order,
            )
        )
    if not safe_orders:
        return None
    return min(safe_orders, key=lambda item: item[0])[1]


def _equilateral_destinations(
    origin: Position,
    center: Position,
    spacing_km: float,
) -> tuple[Position, Position, Position]:
    center_xy, projection = _relative_xy(center, origin)
    route_length = hypot(*center_xy)
    if route_length <= 0:
        raise ValueError("search destination must differ from its launch position")
    forward = center_xy[0] / route_length, center_xy[1] / route_length
    lateral = -forward[1], forward[0]
    half_spacing = spacing_km / 2.0
    rear_offset = sqrt(3.0) * half_spacing
    rear = (
        center_xy[0] - rear_offset * forward[0],
        center_xy[1] - rear_offset * forward[1],
    )
    left = (
        center_xy[0] + half_spacing * lateral[0],
        center_xy[1] + half_spacing * lateral[1],
    )
    right = (
        center_xy[0] - half_spacing * lateral[0],
        center_xy[1] - half_spacing * lateral[1],
    )
    return tuple(_position_from_relative_xy(point, origin, projection) for point in (rear, left, right))


def _point_to_segment_distance_km(point: Position, start: Position, end: Position) -> float:
    point_xy, projection = _relative_xy(point, start)
    end_xy, _ = _relative_xy(end, start, projection)
    length_squared = end_xy[0] ** 2 + end_xy[1] ** 2
    if length_squared <= 0:
        return hypot(*point_xy)
    fraction = max(
        0.0,
        min(
            1.0,
            (point_xy[0] * end_xy[0] + point_xy[1] * end_xy[1]) / length_squared,
        ),
    )
    return hypot(
        point_xy[0] - fraction * end_xy[0],
        point_xy[1] - fraction * end_xy[1],
    )


def _inside_convex_polygon(point: Position, polygon: tuple[Position, ...]) -> bool:
    point_xy, projection = _relative_xy(point, polygon[0])
    polygon_xy = tuple(_relative_xy(item, polygon[0], projection)[0] for item in polygon)
    signs: set[int] = set()
    for start, end in zip(polygon_xy, polygon_xy[1:] + polygon_xy[:1]):
        cross = (
            (end[0] - start[0]) * (point_xy[1] - start[1])
            - (end[1] - start[1]) * (point_xy[0] - start[0])
        )
        if abs(cross) > 1e-7:
            signs.add(1 if cross > 0 else -1)
    return len(signs) <= 1


def _relative_xy(
    point: Position,
    origin: Position,
    projection: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if projection is None:
        reference_latitude = (point.lat + origin.lat) / 2.0
        projection = (
            _KM_PER_DEGREE_LATITUDE * cos(radians(reference_latitude)),
            _KM_PER_DEGREE_LATITUDE,
        )
    return (
        (point.lon - origin.lon) * projection[0],
        (point.lat - origin.lat) * projection[1],
    ), projection


def _position_from_relative_xy(
    point: tuple[float, float],
    origin: Position,
    projection: tuple[float, float],
) -> Position:
    return Position(
        lon=origin.lon + point[0] / projection[0],
        lat=origin.lat + point[1] / projection[1],
        alt=origin.alt,
    )
