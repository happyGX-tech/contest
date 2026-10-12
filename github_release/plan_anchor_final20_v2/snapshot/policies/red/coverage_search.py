"""Coverage planning for public static-search regions.

The planner deliberately accepts only a public convex search polygon, sensor
radius, and red platform starts.  Hidden target coordinates are not part of
the interface.  It creates a boustrophedon (lawnmower) path and divides that
path into contiguous routes while accounting for each platform's preparation
distance to either end of its assigned route.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, hypot, isfinite, pi, radians, sin
from typing import Sequence

from .contracts import Position


_KM_PER_DEGREE_LATITUDE = 111.32
_EPSILON = 1e-9
_MAX_ALLOCATION_EDGES = 4096
_XY = tuple[float, float]


@dataclass(frozen=True)
class SearchPlatformStart:
    """A platform that is available for the static-area search."""

    platform_id: int
    position: Position


@dataclass(frozen=True)
class CoverageRoute:
    """One contiguous portion of the common coverage path."""

    platform_id: int
    waypoints: tuple[Position, ...]
    preparation_distance_km: float
    search_distance_km: float

    @property
    def total_distance_km(self) -> float:
        return self.preparation_distance_km + self.search_distance_km


@dataclass(frozen=True)
class CoveragePlan:
    """A complete, target-independent search plan for a public region."""

    routes: tuple[CoverageRoute, ...]
    coverage_waypoints: tuple[Position, ...]
    search_radius_km: float
    lane_spacing_km: float
    lane_count: int
    sweep_bearing_degrees: float
    coverage_distance_km: float
    idle_platform_ids: tuple[int, ...] = ()

    @property
    def max_total_distance_km(self) -> float:
        return max((route.total_distance_km for route in self.routes), default=0.0)


@dataclass(frozen=True)
class _Projection:
    reference_lon: float
    reference_lat: float
    altitude: float

    @property
    def longitude_scale(self) -> float:
        return _KM_PER_DEGREE_LATITUDE * cos(radians(self.reference_lat))

    def to_xy(self, position: Position) -> _XY:
        return (
            (position.lon - self.reference_lon) * self.longitude_scale,
            (position.lat - self.reference_lat) * _KM_PER_DEGREE_LATITUDE,
        )

    def to_position(self, point: _XY) -> Position:
        return Position(
            lon=self.reference_lon + point[0] / self.longitude_scale,
            lat=self.reference_lat + point[1] / _KM_PER_DEGREE_LATITUDE,
            alt=self.altitude,
        )


@dataclass(frozen=True)
class _SweepPath:
    points: tuple[_XY, ...]
    lane_spacing_km: float
    lane_count: int
    bearing_degrees: float
    length_km: float


@dataclass(frozen=True)
class _Allocation:
    cuts: tuple[int, ...]
    platform_by_segment: tuple[int, ...]
    preparation_by_segment: tuple[float, ...]
    total_by_segment: tuple[float, ...]

    @property
    def objective(self) -> tuple[float, float]:
        return max(self.total_by_segment), sum(value * value for value in self.total_by_segment)


def plan_static_coverage(
    polygon: Sequence[Position],
    search_radius_km: float,
    platforms: Sequence[SearchPlatformStart],
) -> CoveragePlan:
    """Create and allocate a full-coverage path over a convex public polygon.

    ``search_radius_km`` is the effective circular search radius.  Adjacent
    lane centers are never farther apart than twice that value.  The returned
    routes partition the coverage path without gaps; a route may be reversed
    so its assigned platform approaches the nearer endpoint.

    Raises:
        ValueError: if the polygon is degenerate/non-convex, the radius is not
            positive, platform identifiers are duplicated, or an input
            coordinate is not finite.
    """

    vertices = _normalize_polygon(polygon)
    platform_starts = tuple(platforms)
    _validate_platforms(platform_starts)
    if not isfinite(search_radius_km) or search_radius_km <= 0:
        raise ValueError("search radius must be a positive finite distance")
    if not platform_starts:
        raise ValueError("at least one search platform is required")

    projection = _projection_for(vertices)
    polygon_xy = tuple(projection.to_xy(vertex) for vertex in vertices)
    _validate_convex_polygon(polygon_xy)

    sweep = _choose_sweep_path(polygon_xy, search_radius_km)
    platform_xy = tuple((item.platform_id, projection.to_xy(item.position)) for item in platform_starts)
    dense_path = _densify_path(sweep.points, search_radius_km, len(platform_xy))
    routes_xy, idle_platform_ids = _allocate_path(dense_path, platform_xy)

    routes = tuple(
        CoverageRoute(
            platform_id=platform_id,
            waypoints=tuple(projection.to_position(point) for point in route_points),
            preparation_distance_km=preparation_distance,
            search_distance_km=search_distance,
        )
        for platform_id, route_points, preparation_distance, search_distance in routes_xy
    )
    return CoveragePlan(
        routes=tuple(sorted(routes, key=lambda route: route.platform_id)),
        coverage_waypoints=tuple(projection.to_position(point) for point in sweep.points),
        search_radius_km=search_radius_km,
        lane_spacing_km=sweep.lane_spacing_km,
        lane_count=sweep.lane_count,
        sweep_bearing_degrees=sweep.bearing_degrees,
        coverage_distance_km=sweep.length_km,
        idle_platform_ids=tuple(sorted(idle_platform_ids)),
    )


def radial_fan_endpoints(
    origin: Position,
    bounds: Sequence[Position],
    count: int,
    max_range_km: float,
) -> tuple[Position, ...]:
    """Return evenly spaced radial endpoints toward a public convex boundary.

    For an origin outside the region, the rays span the smallest bearing arc
    containing the polygon.  For an origin inside it, they span 360 degrees.
    Each ray stops at the nearer of the far-side polygon boundary and
    ``max_range_km``.  When the origin is outside the polygon, this carries a
    searcher through the region instead of stopping it at the entry edge.
    Target positions and threat information are neither accepted nor
    inspected.
    """

    vertices = _normalize_polygon(bounds)
    if count <= 0:
        raise ValueError("radial endpoint count must be positive")
    if not isfinite(max_range_km) or max_range_km <= 0:
        raise ValueError("maximum radial range must be a positive finite distance")
    if not all(isfinite(value) for value in (origin.lon, origin.lat, origin.alt)):
        raise ValueError("origin coordinates must be finite")

    base_projection = _projection_for(vertices + (origin,))
    projection = _Projection(
        base_projection.reference_lon,
        base_projection.reference_lat,
        origin.alt,
    )
    polygon_xy = tuple(projection.to_xy(vertex) for vertex in vertices)
    _validate_convex_polygon(polygon_xy)
    origin_xy = projection.to_xy(origin)

    if _point_in_convex_polygon(origin_xy, polygon_xy):
        bearings = tuple(2.0 * pi * index / count for index in range(count))
    else:
        bearings = _visible_bearings(origin_xy, polygon_xy, count)

    endpoints: list[Position] = []
    for bearing in bearings:
        direction = cos(bearing), sin(bearing)
        boundary_distance = _ray_polygon_exit_distance(origin_xy, direction, polygon_xy)
        distance = min(max_range_km, boundary_distance)
        endpoints.append(projection.to_position(_add(origin_xy, _scale(direction, distance))))
    return tuple(endpoints)


def _normalize_polygon(polygon: Sequence[Position]) -> tuple[Position, ...]:
    vertices = tuple(polygon)
    if len(vertices) >= 2 and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise ValueError("a search polygon requires at least three vertices")
    for vertex in vertices:
        if not all(isfinite(value) for value in (vertex.lon, vertex.lat, vertex.alt)):
            raise ValueError("polygon coordinates must be finite")
    return vertices


def _validate_platforms(platforms: tuple[SearchPlatformStart, ...]) -> None:
    identifiers = [item.platform_id for item in platforms]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("search platform identifiers must be unique")
    for item in platforms:
        if not all(isfinite(value) for value in (item.position.lon, item.position.lat, item.position.alt)):
            raise ValueError("platform coordinates must be finite")


def _projection_for(vertices: tuple[Position, ...]) -> _Projection:
    count = len(vertices)
    projection = _Projection(
        reference_lon=sum(item.lon for item in vertices) / count,
        reference_lat=sum(item.lat for item in vertices) / count,
        altitude=sum(item.alt for item in vertices) / count,
    )
    if abs(projection.longitude_scale) <= _EPSILON:
        raise ValueError("search polygons at the geographic poles are unsupported")
    return projection


def _validate_convex_polygon(vertices: tuple[_XY, ...]) -> None:
    signed_area_twice = sum(
        _cross(vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    )
    if abs(signed_area_twice) <= _EPSILON:
        raise ValueError("search polygon must have positive area")

    turn_sign = 0
    for index in range(len(vertices)):
        first = vertices[index]
        second = vertices[(index + 1) % len(vertices)]
        third = vertices[(index + 2) % len(vertices)]
        turn = _cross(_subtract(second, first), _subtract(third, second))
        if abs(turn) <= _EPSILON:
            continue
        current_sign = 1 if turn > 0 else -1
        if turn_sign and current_sign != turn_sign:
            raise ValueError("search polygon must be convex with ordered boundary vertices")
        turn_sign = current_sign
    if turn_sign == 0:
        raise ValueError("search polygon must have positive area")


def _choose_sweep_path(polygon: tuple[_XY, ...], radius_km: float) -> _SweepPath:
    candidates: list[_SweepPath] = []
    seen_axes: set[tuple[int, int]] = set()
    for index, point in enumerate(polygon):
        edge = _subtract(polygon[(index + 1) % len(polygon)], point)
        length = _norm(edge)
        if length <= _EPSILON:
            continue
        axis = (edge[0] / length, edge[1] / length)
        if axis[0] < -_EPSILON or (abs(axis[0]) <= _EPSILON and axis[1] < 0):
            axis = (-axis[0], -axis[1])
        axis_key = (round(axis[0] * 1_000_000_000), round(axis[1] * 1_000_000_000))
        if axis_key in seen_axes:
            continue
        seen_axes.add(axis_key)
        candidates.append(_build_sweep_path(polygon, radius_km, axis))
    if not candidates:
        raise ValueError("search polygon has no usable edge")
    return min(candidates, key=lambda item: (item.length_km, item.lane_count, item.bearing_degrees))


def _build_sweep_path(polygon: tuple[_XY, ...], radius_km: float, axis: _XY) -> _SweepPath:
    normal = (-axis[1], axis[0])
    normal_values = [_dot(point, normal) for point in polygon]
    lower, upper = min(normal_values), max(normal_values)
    width = upper - lower
    lane_count = max(1, ceil(width / (2.0 * radius_km) - _EPSILON))
    spacing = width / lane_count

    path: list[_XY] = []
    for lane_index in range(lane_count):
        offset = lower + (lane_index + 0.5) * spacing
        band = _clip_to_band(polygon, normal, offset - radius_km, offset + radius_km)
        if not band:
            raise RuntimeError("coverage lane does not intersect its polygon band")
        axial_values = [_dot(point, axis) for point in band]
        start = _add(_scale(axis, min(axial_values)), _scale(normal, offset))
        end = _add(_scale(axis, max(axial_values)), _scale(normal, offset))
        path.extend((start, end) if lane_index % 2 == 0 else (end, start))

    length = _polyline_length(path)
    bearing = (degrees(atan2(axis[0], axis[1])) + 360.0) % 180.0
    return _SweepPath(tuple(path), spacing, lane_count, bearing, length)


def _clip_to_band(polygon: tuple[_XY, ...], normal: _XY, lower: float, upper: float) -> tuple[_XY, ...]:
    below_upper = _clip_half_plane(polygon, normal, upper, keep_below=True)
    return _clip_half_plane(below_upper, normal, lower, keep_below=False)


def _visible_bearings(origin: _XY, polygon: tuple[_XY, ...], count: int) -> tuple[float, ...]:
    angles = sorted((atan2(point[1] - origin[1], point[0] - origin[0]) % (2.0 * pi)) for point in polygon)
    gaps = [
        (angles[(index + 1) % len(angles)] - angles[index]) % (2.0 * pi)
        for index in range(len(angles))
    ]
    gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    start = angles[(gap_index + 1) % len(angles)]
    span = 2.0 * pi - gaps[gap_index]
    if count == 1:
        return ((start + span / 2.0) % (2.0 * pi),)
    return tuple((start + span * index / (count - 1)) % (2.0 * pi) for index in range(count))


def _point_in_convex_polygon(point: _XY, polygon: tuple[_XY, ...]) -> bool:
    signs: set[int] = set()
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        side = _cross(_subtract(end, start), _subtract(point, start))
        if abs(side) > _EPSILON:
            signs.add(1 if side > 0 else -1)
    return len(signs) <= 1


def _ray_polygon_exit_distance(origin: _XY, direction: _XY, polygon: tuple[_XY, ...]) -> float:
    intersections: list[float] = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        edge = _subtract(end, start)
        denominator = _cross(direction, edge)
        if abs(denominator) <= _EPSILON:
            continue
        offset = _subtract(start, origin)
        ray_distance = _cross(offset, edge) / denominator
        edge_fraction = _cross(offset, direction) / denominator
        if ray_distance >= -_EPSILON and -_EPSILON <= edge_fraction <= 1.0 + _EPSILON:
            intersections.append(max(0.0, ray_distance))
    if not intersections:
        raise RuntimeError("radial bearing does not intersect the convex boundary")
    return max(intersections)


def _clip_half_plane(
    polygon: Sequence[_XY],
    normal: _XY,
    threshold: float,
    *,
    keep_below: bool,
) -> tuple[_XY, ...]:
    if not polygon:
        return ()

    def inside(point: _XY) -> bool:
        signed = _dot(point, normal) - threshold
        return signed <= _EPSILON if keep_below else signed >= -_EPSILON

    output: list[_XY] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            direction = _subtract(current, previous)
            denominator = _dot(direction, normal)
            if abs(denominator) > _EPSILON:
                fraction = (threshold - _dot(previous, normal)) / denominator
                output.append(_add(previous, _scale(direction, fraction)))
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return tuple(output)


def _densify_path(points: tuple[_XY, ...], radius_km: float, platform_count: int) -> tuple[_XY, ...]:
    path_length = _polyline_length(points)
    desired_resolution = min(radius_km / 2.0, path_length / max(1, platform_count * 8))
    resolution = max(path_length / _MAX_ALLOCATION_EDGES, desired_resolution)
    dense: list[_XY] = [points[0]]
    for start, end in zip(points, points[1:]):
        distance = _distance(start, end)
        pieces = max(1, ceil(distance / resolution))
        for piece in range(1, pieces + 1):
            fraction = piece / pieces
            dense.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
    return tuple(dense)


def _allocate_path(
    points: tuple[_XY, ...],
    platforms: tuple[tuple[int, _XY], ...],
) -> tuple[tuple[tuple[int, tuple[_XY, ...], float, float], ...], tuple[int, ...]]:
    edge_count = len(points) - 1
    segment_count = min(len(platforms), edge_count)
    cuts = tuple(index * edge_count // segment_count for index in range(segment_count)) + (edge_count,)
    prefix_lengths = _prefix_lengths(points)
    allocation = _evaluate_allocation(cuts, points, prefix_lengths, platforms)

    for _ in range(min(64, edge_count)):
        best = allocation
        maximum = allocation.objective[0]
        overloaded_segments = [
            index
            for index, total in enumerate(allocation.total_by_segment)
            if abs(total - maximum) <= _EPSILON
        ]
        for segment_index in overloaded_segments:
            moves: list[tuple[int, int]] = []
            if segment_index > 0:
                boundary = segment_index
                transfer = max(
                    0.0,
                    (maximum - allocation.total_by_segment[segment_index - 1]) / 2.0,
                )
                target_length = prefix_lengths[allocation.cuts[boundary]] + transfer
                moves.append((boundary, bisect_left(prefix_lengths, target_length)))
            if segment_index + 1 < segment_count:
                boundary = segment_index + 1
                transfer = max(
                    0.0,
                    (maximum - allocation.total_by_segment[segment_index + 1]) / 2.0,
                )
                target_length = prefix_lengths[allocation.cuts[boundary]] - transfer
                moves.append((boundary, bisect_left(prefix_lengths, target_length)))

            for boundary, proposed_index in moves:
                for candidate_index in (proposed_index - 1, proposed_index, proposed_index + 1):
                    moved = list(allocation.cuts)
                    moved[boundary] = candidate_index
                    if moved[boundary] <= moved[boundary - 1] or moved[boundary] >= moved[boundary + 1]:
                        continue
                    candidate = _evaluate_allocation(tuple(moved), points, prefix_lengths, platforms)
                    if _objective_is_better(candidate.objective, best.objective):
                        best = candidate
        if not _objective_is_better(best.objective, allocation.objective):
            break
        allocation = best

    assigned_platform_indexes = set(allocation.platform_by_segment)
    idle = tuple(
        platform_id
        for index, (platform_id, _) in enumerate(platforms)
        if index not in assigned_platform_indexes
    )
    routes: list[tuple[int, tuple[_XY, ...], float, float]] = []
    for segment_index, platform_index in enumerate(allocation.platform_by_segment):
        start_index = allocation.cuts[segment_index]
        end_index = allocation.cuts[segment_index + 1]
        route_points = points[start_index : end_index + 1]
        platform_id, platform_start = platforms[platform_index]
        distance_to_start = _distance(platform_start, route_points[0])
        distance_to_end = _distance(platform_start, route_points[-1])
        if distance_to_end + _EPSILON < distance_to_start:
            route_points = tuple(reversed(route_points))
        search_distance = prefix_lengths[end_index] - prefix_lengths[start_index]
        routes.append(
            (
                platform_id,
                route_points,
                allocation.preparation_by_segment[segment_index],
                search_distance,
            )
        )
    return tuple(routes), idle


def _evaluate_allocation(
    cuts: tuple[int, ...],
    points: tuple[_XY, ...],
    prefix_lengths: tuple[float, ...],
    platforms: tuple[tuple[int, _XY], ...],
) -> _Allocation:
    segment_count = len(cuts) - 1
    costs: list[list[float]] = []
    preparations: list[list[float]] = []
    for _, platform_start in platforms:
        platform_costs: list[float] = []
        platform_preparations: list[float] = []
        for segment_index in range(segment_count):
            start_index, end_index = cuts[segment_index], cuts[segment_index + 1]
            preparation = min(
                _distance(platform_start, points[start_index]),
                _distance(platform_start, points[end_index]),
            )
            search_distance = prefix_lengths[end_index] - prefix_lengths[start_index]
            platform_preparations.append(preparation)
            platform_costs.append(preparation + search_distance)
        preparations.append(platform_preparations)
        costs.append(platform_costs)

    platform_by_segment = list(_minimum_cost_platforms(costs))
    preparation_by_segment = [0.0] * segment_count
    total_by_segment = [0.0] * segment_count
    for segment_index, platform_index in enumerate(platform_by_segment):
        preparation_by_segment[segment_index] = preparations[platform_index][segment_index]
        total_by_segment[segment_index] = costs[platform_index][segment_index]
    return _Allocation(
        cuts=cuts,
        platform_by_segment=tuple(platform_by_segment),
        preparation_by_segment=tuple(preparation_by_segment),
        total_by_segment=tuple(total_by_segment),
    )


def _minimum_cost_platforms(costs: list[list[float]]) -> tuple[int, ...]:
    """Assign each route segment to one platform with minimum total cost."""

    platform_count = len(costs)
    segment_count = len(costs[0])
    # The shortest-augmenting-path form below assigns every row, so segments
    # are rows and the (at least equally numerous) platforms are columns.
    matrix = [
        [costs[platform_index][segment_index] for platform_index in range(platform_count)]
        for segment_index in range(segment_count)
    ]
    row_potential = [0.0] * (segment_count + 1)
    column_potential = [0.0] * (platform_count + 1)
    matched_row = [0] * (platform_count + 1)
    previous_column = [0] * (platform_count + 1)

    for row in range(1, segment_count + 1):
        matched_row[0] = row
        minimum_slack = [float("inf")] * (platform_count + 1)
        used = [False] * (platform_count + 1)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, platform_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    matrix[active_row - 1][candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum_slack[candidate_column]:
                    minimum_slack[candidate_column] = reduced_cost
                    previous_column[candidate_column] = column
                if minimum_slack[candidate_column] < delta:
                    delta = minimum_slack[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(platform_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum_slack[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break

        while True:
            prior_column = previous_column[column]
            matched_row[column] = matched_row[prior_column]
            column = prior_column
            if column == 0:
                break

    platform_by_segment = [-1] * segment_count
    for column in range(1, platform_count + 1):
        if matched_row[column]:
            platform_by_segment[matched_row[column] - 1] = column - 1
    return tuple(platform_by_segment)


def _objective_is_better(candidate: tuple[float, float], current: tuple[float, float]) -> bool:
    if candidate[0] < current[0] - _EPSILON:
        return True
    return abs(candidate[0] - current[0]) <= _EPSILON and candidate[1] < current[1] - _EPSILON


def _prefix_lengths(points: tuple[_XY, ...]) -> tuple[float, ...]:
    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + _distance(start, end))
    return tuple(lengths)


def _polyline_length(points: Sequence[_XY]) -> float:
    return sum(_distance(start, end) for start, end in zip(points, points[1:]))


def _add(left: _XY, right: _XY) -> _XY:
    return left[0] + right[0], left[1] + right[1]


def _subtract(left: _XY, right: _XY) -> _XY:
    return left[0] - right[0], left[1] - right[1]


def _scale(point: _XY, scalar: float) -> _XY:
    return point[0] * scalar, point[1] * scalar


def _dot(left: _XY, right: _XY) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _cross(left: _XY, right: _XY) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _norm(point: _XY) -> float:
    return hypot(point[0], point[1])


def _distance(left: _XY, right: _XY) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])
