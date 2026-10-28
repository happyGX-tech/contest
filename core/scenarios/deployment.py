"""Validate scenario-owned deployment constraints before constructing agents."""

import math

RED_PLATFORM_TYPES = (21000, 21001, 21002)


def resolve_deployment_constraints(profile, deployment_area, deployment_plan=None):
    """Return matching scene/metadata bounds and the optional launch altitude."""
    red_area = profile.imagineProfile.redArea
    coordinates = red_area.coordinates
    coordinates_hm = red_area.coordinatesHM
    launch_altitude = None
    if deployment_area is None:
        return coordinates, coordinates_hm, launch_altitude

    mode = deployment_area.get("mode", "free")
    if mode != "free":
        raise ValueError(f"Unsupported deployment mode '{mode}'")
    if deployment_area.get("grid_required", False) and deployment_plan is None:
        raise ValueError("This scenario requires --deployment-plan")
    if deployment_area.get("all_units_depart_at_start") is False:
        raise ValueError("This engine only supports all_units_depart_at_start=true")

    declared_coordinates = deployment_area.get("coordinates")
    if declared_coordinates is not None:
        if declared_coordinates != coordinates:
            raise ValueError("deployment_area coordinates disagree with scenario redArea")
        coordinates = declared_coordinates
    declared_coordinates_hm = deployment_area.get("coordinatesHM")
    if declared_coordinates_hm is not None:
        if declared_coordinates_hm != coordinates_hm:
            raise ValueError("deployment_area coordinatesHM disagree with scenario redArea")
        coordinates_hm = declared_coordinates_hm

    if "launch_altitude_external_m" in deployment_area:
        launch_altitude = float(deployment_area["launch_altitude_external_m"])
        if not math.isfinite(launch_altitude):
            raise ValueError("launch_altitude_external_m must be finite")
        for item in profile.imagineProfile.entityList:
            if item.entity.entityType not in RED_PLATFORM_TYPES:
                continue
            altitude = float(item.entity.lla.z)
            if not math.isfinite(altitude) or abs(altitude - launch_altitude) > 1e-6:
                raise ValueError(
                    f"Red entity {item.entity.id} altitude disagrees with "
                    "deployment_area launch_altitude_external_m"
                )
    return coordinates, coordinates_hm, launch_altitude
