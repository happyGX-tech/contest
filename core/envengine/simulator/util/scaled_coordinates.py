"""Velocity under the engine's (longitude, latitude, height) / scale mapping."""
from math import cos, radians, sin, sqrt

from envengine.sdk.base_struct.Basic import Vector3d


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _curvature_radii(sin_latitude):
    denominator = 1.0 - _WGS84_E2 * sin_latitude ** 2
    prime_vertical = _WGS84_A / sqrt(denominator)
    meridian = prime_vertical * (1.0 - _WGS84_E2) / denominator
    return prime_vertical, meridian


def _component(vector, axis: str) -> float:
    """同时兼容 Vector3d（属性取值）与 Vector3D（方法取值）."""
    value = getattr(vector, axis)
    return value() if callable(value) else value


def unscaled_ecef_velocity(engine_lla, engine_velocity, scale):
    """scaled_ecef_velocity 的逆变换：把引擎尺度下的 ECEF 速度还原到模型尺度.

    输入的经纬高与速度为引擎（缩小后）尺度，输出为模型（放大后）尺度的 ECEF 速度。
    """
    lon, lat, height = radians(_component(engine_lla, "x")), radians(_component(engine_lla, "y")), _component(
        engine_lla, "z")
    vx, vy, vz = _component(engine_velocity, "x"), _component(engine_velocity, "y"), _component(engine_velocity, "z")

    sin_lon, cos_lon = sin(lon), cos(lon)
    sin_lat, cos_lat = sin(lat), cos(lat)
    east = -sin_lon * vx + cos_lon * vy
    north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz
    up = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz
    mapped_n, mapped_m = _curvature_radii(sin_lat)
    mapped_cos_lat = cos_lat

    lon, lat, height = lon * scale, lat * scale, height * scale
    sin_lon, cos_lon = sin(lon), cos(lon)
    sin_lat, cos_lat = sin(lat), cos(lat)
    source_n, source_m = _curvature_radii(sin_lat)

    # 还原曲率与尺度带来的缩放（height 为放大后的高度，引擎高度为 height / scale）
    east *= (source_n + height) * cos_lat * scale / ((mapped_n + height / scale) * mapped_cos_lat)
    north *= (source_m + height) * scale / (mapped_m + height / scale)
    up *= scale

    return Vector3d(
        -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up,
        cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up,
        cos_lat * north + sin_lat * up,
    )


def scaled_ecef_velocity(model_lla, model_velocity, scale):
    """Apply the position mapping's derivative, including the change of frame.

    Inputs use the native model's geodetic degrees/metres and ECEF metres/second.
    The result is the time derivative of the mapped ECEF position. Rotating
    directions alone is insufficient: curvature changes metres per degree.
    """
    lon, lat, height = radians(model_lla.x()), radians(model_lla.y()), model_lla.z()
    sin_lon, cos_lon, sin_lat, cos_lat = sin(lon), cos(lon), sin(lat), cos(lat)
    vx, vy, vz = model_velocity.x(), model_velocity.y(), model_velocity.z()
    east = -sin_lon * vx + cos_lon * vy
    north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz
    up = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz
    source_n, source_m = _curvature_radii(sin_lat)

    lon, lat, mapped_height = lon / scale, lat / scale, height / scale
    sin_lon, cos_lon, mapped_sin_lat, mapped_cos_lat = sin(lon), cos(lon), sin(lat), cos(lat)
    mapped_n, mapped_m = _curvature_radii(mapped_sin_lat)
    east *= (mapped_n + mapped_height) * mapped_cos_lat / ((source_n + height) * cos_lat * scale)
    north *= (mapped_m + mapped_height) / ((source_m + height) * scale)
    up /= scale

    return Vector3d(
        -sin_lon * east - mapped_sin_lat * cos_lon * north + mapped_cos_lat * cos_lon * up,
        cos_lon * east - mapped_sin_lat * sin_lon * north + mapped_cos_lat * sin_lon * up,
        mapped_cos_lat * north + mapped_sin_lat * up,
    )
