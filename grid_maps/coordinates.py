"""Coordinate mapping; the map board is not a metric coordinate system."""
import math


def lla_to_ecef(lon, lat, alt):
    lon, lat = math.radians(lon), math.radians(lat)
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return ((n + alt) * math.cos(lat) * math.cos(lon),
            (n + alt) * math.cos(lat) * math.sin(lon),
            (n * (1 - e2) + alt) * math.sin(lat))


def board_to_lla(x, y, width, height, area, altitude=0):
    return {'x': area['lonMin'] + x / width * (area['lonMax'] - area['lonMin']),
            'y': area['latMax'] - y / height * (area['latMax'] - area['latMin']),
            'z': altitude}


def set_position(entity, lla):
    entity['lla'] = dict(lla)
    entity['posEcf'] = dict(zip(('x', 'y', 'z'), lla_to_ecef(lla['x'], lla['y'], lla['z'])))
    entity['velEcf'] = {'x': 0., 'y': 0., 'z': 0.}
    entity['velNue'] = {'x': 0., 'y': 0., 'z': 0.}


def blue_detects_position(sensor, target, stage=3, visible=True, health=1):
    """Mirror ad45a7b: ECEF XY distance <= 20000, with target eligibility."""
    a = lla_to_ecef(sensor['x'], sensor['y'], sensor['z'])
    b = lla_to_ecef(target['x'], target['y'], target['z'])
    return visible and health > 0 and stage >= 3 and (a[0]-b[0])**2 + (a[1]-b[1])**2 <= 20000**2
