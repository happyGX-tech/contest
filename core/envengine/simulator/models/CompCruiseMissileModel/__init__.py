import os


if os.name == "nt":
    from .CompCruiseMissileHPy import Missile
else:
    from .CompCruiseMissilePy import Missile
