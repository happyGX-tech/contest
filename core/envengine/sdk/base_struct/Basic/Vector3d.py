# -*- coding: utf-8 -*-#
import math
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Vector3d:
    # x
    x: float = 0.0
    # y
    y: float = 0.0
    # z
    z: float = 0.0

    # ====================== 向量加减 ======================

    def __add__(self, other: "Vector3d") -> "Vector3d":
        """向量加"""
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3d") -> "Vector3d":
        """向量减"""
        return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vector3d":
        """取反"""
        return Vector3d(-self.x, -self.y, -self.z)

    # ====================== 向量缩放（乘/除标量） ======================

    def __mul__(self, scalar: float) -> "Vector3d":
        """数乘（缩放）"""
        return Vector3d(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3d":
        """数乘（缩放），支持 scalar * vector"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3d":
        """数除（缩放）"""
        return Vector3d(self.x / scalar, self.y / scalar, self.z / scalar)

    # ====================== 长度 / 归一化 ======================

    @property
    def length_square(self) -> float:
        """长度的平方"""
        return self.x * self.x + self.y * self.y + self.z * self.z

    @property
    def length(self) -> float:
        """长度（模）"""
        return round(math.sqrt(self.length_square), 6)

    def normalized(self) -> "Vector3d":
        """
        返回单位化（归一化）向量，零向量返回零向量
        """
        norm = self.length
        if norm <= 0.0:
            return Vector3d(0.0, 0.0, 0.0)
        return self / norm

    # ====================== 点乘 / 叉乘 ======================

    def dot(self, other: "Vector3d") -> float:
        """点乘"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3d") -> "Vector3d":
        """叉乘"""
        return Vector3d(self.y * other.z - self.z * other.y,
                        self.z * other.x - self.x * other.z,
                        self.x * other.y - self.y * other.x)
