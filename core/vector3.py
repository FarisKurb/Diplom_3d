"""Three-dimensional vector class for geometric computations."""

from __future__ import annotations

import math
from typing import Tuple


class Vector3:
    """Immutable 3-component vector with standard linear-algebra operations.

    Attributes:
        x: X component.
        y: Y component.
        z: Z component.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("Vector3 is immutable")

    # ---- factory helpers ----

    @staticmethod
    def zero() -> Vector3:
        """Return the zero vector."""
        return Vector3(0.0, 0.0, 0.0)

    @staticmethod
    def from_tuple(t: Tuple[float, float, float]) -> Vector3:
        """Create a Vector3 from a 3-tuple."""
        return Vector3(t[0], t[1], t[2])

    # ---- arithmetic operators ----

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        inv = 1.0 / scalar
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    # ---- comparison / hashing ----

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    # ---- core operations ----

    def dot(self, other: Vector3) -> float:
        """Dot product of *self* and *other*."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Cross product of *self* and *other*."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        """Euclidean length."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        """Squared Euclidean length (avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> Vector3:
        """Unit vector in the same direction.

        Raises:
            ValueError: If the vector has zero length.
        """
        lng = self.length()
        if lng < 1e-12:
            raise ValueError("Cannot normalize a zero-length vector")
        return self / lng

    def distance_to(self, other: Vector3) -> float:
        """Euclidean distance to *other*."""
        return (self - other).length()

    def approx_equal(self, other: Vector3, eps: float = 1e-9) -> bool:
        """Check approximate component-wise equality."""
        return (
            abs(self.x - other.x) < eps
            and abs(self.y - other.y) < eps
            and abs(self.z - other.z) < eps
        )

    def lerp(self, other: Vector3, t: float) -> Vector3:
        """Linear interpolation between *self* and *other*."""
        return self * (1.0 - t) + other * t

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return the vector as a plain tuple."""
        return (self.x, self.y, self.z)
