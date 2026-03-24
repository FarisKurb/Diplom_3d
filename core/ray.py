"""Ray primitive for raycasting and intersection tests."""

from __future__ import annotations

from core.vector3 import Vector3


class Ray:
    """A ray defined by an origin point and a direction vector.

    Attributes:
        origin:    Starting point of the ray.
        direction: *Normalized* direction of the ray.
    """

    __slots__ = ("origin", "direction")

    def __init__(self, origin: Vector3, direction: Vector3) -> None:
        self.origin = origin
        self.direction = direction.normalized()

    def point_at(self, t: float) -> Vector3:
        """Return the point along the ray at parameter *t*.

        ``ray.origin + t * ray.direction``
        """
        return self.origin + self.direction * t

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin}, direction={self.direction})"
