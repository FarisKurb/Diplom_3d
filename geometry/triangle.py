"""Triangle primitive for 3-D geometry."""

from __future__ import annotations

from typing import Tuple

from core.vector3 import Vector3
from core.math_utils import barycentric_coordinates, EPSILON


class Triangle:
    """A triangle defined by three vertices in 3-D space.

    Attributes:
        v0: First vertex.
        v1: Second vertex.
        v2: Third vertex.
    """

    __slots__ = ("v0", "v1", "v2")

    def __init__(self, v0: Vector3, v1: Vector3, v2: Vector3) -> None:
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    # ── derived quantities ──────────────────────────────────

    def edge0(self) -> Vector3:
        """Edge vector v0 → v1."""
        return self.v1 - self.v0

    def edge1(self) -> Vector3:
        """Edge vector v0 → v2."""
        return self.v2 - self.v0

    def edge2(self) -> Vector3:
        """Edge vector v1 → v2."""
        return self.v2 - self.v1

    def normal(self) -> Vector3:
        """Unit outward normal (right-hand rule on v0→v1, v0→v2).

        Raises:
            ValueError: If the triangle is degenerate.
        """
        return self.edge0().cross(self.edge1()).normalized()

    def area(self) -> float:
        """Area of the triangle."""
        return 0.5 * self.edge0().cross(self.edge1()).length()

    def centroid(self) -> Vector3:
        """Centroid (average of the three vertices)."""
        return (self.v0 + self.v1 + self.v2) / 3.0

    def edge_midpoints(self) -> Tuple[Vector3, Vector3, Vector3]:
        """Midpoints of edges (v0v1, v1v2, v0v2)."""
        return (
            self.v0.lerp(self.v1, 0.5),
            self.v1.lerp(self.v2, 0.5),
            self.v0.lerp(self.v2, 0.5),
        )

    def barycentric(self, p: Vector3) -> Tuple[float, float, float]:
        """Barycentric coordinates of *p* w.r.t. this triangle."""
        return barycentric_coordinates(p, self.v0, self.v1, self.v2)

    def contains_point(self, p: Vector3, eps: float = EPSILON) -> bool:
        """Return *True* if *p* lies on (or very near) this triangle's surface.

        The test checks that barycentric coordinates are all in [−eps, 1+eps]
        and that *p* is close to the triangle plane.
        """
        u, v, w = self.barycentric(p)
        if u < -eps or v < -eps or w < -eps:
            return False
        # Check distance to triangle plane.
        try:
            n = self.normal()
        except ValueError:
            return False
        plane_dist = abs((p - self.v0).dot(n))
        return plane_dist < max(eps, 1e-6)

    def vertices(self) -> Tuple[Vector3, Vector3, Vector3]:
        """Return the three vertices as a tuple."""
        return (self.v0, self.v1, self.v2)

    def __repr__(self) -> str:
        return f"Triangle({self.v0}, {self.v1}, {self.v2})"
