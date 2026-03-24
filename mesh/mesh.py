"""Core mesh data structure holding vertices, faces, and derived triangles."""

from __future__ import annotations

from typing import List, Tuple

from core.vector3 import Vector3
from geometry.triangle import Triangle


class Mesh:
    """A triangle mesh defined by a list of vertices and face indices.

    Each face is a tuple of three 0-based vertex indices forming a triangle.

    Attributes:
        vertices: List of vertex positions.
        faces:    List of index triples (i0, i1, i2).
    """

    def __init__(
        self,
        vertices: List[Vector3],
        faces: List[Tuple[int, int, int]],
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self._triangles: List[Triangle] | None = None

    # ── derived data ────────────────────────────────────────

    @property
    def triangles(self) -> List[Triangle]:
        """Lazily build and cache the list of Triangle objects."""
        if self._triangles is None:
            self._triangles = [
                Triangle(
                    self.vertices[f[0]],
                    self.vertices[f[1]],
                    self.vertices[f[2]],
                )
                for f in self.faces
            ]
        return self._triangles

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    # ── bounding info ───────────────────────────────────────

    def bounding_box(self) -> Tuple[Vector3, Vector3]:
        """Axis-aligned bounding box as (min_corner, max_corner)."""
        if not self.vertices:
            return (Vector3.zero(), Vector3.zero())
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        return (
            Vector3(min(xs), min(ys), min(zs)),
            Vector3(max(xs), max(ys), max(zs)),
        )

    def center(self) -> Vector3:
        """Geometric center (average of bounding box corners)."""
        lo, hi = self.bounding_box()
        return (lo + hi) / 2.0

    def invalidate_cache(self) -> None:
        """Clear cached derived data (call after modifying vertices/faces)."""
        self._triangles = None

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.num_vertices}, faces={self.num_faces})"
