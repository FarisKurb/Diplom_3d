"""Generate sample points on mesh triangle faces for the visibility graph."""

from __future__ import annotations

from typing import List

from core.vector3 import Vector3
from geometry.triangle import Triangle
from mesh.mesh import Mesh


def sample_triangle(
    tri: Triangle,
    bary_steps: int = 3,
    include_centroid: bool = True,
    include_edge_midpoints: bool = True,
) -> List[Vector3]:
    """Generate sample points on a single triangle.

    Args:
        tri:                   The triangle to sample.
        bary_steps:            Number of subdivisions along each barycentric
                               axis.  ``bary_steps=3`` yields an interior grid
                               excluding the vertices themselves.
        include_centroid:      Whether to include the triangle centroid.
        include_edge_midpoints: Whether to include the three edge midpoints.

    Returns:
        A list of unique sample points (does **not** include the triangle
        vertices — those are already in the graph as mesh vertices).
    """
    points: List[Vector3] = []
    seen: set[tuple[float, float, float]] = set()

    def _add(p: Vector3) -> None:
        key = (round(p.x, 9), round(p.y, 9), round(p.z, 9))
        if key not in seen:
            seen.add(key)
            points.append(p)

    if include_centroid:
        _add(tri.centroid())

    if include_edge_midpoints:
        for mp in tri.edge_midpoints():
            _add(mp)

    # Barycentric grid — skip corners (those are mesh vertices).
    if bary_steps >= 2:
        n = bary_steps
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                # Skip pure vertex positions (one coordinate == n).
                if i == n or j == n or k == n:
                    continue
                u = i / n
                v = j / n
                w = k / n
                p = tri.v0 * u + tri.v1 * v + tri.v2 * w
                _add(p)

    return points


def sample_mesh_faces(
    mesh: Mesh,
    bary_steps: int = 3,
    include_centroid: bool = True,
    include_edge_midpoints: bool = True,
) -> List[Vector3]:
    """Generate sample points across all faces of a mesh.

    Duplicate points (shared edge midpoints, etc.) are removed.

    Returns:
        A deduplicated list of sample points.
    """
    global_seen: set[tuple[float, float, float]] = set()
    all_points: List[Vector3] = []

    for tri in mesh.triangles:
        pts = sample_triangle(
            tri,
            bary_steps=bary_steps,
            include_centroid=include_centroid,
            include_edge_midpoints=include_edge_midpoints,
        )
        for p in pts:
            key = (round(p.x, 9), round(p.y, 9), round(p.z, 9))
            if key not in global_seen:
                global_seen.add(key)
                all_points.append(p)

    return all_points
