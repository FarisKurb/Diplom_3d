"""Distance helper functions for points, segments, and triangles."""

from __future__ import annotations

from core.vector3 import Vector3
from core.math_utils import (
    EPSILON,
    clamp,
    point_to_segment_distance,
    closest_point_on_segment,
)
from geometry.triangle import Triangle


def point_to_triangle_distance(point: Vector3, tri: Triangle) -> float:
    """Return the minimum distance from *point* to triangle *tri*.

    Considers the interior of the triangle as well as all three edges.
    """
    # Project onto the triangle plane.
    try:
        n = tri.normal()
    except ValueError:
        # Degenerate triangle — fall back to edge distances.
        d0 = point_to_segment_distance(point, tri.v0, tri.v1)
        d1 = point_to_segment_distance(point, tri.v1, tri.v2)
        d2 = point_to_segment_distance(point, tri.v0, tri.v2)
        return min(d0, d1, d2)

    # Signed distance to the plane.
    diff = point - tri.v0
    plane_dist = diff.dot(n)
    projected = point - n * plane_dist

    # Check if the projection lies inside the triangle.
    if tri.contains_point(projected, eps=EPSILON):
        return abs(plane_dist)

    # Otherwise the closest point is on one of the edges.
    d0 = point_to_segment_distance(point, tri.v0, tri.v1)
    d1 = point_to_segment_distance(point, tri.v1, tri.v2)
    d2 = point_to_segment_distance(point, tri.v0, tri.v2)
    return min(d0, d1, d2)


def closest_point_on_triangle(point: Vector3, tri: Triangle) -> Vector3:
    """Return the closest point on triangle *tri* to *point*."""
    try:
        n = tri.normal()
    except ValueError:
        # Degenerate — closest point on edges.
        candidates = [
            closest_point_on_segment(point, tri.v0, tri.v1),
            closest_point_on_segment(point, tri.v1, tri.v2),
            closest_point_on_segment(point, tri.v0, tri.v2),
        ]
        return min(candidates, key=lambda c: point.distance_to(c))

    diff = point - tri.v0
    plane_dist = diff.dot(n)
    projected = point - n * plane_dist

    if tri.contains_point(projected, eps=EPSILON):
        return projected

    candidates = [
        closest_point_on_segment(point, tri.v0, tri.v1),
        closest_point_on_segment(point, tri.v1, tri.v2),
        closest_point_on_segment(point, tri.v0, tri.v2),
    ]
    return min(candidates, key=lambda c: point.distance_to(c))
