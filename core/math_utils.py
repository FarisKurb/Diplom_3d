"""Low-level math helpers shared across the project."""

from __future__ import annotations

import math
from typing import Tuple

from core.vector3 import Vector3

# Default tolerance used for floating-point comparisons.
EPSILON: float = 1e-9


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(hi, value))


def point_to_segment_distance(
    point: Vector3, seg_start: Vector3, seg_end: Vector3
) -> float:
    """Return the minimum Euclidean distance from *point* to the line segment
    *seg_start* → *seg_end*.
    """
    seg = seg_end - seg_start
    seg_len_sq = seg.length_squared()
    if seg_len_sq < EPSILON * EPSILON:
        return point.distance_to(seg_start)
    t = clamp((point - seg_start).dot(seg) / seg_len_sq, 0.0, 1.0)
    projection = seg_start + seg * t
    return point.distance_to(projection)


def closest_point_on_segment(
    point: Vector3, seg_start: Vector3, seg_end: Vector3
) -> Vector3:
    """Return the closest point on segment *seg_start* → *seg_end* to *point*."""
    seg = seg_end - seg_start
    seg_len_sq = seg.length_squared()
    if seg_len_sq < EPSILON * EPSILON:
        return seg_start
    t = clamp((point - seg_start).dot(seg) / seg_len_sq, 0.0, 1.0)
    return seg_start + seg * t


def segments_closest_distance(
    a0: Vector3, a1: Vector3, b0: Vector3, b1: Vector3
) -> float:
    """Return the minimum distance between two line segments in 3-D."""
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0

    a = d1.dot(d1)
    e = d2.dot(d2)
    f = d2.dot(r)

    if a < EPSILON and e < EPSILON:
        return r.length()

    if a < EPSILON:
        s = 0.0
        t = clamp(f / e, 0.0, 1.0)
    else:
        c = d1.dot(r)
        if e < EPSILON:
            t = 0.0
            s = clamp(-c / a, 0.0, 1.0)
        else:
            b_ = d1.dot(d2)
            denom = a * e - b_ * b_
            if abs(denom) > EPSILON:
                s = clamp((b_ * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b_ * s + f) / e
            if t < 0.0:
                t = 0.0
                s = clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = clamp((b_ - c) / a, 0.0, 1.0)

    closest_a = a0 + d1 * s
    closest_b = b0 + d2 * t
    return closest_a.distance_to(closest_b)


def barycentric_coordinates(
    p: Vector3, a: Vector3, b: Vector3, c: Vector3
) -> Tuple[float, float, float]:
    """Compute the barycentric coordinates *(u, v, w)* of *p* with respect to
    triangle *ABC*.  ``p ≈ u*A + v*B + w*C``.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < EPSILON:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)


def triangle_area(a: Vector3, b: Vector3, c: Vector3) -> float:
    """Return the area of triangle *ABC*."""
    return 0.5 * (b - a).cross(c - a).length()


def triangle_normal(a: Vector3, b: Vector3, c: Vector3) -> Vector3:
    """Return the unit outward normal of triangle *ABC*.

    Raises:
        ValueError: If the triangle is degenerate.
    """
    return (b - a).cross(c - a).normalized()
