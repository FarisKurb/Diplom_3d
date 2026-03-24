"""Intersection algorithms for rays, segments, and triangles.

Core algorithms:
    - Möller–Trumbore ray-triangle intersection
    - Segment-triangle intersection
    - Segment-mesh intersection (against a list of triangles)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from core.vector3 import Vector3
from core.ray import Ray
from core.math_utils import EPSILON
from geometry.triangle import Triangle

if TYPE_CHECKING:
    pass  # future type-only imports


# ── Ray–Triangle (Möller–Trumbore) ─────────────────────────


def ray_triangle_intersection(
    ray: Ray,
    tri: Triangle,
    *,
    cull_backface: bool = False,
    eps: float = EPSILON,
) -> Optional[Tuple[float, float, float]]:
    """Möller–Trumbore ray–triangle intersection.

    Args:
        ray:           The ray to test.
        tri:           The triangle to test against.
        cull_backface: If *True*, ignore hits on the back face.
        eps:           Floating-point tolerance.

    Returns:
        ``(t, u, v)`` where *t* is the ray parameter and *(u, v)* are
        barycentric coordinates of the hit point on the triangle, or
        ``None`` if no intersection.
    """
    edge1 = tri.v1 - tri.v0
    edge2 = tri.v2 - tri.v0

    h = ray.direction.cross(edge2)
    det = edge1.dot(h)

    if cull_backface:
        if det < eps:
            return None
    else:
        if abs(det) < eps:
            return None

    inv_det = 1.0 / det
    s = ray.origin - tri.v0
    u = s.dot(h) * inv_det
    if u < -eps or u > 1.0 + eps:
        return None

    q = s.cross(edge1)
    v = ray.direction.dot(q) * inv_det
    if v < -eps or u + v > 1.0 + eps:
        return None

    t = edge2.dot(q) * inv_det
    if t < -eps:
        return None

    return (t, u, v)


# ── Segment–Triangle ───────────────────────────────────────


def segment_triangle_intersection(
    p0: Vector3,
    p1: Vector3,
    tri: Triangle,
    *,
    eps: float = EPSILON,
) -> Optional[Tuple[float, float, float]]:
    """Test whether the line segment *p0* → *p1* intersects *tri*.

    Returns:
        ``(t, u, v)`` where ``hit = p0 + t*(p1 - p0)``, or ``None``.
        *t* is in [0, 1] for a valid segment hit.
    """
    seg_dir = p1 - p0
    seg_len_sq = seg_dir.length_squared()
    if seg_len_sq < eps * eps:
        # Degenerate segment (a point) — check containment.
        if tri.contains_point(p0, eps):
            return (0.0, 0.0, 0.0)
        return None

    edge1 = tri.v1 - tri.v0
    edge2 = tri.v2 - tri.v0

    h = seg_dir.cross(edge2)
    det = edge1.dot(h)

    if abs(det) < eps:
        return None  # segment parallel to triangle

    inv_det = 1.0 / det
    s = p0 - tri.v0
    u = s.dot(h) * inv_det
    if u < -eps or u > 1.0 + eps:
        return None

    q = s.cross(edge1)
    v = seg_dir.dot(q) * inv_det
    if v < -eps or u + v > 1.0 + eps:
        return None

    t = edge2.dot(q) * inv_det
    if t < -eps or t > 1.0 + eps:
        return None

    return (t, u, v)


# ── Segment–Mesh ───────────────────────────────────────────


def segment_intersects_mesh(
    p0: Vector3,
    p1: Vector3,
    triangles: List[Triangle],
    *,
    eps: float = EPSILON,
) -> bool:
    """Return *True* if the segment *p0* → *p1* intersects the **interior**
    of the mesh represented by *triangles*.

    A hit that lies exactly on the mesh surface (within *eps*) is **not**
    considered an interior intersection — this allows paths that travel
    along faces / edges.

    Strategy:
        1. Count signed crossings.  For a closed convex mesh, a segment that
           enters the interior must cross ≥2 faces with the segment strictly
           passing through.
        2. When both endpoints lie on the surface (endpoint hits were
           skipped) and no interior crossings were found, test whether the
           midpoint of the segment is inside the mesh via ray casting.
    """
    hit_ts: list[float] = []
    had_start_hit = False
    had_end_hit = False

    for tri in triangles:
        result = segment_triangle_intersection(p0, p1, tri, eps=eps)
        if result is None:
            continue
        t, u, v = result
        # Track endpoint hits before skipping them.
        if t < eps:
            had_start_hit = True
            continue
        if t > 1.0 - eps:
            had_end_hit = True
            continue
        # Deduplicate: skip if we already have a hit at essentially the same t.
        if any(abs(t - prev) < eps for prev in hit_ts):
            continue
        hit_ts.append(t)

    # For a convex mesh, two distinct crossing points means the segment
    # enters and exits → it passes through the interior.
    if len(hit_ts) >= 2:
        return True

    # One endpoint on the surface + at least one interior crossing means
    # the segment enters the mesh at the crossing and exits (or starts)
    # at the surface endpoint → interior passage.
    if (had_start_hit or had_end_hit) and len(hit_ts) >= 1:
        return True

    # When both endpoints are on the surface but no interior crossings
    # were detected, the segment might still pass through the interior
    # (e.g. front face → back face of a cube).  Check the midpoint.
    if had_start_hit and had_end_hit and len(hit_ts) == 0:
        mid = (p0 + p1) * 0.5
        # If the midpoint lies on the mesh surface, the segment travels
        # along the surface and does NOT pass through the interior.
        on_surface = any(tri.contains_point(mid, eps=1e-6) for tri in triangles)
        if not on_surface and _point_inside_mesh(mid, triangles, eps=eps):
            return True

    return False


def _point_inside_mesh(
    point: Vector3,
    triangles: List[Triangle],
    *,
    eps: float = EPSILON,
) -> bool:
    """Return *True* if *point* is inside a closed mesh (ray-casting parity test).

    Hits at the same *t* (shared edges / vertices) are deduplicated so
    that a single crossing through a shared edge counts once.
    """
    ray = Ray(point, Vector3(1.0, 0.0, 0.0))
    hit_ts: list[float] = []
    for tri in triangles:
        result = ray_triangle_intersection(ray, tri, eps=eps)
        if result is not None:
            t, _u, _v = result
            if t > eps:
                # Deduplicate hits at essentially the same t.
                if not any(abs(t - prev) < eps * 1000 for prev in hit_ts):
                    hit_ts.append(t)
    return len(hit_ts) % 2 == 1


def segment_mesh_closest_intersection(
    p0: Vector3,
    p1: Vector3,
    triangles: List[Triangle],
    *,
    eps: float = EPSILON,
) -> Optional[Tuple[float, Vector3, int]]:
    """Return the closest intersection of the segment with the mesh.

    Returns:
        ``(t, hit_point, triangle_index)`` for the nearest hit, or ``None``.
    """
    best_t: Optional[float] = None
    best_point: Optional[Vector3] = None
    best_idx: int = -1

    seg_dir = p1 - p0

    for i, tri in enumerate(triangles):
        result = segment_triangle_intersection(p0, p1, tri, eps=eps)
        if result is None:
            continue
        t, _u, _v = result
        if best_t is None or t < best_t:
            best_t = t
            best_point = p0 + seg_dir * t
            best_idx = i

    if best_t is None:
        return None
    assert best_point is not None
    return (best_t, best_point, best_idx)
