"""Visibility tests between two points in the presence of a mesh obstacle."""

from __future__ import annotations

from typing import List

from core.vector3 import Vector3
from geometry.triangle import Triangle
from geometry.intersection import segment_intersects_mesh


def is_visible(
    a: Vector3,
    b: Vector3,
    triangles: List[Triangle],
) -> bool:
    """Return *True* if the segment *a* → *b* does **not** pass through the
    mesh interior.

    The segment is allowed to:
      - travel through free space
      - touch or lie on the mesh surface
      - travel along mesh edges or faces

    It is **not** allowed to:
      - pass through the interior of the mesh

    This delegates to :func:`segment_intersects_mesh` from the
    intersection module which uses the two-crossing convex-interior test.
    """
    if a.approx_equal(b):
        return True
    return not segment_intersects_mesh(a, b, triangles)
