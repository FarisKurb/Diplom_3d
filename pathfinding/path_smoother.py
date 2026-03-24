"""Path smoothing via greedy visibility shortcutting.

After Dijkstra produces a shortest path through the visibility graph, the
result may contain redundant intermediate waypoints.  This module removes
them by iteratively checking whether non-adjacent waypoints can see each
other directly (i.e. no mesh intersection), thereby shortening the path
while preserving obstacle avoidance.
"""

from __future__ import annotations

from typing import List

from core.vector3 import Vector3
from geometry.triangle import Triangle
from geometry.visibility import is_visible


def smooth_path(
    points: List[Vector3],
    triangles: List[Triangle],
) -> List[Vector3]:
    """Remove redundant waypoints from *points* using greedy shortcutting.

    Starting from the first point, jump as far ahead as possible while the
    straight line to the later point does not intersect the mesh interior.
    Repeat from the farthest visible point until the end is reached.

    Args:
        points:    Ordered waypoints (at least 2).
        triangles: Mesh triangles to test visibility against.

    Returns:
        A new (usually shorter) list of waypoints that is a subsequence of
        *points*.  The first and last points are always preserved.
    """
    if len(points) <= 2:
        return list(points)

    smoothed: List[Vector3] = [points[0]]
    current = 0

    while current < len(points) - 1:
        # Try to skip ahead as far as possible.
        farthest = current + 1
        for candidate in range(len(points) - 1, current + 1, -1):
            if is_visible(points[current], points[candidate], triangles):
                farthest = candidate
                break
        smoothed.append(points[farthest])
        current = farthest

    return smoothed


def compute_path_length(points: List[Vector3]) -> float:
    """Return the total Euclidean length of a polyline path.

    Args:
        points: Ordered 3-D waypoints.

    Returns:
        Sum of segment lengths.  Returns ``0.0`` for fewer than 2 points.
    """
    total = 0.0
    for i in range(len(points) - 1):
        total += points[i].distance_to(points[i + 1])
    return total
