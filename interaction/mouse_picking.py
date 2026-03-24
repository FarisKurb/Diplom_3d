"""Mouse-picking utilities: cast a ray from screen coords and find the
closest mesh intersection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from core.vector3 import Vector3
from core.ray import Ray
from core.math_utils import EPSILON
from geometry.triangle import Triangle
from geometry.intersection import ray_triangle_intersection
from mesh.mesh import Mesh
from render.camera import Camera


def pick_mesh(
    screen_x: float,
    screen_y: float,
    viewport_width: int,
    viewport_height: int,
    camera: Camera,
    mesh: Mesh,
    *,
    eps: float = EPSILON,
) -> Optional[Tuple[Vector3, int]]:
    """Cast a ray from a screen pixel through the camera and return the
    closest intersection with *mesh*.

    Args:
        screen_x / screen_y: Pixel coordinates (GLFW convention, origin top-left).
        viewport_width / viewport_height: Framebuffer dimensions.
        camera: The active camera.
        mesh: The mesh to test against.
        eps: Floating-point tolerance.

    Returns:
        ``(hit_point, triangle_index)`` for the nearest intersection, or
        ``None`` if the ray misses the mesh entirely.
    """
    ray = camera.screen_to_ray(screen_x, screen_y, viewport_width, viewport_height)
    return ray_mesh_closest_intersection(ray, mesh.triangles, eps=eps)


def ray_mesh_closest_intersection(
    ray: Ray,
    triangles: List[Triangle],
    *,
    eps: float = EPSILON,
) -> Optional[Tuple[Vector3, int]]:
    """Find the closest intersection of *ray* with a list of triangles.

    Returns:
        ``(hit_point, triangle_index)`` or ``None``.
    """
    best_t: Optional[float] = None
    best_idx: int = -1

    for i, tri in enumerate(triangles):
        result = ray_triangle_intersection(ray, tri, eps=eps)
        if result is None:
            continue
        t, _u, _v = result
        if t < eps:
            continue  # behind camera
        if best_t is None or t < best_t:
            best_t = t
            best_idx = i

    if best_t is None:
        return None

    hit_point = ray.point_at(best_t)
    return (hit_point, best_idx)
