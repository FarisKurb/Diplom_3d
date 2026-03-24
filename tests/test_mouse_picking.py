"""Tests for Stage 8 — Mouse picking and point placement.

Covers:
    - Camera.screen_to_ray (ray unprojection)
    - ray_mesh_closest_intersection
    - pick_mesh (integration)
    - PointPlacer state machine
"""

from __future__ import annotations

import math
import sys
from types import ModuleType
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

# Stub OpenGL / glfw so imports work without a GPU.
_gl_stub = MagicMock()
for mod in ("OpenGL", "OpenGL.GL", "OpenGL.GLU", "glfw"):
    sys.modules.setdefault(mod, _gl_stub)

from core.vector3 import Vector3
from core.ray import Ray
from geometry.triangle import Triangle
from mesh.mesh import Mesh
from render.camera import Camera
from interaction.mouse_picking import pick_mesh, ray_mesh_closest_intersection
from interaction.point_placer import PointPlacer, PlacementState


# ── helpers ─────────────────────────────────────────────────

def _unit_cube_mesh() -> Mesh:
    """Axis-aligned unit cube centred at origin."""
    verts = [
        Vector3(-0.5, -0.5,  0.5), Vector3( 0.5, -0.5,  0.5),
        Vector3( 0.5,  0.5,  0.5), Vector3(-0.5,  0.5,  0.5),
        Vector3(-0.5, -0.5, -0.5), Vector3( 0.5, -0.5, -0.5),
        Vector3( 0.5,  0.5, -0.5), Vector3(-0.5,  0.5, -0.5),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),  # front (+Z)
        (4, 7, 6), (4, 6, 5),  # back  (-Z)
        (0, 3, 7), (0, 7, 4),  # left  (-X)
        (1, 5, 6), (1, 6, 2),  # right (+X)
        (3, 2, 6), (3, 6, 7),  # top   (+Y)
        (0, 4, 5), (0, 5, 1),  # bottom (-Y)
    ]
    return Mesh(verts, faces)


def _single_tri_mesh() -> Mesh:
    """A single large triangle facing +Z at z=0."""
    return Mesh(
        vertices=[Vector3(-5, -5, 0), Vector3(5, -5, 0), Vector3(0, 5, 0)],
        faces=[(0, 1, 2)],
    )


# ═══════════════════════════════════════════════════════════
#  Camera.screen_to_ray
# ═══════════════════════════════════════════════════════════

class TestScreenToRay:
    def test_center_pixel_looks_forward(self) -> None:
        """A ray through the centre of the screen should be ≈ camera forward."""
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        ray = cam.screen_to_ray(640, 360, 1280, 720)
        # Forward vector for yaw=0, pitch=0 is (0, 0, -1).
        fwd = cam.forward
        dot = ray.direction.dot(fwd)
        assert dot > 0.999, f"Center ray not aligned with forward: dot={dot}"

    def test_origin_is_camera_position(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        ray = cam.screen_to_ray(640, 360, 1280, 720)
        assert ray.origin.approx_equal(cam.position, 1e-6)

    def test_left_pixel_aims_left(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        ray_center = cam.screen_to_ray(640, 360, 1280, 720)
        ray_left = cam.screen_to_ray(0, 360, 1280, 720)
        # At yaw=0, right is +X.  Left pixel should have smaller X component.
        right = cam.right
        assert ray_left.direction.dot(right) < ray_center.direction.dot(right)

    def test_top_pixel_aims_up(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        ray_center = cam.screen_to_ray(640, 360, 1280, 720)
        ray_top = cam.screen_to_ray(640, 0, 1280, 720)
        up = cam.up
        assert ray_top.direction.dot(up) > ray_center.direction.dot(up)

    def test_different_viewport_sizes(self) -> None:
        """screen_to_ray should work for non-default viewport sizes."""
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=3.0)
        ray = cam.screen_to_ray(400, 300, 800, 600)
        assert ray.direction.dot(cam.forward) > 0.999

    def test_rotated_camera(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=90.0, pitch=0.0, distance=5.0)
        ray = cam.screen_to_ray(640, 360, 1280, 720)
        fwd = cam.forward
        assert ray.direction.dot(fwd) > 0.999


# ═══════════════════════════════════════════════════════════
#  ray_mesh_closest_intersection
# ═══════════════════════════════════════════════════════════

class TestRayMeshIntersection:
    def test_hits_single_triangle(self) -> None:
        tri = Triangle(Vector3(-1, -1, 0), Vector3(1, -1, 0), Vector3(0, 1, 0))
        ray = Ray(Vector3(0, 0, 5), Vector3(0, 0, -1))
        result = ray_mesh_closest_intersection(ray, [tri])
        assert result is not None
        hit, idx = result
        assert idx == 0
        assert abs(hit.z) < 0.01

    def test_misses_triangle(self) -> None:
        tri = Triangle(Vector3(-1, -1, 0), Vector3(1, -1, 0), Vector3(0, 1, 0))
        ray = Ray(Vector3(10, 10, 5), Vector3(0, 0, -1))
        result = ray_mesh_closest_intersection(ray, [tri])
        assert result is None

    def test_closest_of_two(self) -> None:
        """When a ray hits two triangles, return the closest one."""
        t_near = Triangle(Vector3(-1, -1, 2), Vector3(1, -1, 2), Vector3(0, 1, 2))
        t_far  = Triangle(Vector3(-1, -1, 0), Vector3(1, -1, 0), Vector3(0, 1, 0))
        ray = Ray(Vector3(0, 0, 5), Vector3(0, 0, -1))
        result = ray_mesh_closest_intersection(ray, [t_near, t_far])
        assert result is not None
        hit, idx = result
        assert idx == 0  # t_near is at index 0
        assert abs(hit.z - 2.0) < 0.01

    def test_ignores_behind_camera(self) -> None:
        tri = Triangle(Vector3(-1, -1, 0), Vector3(1, -1, 0), Vector3(0, 1, 0))
        ray = Ray(Vector3(0, 0, -5), Vector3(0, 0, -1))  # pointing away
        result = ray_mesh_closest_intersection(ray, [tri])
        assert result is None

    def test_hits_cube(self) -> None:
        mesh = _unit_cube_mesh()
        ray = Ray(Vector3(0, 0, 5), Vector3(0, 0, -1))
        result = ray_mesh_closest_intersection(ray, mesh.triangles)
        assert result is not None
        hit, _idx = result
        assert abs(hit.z - 0.5) < 0.01


# ═══════════════════════════════════════════════════════════
#  pick_mesh (integration)
# ═══════════════════════════════════════════════════════════

class TestPickMesh:
    def test_center_click_hits_cube(self) -> None:
        """Clicking center of screen on a cube in front of camera should hit."""
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        mesh = _unit_cube_mesh()
        result = pick_mesh(640, 360, 1280, 720, cam, mesh)
        assert result is not None
        hit, tri_idx = result
        assert 0 <= tri_idx < mesh.num_faces
        # Hit should be on or near the +Z face of the cube.
        assert abs(hit.z - 0.5) < 0.05

    def test_miss_off_mesh(self) -> None:
        """Clicking far from the mesh should return None."""
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        mesh = _unit_cube_mesh()
        # Click in the extreme corner of the screen.
        result = pick_mesh(0, 0, 1280, 720, cam, mesh)
        assert result is None

    def test_rotated_camera_pick(self) -> None:
        """pick_mesh should work with a rotated camera."""
        cam = Camera(target=Vector3.zero(), yaw=90.0, pitch=0.0, distance=5.0)
        mesh = _unit_cube_mesh()
        result = pick_mesh(640, 360, 1280, 720, cam, mesh)
        assert result is not None
        hit, _idx = result
        # At yaw=90, camera is on +X axis looking toward origin.
        assert abs(hit.x - 0.5) < 0.05


# ═══════════════════════════════════════════════════════════
#  PointPlacer
# ═══════════════════════════════════════════════════════════

class TestPointPlacerInit:
    def test_initial_state(self) -> None:
        mesh = _unit_cube_mesh()
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        pp = PointPlacer(mesh, cam, lambda: (1280, 720), lambda: (640.0, 360.0))
        assert pp.state == PlacementState.PLACE_START
        assert pp.start_point is None
        assert pp.end_point is None


class TestPointPlacerPlacement:
    def _make_placer(
        self,
        cursor_pos: tuple[float, float] = (640.0, 360.0),
    ) -> PointPlacer:
        mesh = _unit_cube_mesh()
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        return PointPlacer(
            mesh, cam,
            get_viewport_size=lambda: (1280, 720),
            get_cursor_pos=lambda: cursor_pos,
        )

    def test_first_click_sets_start(self) -> None:
        pp = self._make_placer()
        pp.on_click(button=0, action=1, mods=0)
        assert pp.start_point is not None
        assert pp.end_point is None
        assert pp.state == PlacementState.PLACE_END

    def test_second_click_sets_end(self) -> None:
        pp = self._make_placer()
        pp.on_click(button=0, action=1, mods=0)
        pp.on_click(button=0, action=1, mods=0)
        assert pp.start_point is not None
        assert pp.end_point is not None
        assert pp.state == PlacementState.DONE

    def test_third_click_ignored(self) -> None:
        pp = self._make_placer()
        pp.on_click(button=0, action=1, mods=0)
        pp.on_click(button=0, action=1, mods=0)
        old_end = pp.end_point
        pp.on_click(button=0, action=1, mods=0)
        # State should remain DONE, end point unchanged.
        assert pp.state == PlacementState.DONE
        assert pp.end_point is old_end

    def test_right_click_ignored(self) -> None:
        pp = self._make_placer()
        pp.on_click(button=1, action=1, mods=0)  # right button
        assert pp.state == PlacementState.PLACE_START
        assert pp.start_point is None

    def test_release_ignored(self) -> None:
        pp = self._make_placer()
        pp.on_click(button=0, action=0, mods=0)  # release
        assert pp.state == PlacementState.PLACE_START

    def test_callback_fired(self) -> None:
        pp = self._make_placer()
        received: list[tuple[Vector3, Vector3]] = []
        pp.on_both_placed = lambda s, e: received.append((s, e))
        pp.on_click(button=0, action=1, mods=0)
        pp.on_click(button=0, action=1, mods=0)
        assert len(received) == 1
        assert received[0][0] is not None
        assert received[0][1] is not None


class TestPointPlacerReset:
    def test_reset(self) -> None:
        mesh = _unit_cube_mesh()
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        pp = PointPlacer(mesh, cam, lambda: (1280, 720), lambda: (640.0, 360.0))
        pp.on_click(button=0, action=1, mods=0)
        pp.on_click(button=0, action=1, mods=0)
        pp.reset()
        assert pp.state == PlacementState.PLACE_START
        assert pp.start_point is None
        assert pp.end_point is None


class TestPointPlacerMiss:
    def test_click_misses_mesh_places_in_free_space(self) -> None:
        """Clicking empty space places a point on the plane through mesh centre."""
        mesh = _unit_cube_mesh()
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        # Cursor in extreme corner — misses the cube, but free-space fallback fires.
        pp = PointPlacer(mesh, cam, lambda: (1280, 720), lambda: (0.0, 0.0))
        pp.on_click(button=0, action=1, mods=0)
        assert pp.state == PlacementState.PLACE_END
        assert pp.start_point is not None
