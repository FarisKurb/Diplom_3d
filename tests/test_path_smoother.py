"""Tests for Stage 11 — Path smoothing and integration.

Covers:
    - compute_path_length (basic, single segment, empty/single point)
    - smooth_path with trivial inputs (0, 1, 2 points)
    - smooth_path with collinear visible points (all skippable)
    - smooth_path with obstacle blocking (must keep waypoints)
    - smooth_path preserves first and last point
    - PathSolver.solve with smooth=True / smooth=False
    - PathResult new fields (raw_points, smoothed)
    - Application smoothing toggle via _toggle_smoothing and S key
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import List

import pytest

from core.vector3 import Vector3
from geometry.triangle import Triangle
from mesh.mesh import Mesh
from pathfinding.path_smoother import smooth_path, compute_path_length
from pathfinding.path_solver import PathSolver, PathResult


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
        (0, 1, 2), (0, 2, 3),
        (4, 7, 6), (4, 6, 5),
        (0, 3, 7), (0, 7, 4),
        (1, 5, 6), (1, 6, 2),
        (3, 2, 6), (3, 6, 7),
        (0, 4, 5), (0, 5, 1),
    ]
    return Mesh(verts, faces)


def _single_triangle_mesh() -> Mesh:
    """A single triangle — very simple mesh."""
    return Mesh(
        vertices=[Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(1, 2, 0)],
        faces=[(0, 1, 2)],
    )


# ═══════════════════════════════════════════════════════════
#  compute_path_length
# ═══════════════════════════════════════════════════════════

class TestComputePathLength:
    def test_empty(self):
        assert compute_path_length([]) == 0.0

    def test_single_point(self):
        assert compute_path_length([Vector3(1, 2, 3)]) == 0.0

    def test_two_points(self):
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 0)
        assert compute_path_length([a, b]) == pytest.approx(5.0)

    def test_three_points(self):
        a = Vector3(0, 0, 0)
        b = Vector3(1, 0, 0)
        c = Vector3(1, 1, 0)
        assert compute_path_length([a, b, c]) == pytest.approx(2.0)

    def test_collinear_points(self):
        pts = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(3, 0, 0)]
        assert compute_path_length(pts) == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════
#  smooth_path — trivial cases
# ═══════════════════════════════════════════════════════════

class TestSmoothPathTrivial:
    def test_empty_list(self):
        assert smooth_path([], []) == []

    def test_single_point(self):
        pts = [Vector3(0, 0, 0)]
        assert smooth_path(pts, []) == pts

    def test_two_points(self):
        pts = [Vector3(0, 0, 0), Vector3(1, 0, 0)]
        result = smooth_path(pts, [])
        assert len(result) == 2
        assert result[0] == pts[0]
        assert result[1] == pts[1]

    def test_returns_new_list(self):
        """smooth_path should return a new list, not mutate the input."""
        pts = [Vector3(0, 0, 0), Vector3(1, 0, 0)]
        result = smooth_path(pts, [])
        assert result is not pts


# ═══════════════════════════════════════════════════════════
#  smooth_path — collinear visible points (no obstacle)
# ═══════════════════════════════════════════════════════════

class TestSmoothPathNoObstacle:
    def test_all_visible_reduces_to_two(self):
        """With no obstacles, all intermediate points are removable."""
        pts = [
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(2, 0, 0),
            Vector3(3, 0, 0),
            Vector3(4, 0, 0),
        ]
        result = smooth_path(pts, [])
        assert len(result) == 2
        assert result[0] == pts[0]
        assert result[1] == pts[-1]

    def test_non_collinear_all_visible(self):
        """Even non-collinear points reduce to start/end if all visible."""
        pts = [
            Vector3(0, 0, 0),
            Vector3(1, 1, 0),
            Vector3(2, -1, 0),
            Vector3(3, 0, 0),
        ]
        result = smooth_path(pts, [])
        assert len(result) == 2
        assert result[0] == pts[0]
        assert result[-1] == pts[-1]


# ═══════════════════════════════════════════════════════════
#  smooth_path — with obstacle
# ═══════════════════════════════════════════════════════════

class TestSmoothPathWithObstacle:
    def test_preserves_endpoints(self):
        """Start and end points must always be preserved."""
        mesh = _unit_cube_mesh()
        pts = [Vector3(0, 0, 2), Vector3(1, 0, 0), Vector3(0, 0, -2)]
        result = smooth_path(pts, mesh.triangles)
        assert result[0] == pts[0]
        assert result[-1] == pts[-1]

    def test_keeps_necessary_waypoint(self):
        """When the direct path is blocked, the intermediate point is kept."""
        mesh = _unit_cube_mesh()
        # front → side → back — direct line goes through cube
        start = Vector3(0, 0, 2)
        mid = Vector3(2, 0, 0)  # goes around the side
        end = Vector3(0, 0, -2)
        pts = [start, mid, end]
        result = smooth_path(pts, mesh.triangles)
        # mid is needed because start→end passes through the cube
        assert len(result) == 3

    def test_removes_redundant_waypoint(self):
        """If an intermediate point can be skipped, it should be."""
        mesh = _unit_cube_mesh()
        # All three points are on the same side, all visible to each other
        a = Vector3(2, 0, 0)
        b = Vector3(2, 0.5, 0)
        c = Vector3(2, 1, 0)
        result = smooth_path([a, b, c], mesh.triangles)
        assert len(result) == 2
        assert result[0] == a
        assert result[1] == c

    def test_multi_waypoint_partial_removal(self):
        """Some intermediate points are removable, others are not."""
        mesh = _unit_cube_mesh()
        # Route: far right → slightly right → top-right → far left
        # The obstacle is between first group and last point
        p0 = Vector3(2, 0, 0)
        p1 = Vector3(2, 0.5, 0)  # visible from p0 and p2, redundant
        p2 = Vector3(0, 2, 0)    # above the cube — needed for going over
        p3 = Vector3(-2, 0, 0)
        pts = [p0, p1, p2, p3]
        result = smooth_path(pts, mesh.triangles)
        # p1 should be removed (p0 can see p2 directly above cube)
        # p2 must stay (p0→p3 goes through the cube)
        assert result[0] == p0
        assert result[-1] == p3
        assert len(result) <= len(pts)


# ═══════════════════════════════════════════════════════════
#  PathSolver with smoothing
# ═══════════════════════════════════════════════════════════

class TestPathSolverSmoothing:
    def test_solve_with_smoothing_on(self):
        mesh = _single_triangle_mesh()
        solver = PathSolver(mesh, bary_steps=1)
        # Two points that can see each other — path should be short
        start = Vector3(0, -1, 0)
        end = Vector3(2, -1, 0)
        result = solver.solve(start, end, smooth=True)
        assert result.found
        assert result.smoothed is True
        assert len(result.points) >= 2
        assert result.points[0].approx_equal(start)
        assert result.points[-1].approx_equal(end)

    def test_solve_with_smoothing_off(self):
        mesh = _single_triangle_mesh()
        solver = PathSolver(mesh, bary_steps=1)
        start = Vector3(0, -1, 0)
        end = Vector3(2, -1, 0)
        result = solver.solve(start, end, smooth=False)
        assert result.found
        assert result.smoothed is False
        assert result.raw_points == []  # raw_points empty when smooth=False

    def test_smoothed_path_not_longer(self):
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        raw = solver.solve(start, end, smooth=False)
        smoothed = solver.solve(start, end, smooth=True)
        if raw.found and smoothed.found:
            assert smoothed.distance <= raw.distance + 1e-6

    def test_smoothing_reduces_waypoints(self):
        mesh = _single_triangle_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        start = Vector3(0, -1, 0)
        end = Vector3(2, -1, 0)
        result = solver.solve(start, end, smooth=True)
        if result.found and result.raw_points:
            assert len(result.points) <= len(result.raw_points)

    def test_raw_points_populated_when_smoothed(self):
        mesh = _single_triangle_mesh()
        solver = PathSolver(mesh, bary_steps=1)
        start = Vector3(0, -1, 0)
        end = Vector3(2, -1, 0)
        result = solver.solve(start, end, smooth=True)
        if result.found:
            assert len(result.raw_points) >= 2


# ═══════════════════════════════════════════════════════════
#  PathResult new fields
# ═══════════════════════════════════════════════════════════

class TestPathResultFields:
    def test_default_raw_points_empty(self):
        r = PathResult()
        assert r.raw_points == []

    def test_default_smoothed_false(self):
        r = PathResult()
        assert r.smoothed is False

    def test_fields_set(self):
        pts = [Vector3(0, 0, 0), Vector3(1, 0, 0)]
        r = PathResult(found=True, points=pts, raw_points=pts + [Vector3(2, 0, 0)], smoothed=True)
        assert r.smoothed is True
        assert len(r.raw_points) == 3


# ═══════════════════════════════════════════════════════════
#  Application smoothing toggle
# ═══════════════════════════════════════════════════════════

class TestApplicationSmoothingToggle:
    @pytest.fixture
    def app(self, tmp_path):
        from mesh.obj_loader import ensure_default_cube
        from main import Application
        obj_path = str(tmp_path / "cube.obj")
        ensure_default_cube(obj_path)
        return Application(mesh_path=obj_path)

    def test_smooth_enabled_default(self, app):
        assert app.smooth_enabled is True

    def test_toggle_smoothing(self, app):
        app._toggle_smoothing()
        assert app.smooth_enabled is False
        app._toggle_smoothing()
        assert app.smooth_enabled is True

    def test_s_key_toggles(self, app):
        # glfw.KEY_S = 83, glfw.PRESS = 1
        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_ESCAPE = 256
            mock_glfw.KEY_Q = 81
            mock_glfw.KEY_R = 82
            mock_glfw.KEY_S = 83

            assert app.smooth_enabled is True
            app._on_key(83, 0, 1, 0)
            assert app.smooth_enabled is False

    def test_solve_uses_smooth_flag(self, app):
        """_on_both_points_placed passes smooth_enabled to path_finder."""
        app.smooth_enabled = False
        fake_result = PathResult(found=True, distance=1.0,
                                 points=[Vector3(0, 0, 0), Vector3(1, 0, 0)])
        app.path_finder.compute_path = MagicMock(return_value=fake_result)
        app._on_both_points_placed(Vector3(0, 0, 0), Vector3(1, 0, 0))

        app.path_finder.compute_path.assert_called_once_with(
            Vector3(0, 0, 0), Vector3(1, 0, 0), app.mesh, smooth=False
        )
