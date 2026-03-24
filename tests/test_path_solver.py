"""Tests for Stage 9 — PathSolver (shortest-path computation).

Covers:
    - PathResult dataclass
    - PathSolver with a cube mesh (paths around obstacle)
    - Trivial same-point case
    - Direct line-of-sight path
    - Sample caching
    - PathSolver with non-default bary_steps
"""

from __future__ import annotations

import pytest

from core.vector3 import Vector3
from mesh.mesh import Mesh
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
#  PathResult
# ═══════════════════════════════════════════════════════════

class TestPathResult:
    def test_default(self) -> None:
        r = PathResult()
        assert r.found is False
        assert r.distance == 0.0
        assert r.points == []
        assert r.graph is None
        assert r.num_samples == 0

    def test_custom(self) -> None:
        r = PathResult(found=True, distance=3.14, points=[Vector3.zero()])
        assert r.found is True
        assert abs(r.distance - 3.14) < 1e-9
        assert len(r.points) == 1


# ═══════════════════════════════════════════════════════════
#  PathSolver — basic behaviour
# ═══════════════════════════════════════════════════════════

class TestPathSolverBasic:
    def test_solve_returns_result(self) -> None:
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=1)
        start = Vector3(0, 0, 0.5)   # on front face
        end = Vector3(0, 0, -0.5)    # on back face
        result = solver.solve(start, end)
        assert isinstance(result, PathResult)
        assert result.graph is not None

    def test_same_point(self) -> None:
        """Start == End should yield a trivial zero-length path."""
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=1)
        p = Vector3(0, 0.5, 0)  # on top face
        result = solver.solve(p, p)
        assert result.found is True
        assert result.distance < 1e-6
        assert len(result.points) >= 1


class TestPathSolverLineOfSight:
    def test_visible_endpoints(self) -> None:
        """Two points outside the cube that can see each other should get
        a short (near-direct) path.
        """
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=1)
        # Both on the same face (+Z)
        start = Vector3(-0.2, 0, 0.5)
        end = Vector3(0.2, 0, 0.5)
        result = solver.solve(start, end)
        assert result.found is True
        assert result.distance < 1.0
        assert len(result.points) >= 2
        # First and last points should match start and end.
        assert result.points[0].approx_equal(start, 1e-6)
        assert result.points[-1].approx_equal(end, 1e-6)


class TestPathSolverAroundObstacle:
    def test_path_around_cube(self) -> None:
        """Points on opposite faces should find a path around the cube.
        The path distance must be greater than the Euclidean distance
        (which would go through the interior).
        """
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        start = Vector3(0, 0, 0.5)    # front centre
        end = Vector3(0, 0, -0.5)     # back centre
        result = solver.solve(start, end)
        assert result.found is True
        # Euclidean would be 1.0; surface path must be longer.
        assert result.distance > 1.0
        assert len(result.points) >= 3

    def test_path_length_reasonable(self) -> None:
        """Path around a unit cube between opposite faces should not
        exceed a generous upper bound.
        """
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        start = Vector3(0, 0, 0.5)
        end = Vector3(0, 0, -0.5)
        result = solver.solve(start, end)
        assert result.found is True
        # A surface path over the top of a unit cube is about 2.0.
        # With sampling imprecision, allow up to 3.0.
        assert result.distance < 3.0


# ═══════════════════════════════════════════════════════════
#  PathSolver — caching
# ═══════════════════════════════════════════════════════════

class TestPathSolverCaching:
    def test_samples_cached(self) -> None:
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        s1 = solver.sample_points
        s2 = solver.sample_points
        assert s1 is s2  # same object, not recomputed

    def test_invalidate_cache(self) -> None:
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        s1 = solver.sample_points
        solver.invalidate_cache()
        s2 = solver.sample_points
        assert s1 is not s2

    def test_num_samples_reported(self) -> None:
        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        result = solver.solve(Vector3(0, 0.5, 0), Vector3(0, -0.5, 0))
        assert result.num_samples > 0


class TestPathSolverConfig:
    def test_bary_steps_affects_samples(self) -> None:
        mesh = _unit_cube_mesh()
        solver_low = PathSolver(mesh, bary_steps=1)
        solver_high = PathSolver(mesh, bary_steps=3)
        assert len(solver_high.sample_points) > len(solver_low.sample_points)

    def test_simple_mesh(self) -> None:
        """Solver works on a minimal single-triangle mesh."""
        mesh = _single_triangle_mesh()
        solver = PathSolver(mesh, bary_steps=2)
        start = Vector3(0.5, 0.5, 0)
        end = Vector3(1.5, 0.5, 0)
        result = solver.solve(start, end)
        # Both points are in the plane of the triangle; path should exist.
        assert result.found is True
        assert result.distance > 0
