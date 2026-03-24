"""Tests for the pathfinding Strategy Pattern (Stage 1 refactor).

Covers:
    - PathfindingStrategy is abstract and cannot be instantiated
    - DijkstraStrategy implements the interface
    - DijkstraStrategy.name returns "Dijkstra"
    - DijkstraStrategy produces correct PathResult
    - PathFinder delegates to the active strategy
    - PathFinder.set_strategy swaps algorithms at runtime
    - PathResult includes algorithm_name
    - Backward compatibility: old PathSolver still works
"""

from __future__ import annotations

import pytest

from core.vector3 import Vector3
from mesh.mesh import Mesh
from pathfinding.strategy import PathfindingStrategy, PathResult
from pathfinding.dijkstra_strategy import DijkstraStrategy
from pathfinding.path_finder import PathFinder


# ── helpers ─────────────────────────────────────────────────

def _unit_cube_mesh() -> Mesh:
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
    return Mesh(
        vertices=[Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(1, 2, 0)],
        faces=[(0, 1, 2)],
    )


# ── Dummy strategy for testing swap ────────────────────────

class _DummyStrategy(PathfindingStrategy):
    """Minimal concrete strategy for testing."""

    @property
    def name(self) -> str:
        return "Dummy"

    def find_path(self, start, end, mesh, *, smooth=True):
        return PathResult(
            found=True,
            distance=42.0,
            points=[start, end],
            algorithm_name=self.name,
        )


# ═══════════════════════════════════════════════════════════
#  PathfindingStrategy (abstract)
# ═══════════════════════════════════════════════════════════

class TestPathfindingStrategyABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            PathfindingStrategy()  # type: ignore[abstract]

    def test_dummy_strategy_is_concrete(self) -> None:
        s = _DummyStrategy()
        assert s.name == "Dummy"


# ═══════════════════════════════════════════════════════════
#  PathResult
# ═══════════════════════════════════════════════════════════

class TestPathResultNew:
    def test_algorithm_name_field(self) -> None:
        r = PathResult(algorithm_name="Dijkstra")
        assert r.algorithm_name == "Dijkstra"

    def test_default_algorithm_name(self) -> None:
        r = PathResult()
        assert r.algorithm_name == ""


# ═══════════════════════════════════════════════════════════
#  DijkstraStrategy
# ═══════════════════════════════════════════════════════════

class TestDijkstraStrategy:
    def test_name(self) -> None:
        s = DijkstraStrategy()
        assert s.name == "Dijkstra"

    def test_find_path_on_cube(self) -> None:
        mesh = _unit_cube_mesh()
        s = DijkstraStrategy(bary_steps=1)
        start = Vector3(0, 0, 0.5)
        end = Vector3(0, 0, -0.5)
        result = s.find_path(start, end, mesh)
        assert result.found
        assert result.distance > 0
        assert len(result.points) >= 2
        assert result.algorithm_name == "Dijkstra"

    def test_find_path_no_smooth(self) -> None:
        mesh = _unit_cube_mesh()
        s = DijkstraStrategy(bary_steps=1)
        start = Vector3(0, 0, 0.5)
        end = Vector3(0, 0, -0.5)
        result = s.find_path(start, end, mesh, smooth=False)
        assert result.found
        assert not result.smoothed
        assert result.raw_points == []

    def test_find_path_smoothed(self) -> None:
        mesh = _unit_cube_mesh()
        s = DijkstraStrategy(bary_steps=1)
        start = Vector3(0, 0, 0.5)
        end = Vector3(0, 0, -0.5)
        result = s.find_path(start, end, mesh, smooth=True)
        assert result.found
        assert result.smoothed

    def test_same_point(self) -> None:
        mesh = _unit_cube_mesh()
        s = DijkstraStrategy(bary_steps=1)
        p = Vector3(0, 0, 0.5)
        result = s.find_path(p, p, mesh)
        assert result.found
        assert result.distance == 0.0

    def test_result_has_graph(self) -> None:
        mesh = _unit_cube_mesh()
        s = DijkstraStrategy(bary_steps=1)
        result = s.find_path(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh)
        assert result.graph is not None

    def test_num_samples_populated(self) -> None:
        mesh = _unit_cube_mesh()
        s = DijkstraStrategy(bary_steps=1)
        result = s.find_path(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh)
        assert result.num_samples > 0

    def test_cache_invalidation(self) -> None:
        s = DijkstraStrategy(bary_steps=1)
        mesh = _unit_cube_mesh()
        s.find_path(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh)
        s.invalidate_cache()
        # Second call should work after invalidation.
        result = s.find_path(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh)
        assert result.found


# ═══════════════════════════════════════════════════════════
#  PathFinder (context)
# ═══════════════════════════════════════════════════════════

class TestPathFinder:
    def test_initial_strategy(self) -> None:
        s = DijkstraStrategy()
        pf = PathFinder(s)
        assert pf.strategy is s
        assert pf.algorithm_name == "Dijkstra"

    def test_compute_path_delegates(self) -> None:
        mesh = _unit_cube_mesh()
        pf = PathFinder(DijkstraStrategy(bary_steps=1))
        result = pf.compute_path(
            Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh
        )
        assert result.found
        assert result.algorithm_name == "Dijkstra"

    def test_set_strategy_swaps(self) -> None:
        mesh = _unit_cube_mesh()
        pf = PathFinder(DijkstraStrategy(bary_steps=1))
        assert pf.algorithm_name == "Dijkstra"

        pf.set_strategy(_DummyStrategy())
        assert pf.algorithm_name == "Dummy"

        result = pf.compute_path(
            Vector3(0, 0, 0), Vector3(1, 0, 0), mesh
        )
        assert result.found
        assert result.distance == 42.0
        assert result.algorithm_name == "Dummy"

    def test_smooth_parameter_forwarded(self) -> None:
        mesh = _unit_cube_mesh()
        pf = PathFinder(DijkstraStrategy(bary_steps=1))
        r1 = pf.compute_path(
            Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh, smooth=True
        )
        r2 = pf.compute_path(
            Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh, smooth=False
        )
        assert r1.smoothed is True
        assert r2.smoothed is False


# ═══════════════════════════════════════════════════════════
#  Backward compatibility
# ═══════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_old_path_solver_still_works(self) -> None:
        """PathSolver (pre-refactor API) must still be importable and functional."""
        from pathfinding.path_solver import PathSolver, PathResult as OldPathResult

        mesh = _unit_cube_mesh()
        solver = PathSolver(mesh, bary_steps=1)
        result = solver.solve(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5))
        assert result.found
        assert result.distance > 0
