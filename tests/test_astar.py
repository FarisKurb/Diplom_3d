"""Tests for Stage 2 — A* algorithm and A* strategy.

Covers:
    - A* graph algorithm correctness
    - A* produces optimal paths (same distance as Dijkstra)
    - A* heuristic: Euclidean distance is admissible
    - A* with custom heuristic
    - AStarStrategy interface compliance
    - AStarStrategy integration with PathFinder
    - Comparison with Dijkstra on same graph
"""

from __future__ import annotations

import math
import pytest

from core.vector3 import Vector3
from graph.graph import Graph
from graph.astar import astar, astar_path_points, _euclidean_heuristic
from graph.dijkstra import dijkstra
from mesh.mesh import Mesh
from pathfinding.astar_strategy import AStarStrategy
from pathfinding.dijkstra_strategy import DijkstraStrategy
from pathfinding.path_finder import PathFinder
from pathfinding.strategy import PathResult


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


def _simple_graph() -> tuple[Graph, int, int]:
    """Build a small 4-node graph:
        0 --1-- 1 --1-- 3
        |               |
        +----- 2 ------+
              (3)
    Shortest 0→3 is via node 1 (cost 2), not via node 2 (cost 3+1=4 if connected).
    """
    g = Graph()
    n0 = g.add_node(Vector3(0, 0, 0))
    n1 = g.add_node(Vector3(1, 0, 0))
    n2 = g.add_node(Vector3(0, 1, 0))
    n3 = g.add_node(Vector3(2, 0, 0))
    g.add_edge(n0, n1, weight=1.0)
    g.add_edge(n1, n3, weight=1.0)
    g.add_edge(n0, n2, weight=3.0)
    g.add_edge(n2, n3, weight=3.0)
    return g, n0, n3


# ═══════════════════════════════════════════════════════════
#  A* graph algorithm
# ═══════════════════════════════════════════════════════════

class TestAStarAlgorithm:
    def test_simple_path(self) -> None:
        g, s, e = _simple_graph()
        result = astar(g, s, e)
        assert result is not None
        dist, path = result
        assert math.isclose(dist, 2.0)
        assert path == [0, 1, 3]

    def test_same_node(self) -> None:
        g, s, _ = _simple_graph()
        result = astar(g, s, s)
        assert result is not None
        dist, path = result
        assert dist == 0.0
        assert path == [s]

    def test_unreachable(self) -> None:
        g = Graph()
        n0 = g.add_node(Vector3(0, 0, 0))
        n1 = g.add_node(Vector3(1, 0, 0))
        # No edge between them.
        result = astar(g, n0, n1)
        assert result is None

    def test_invalid_node(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        result = astar(g, 0, 999)
        assert result is None

    def test_optimal_same_as_dijkstra(self) -> None:
        """A* with admissible heuristic must give same distance as Dijkstra."""
        g, s, e = _simple_graph()
        d_result = dijkstra(g, s, e)
        a_result = astar(g, s, e)
        assert d_result is not None
        assert a_result is not None
        assert math.isclose(d_result[0], a_result[0])

    def test_path_points_variant(self) -> None:
        g, s, e = _simple_graph()
        result = astar_path_points(g, s, e)
        assert result is not None
        dist, points = result
        assert math.isclose(dist, 2.0)
        assert len(points) == 3
        assert isinstance(points[0], Vector3)

    def test_path_points_unreachable(self) -> None:
        g = Graph()
        n0 = g.add_node(Vector3(0, 0, 0))
        n1 = g.add_node(Vector3(1, 0, 0))
        assert astar_path_points(g, n0, n1) is None


# ═══════════════════════════════════════════════════════════
#  Heuristic
# ═══════════════════════════════════════════════════════════

class TestEuclideanHeuristic:
    def test_same_point_zero(self) -> None:
        p = Vector3(1, 2, 3)
        assert _euclidean_heuristic(p, p) == 0.0

    def test_known_distance(self) -> None:
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 0)
        assert math.isclose(_euclidean_heuristic(a, b), 5.0)

    def test_admissibility_on_graph(self) -> None:
        """Heuristic must never overestimate actual shortest distance."""
        g, s, e = _simple_graph()
        d_result = dijkstra(g, s, e)
        assert d_result is not None
        actual_dist = d_result[0]
        h_val = _euclidean_heuristic(g.nodes[s], g.nodes[e])
        assert h_val <= actual_dist + 1e-9

    def test_custom_heuristic(self) -> None:
        """A* with zero heuristic should degenerate to Dijkstra."""
        g, s, e = _simple_graph()
        result = astar(g, s, e, heuristic=lambda a, b: 0.0)
        assert result is not None
        dist, _ = result
        assert math.isclose(dist, 2.0)


# ═══════════════════════════════════════════════════════════
#  AStarStrategy
# ═══════════════════════════════════════════════════════════

class TestAStarStrategy:
    def test_name(self) -> None:
        s = AStarStrategy()
        assert s.name == "A*"

    def test_find_path_on_cube(self) -> None:
        mesh = _unit_cube_mesh()
        s = AStarStrategy(bary_steps=1)
        start = Vector3(0, 0, 0.5)
        end = Vector3(0, 0, -0.5)
        result = s.find_path(start, end, mesh)
        assert result.found
        assert result.distance > 0
        assert len(result.points) >= 2
        assert result.algorithm_name == "A*"

    def test_find_path_no_smooth(self) -> None:
        mesh = _unit_cube_mesh()
        s = AStarStrategy(bary_steps=1)
        result = s.find_path(
            Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh, smooth=False
        )
        assert result.found
        assert not result.smoothed
        assert result.raw_points == []

    def test_find_path_smoothed(self) -> None:
        mesh = _unit_cube_mesh()
        s = AStarStrategy(bary_steps=1)
        result = s.find_path(
            Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh, smooth=True
        )
        assert result.found
        assert result.smoothed

    def test_same_point(self) -> None:
        mesh = _unit_cube_mesh()
        s = AStarStrategy(bary_steps=1)
        p = Vector3(0, 0, 0.5)
        result = s.find_path(p, p, mesh)
        assert result.found
        assert result.distance == 0.0

    def test_result_has_graph(self) -> None:
        mesh = _unit_cube_mesh()
        s = AStarStrategy(bary_steps=1)
        result = s.find_path(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh)
        assert result.graph is not None

    def test_cache_invalidation(self) -> None:
        s = AStarStrategy(bary_steps=1)
        mesh = _unit_cube_mesh()
        s.find_path(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh)
        s.invalidate_cache()
        result = s.find_path(Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh)
        assert result.found


# ═══════════════════════════════════════════════════════════
#  A* vs Dijkstra optimality
# ═══════════════════════════════════════════════════════════

class TestAStarVsDijkstra:
    def test_same_distance_on_cube(self) -> None:
        """A* must find a path of equal or lesser cost than Dijkstra."""
        mesh = _unit_cube_mesh()
        d = DijkstraStrategy(bary_steps=1)
        a = AStarStrategy(bary_steps=1)
        start = Vector3(0, 0, 0.5)
        end = Vector3(0, 0, -0.5)
        dr = d.find_path(start, end, mesh, smooth=False)
        ar = a.find_path(start, end, mesh, smooth=False)
        assert dr.found and ar.found
        assert math.isclose(dr.distance, ar.distance, rel_tol=1e-6)


# ═══════════════════════════════════════════════════════════
#  PathFinder integration
# ═══════════════════════════════════════════════════════════

class TestPathFinderWithAStar:
    def test_swap_to_astar(self) -> None:
        mesh = _unit_cube_mesh()
        pf = PathFinder(DijkstraStrategy(bary_steps=1))
        assert pf.algorithm_name == "Dijkstra"

        pf.set_strategy(AStarStrategy(bary_steps=1))
        assert pf.algorithm_name == "A*"

        result = pf.compute_path(
            Vector3(0, 0, 0.5), Vector3(0, 0, -0.5), mesh
        )
        assert result.found
        assert result.algorithm_name == "A*"
