"""Tests for Stage 3 — Visibility Graph strategy.

Covers:
    - VisibilityGraphStrategy interface compliance
    - Graph construction uses vertices + edge midpoints (no face sampling)
    - Path found around cube obstacle
    - Smoothing on/off
    - Same-point trivial case
    - Comparison with Dijkstra/A* (finds a valid path)
    - PathFinder integration with strategy swap
"""

from __future__ import annotations

import math
import pytest

from core.vector3 import Vector3
from mesh.mesh import Mesh
from mesh.mesh_topology import extract_edges
from pathfinding.visibility_graph_strategy import VisibilityGraphStrategy
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


# ═══════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════

class TestVisibilityGraphInterface:
    def test_name(self) -> None:
        s = VisibilityGraphStrategy()
        assert s.name == "Visibility Graph"

    def test_is_pathfinding_strategy(self) -> None:
        from pathfinding.strategy import PathfindingStrategy
        assert isinstance(VisibilityGraphStrategy(), PathfindingStrategy)


# ═══════════════════════════════════════════════════════════
#  Graph construction
# ═══════════════════════════════════════════════════════════

class TestVisibilityGraphConstruction:
    def test_graph_has_vertices_and_edge_midpoints(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        graph, sid, eid = s._build_graph(start, end, mesh)
        num_edges = len(extract_edges(mesh))
        num_verts = mesh.num_vertices
        # Graph nodes = vertices + edge midpoints + start + end
        expected = num_verts + num_edges + 2
        assert graph.num_nodes == expected

    def test_num_samples_is_zero(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        result = s.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh)
        assert result.num_samples == 0

    def test_start_end_in_graph(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        start = Vector3(10, 10, 10)
        end = Vector3(-10, -10, -10)
        graph, sid, eid = s._build_graph(start, end, mesh)
        assert graph.nodes[sid].approx_equal(start)
        assert graph.nodes[eid].approx_equal(end)


# ═══════════════════════════════════════════════════════════
#  Path finding
# ═══════════════════════════════════════════════════════════

class TestVisibilityGraphPathfinding:
    def test_find_path_around_cube(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        result = s.find_path(start, end, mesh)
        assert result.found
        assert result.distance > 0
        assert len(result.points) >= 2
        assert result.algorithm_name == "Visibility Graph"

    def test_find_path_no_smooth(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        result = s.find_path(
            Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=False,
        )
        assert result.found
        assert not result.smoothed
        assert result.raw_points == []

    def test_find_path_smoothed(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        result = s.find_path(
            Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=True,
        )
        assert result.found
        assert result.smoothed

    def test_same_point(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        p = Vector3(0, 0, 2)
        result = s.find_path(p, p, mesh)
        assert result.found
        assert result.distance == 0.0

    def test_result_has_graph(self) -> None:
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        result = s.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh)
        assert result.graph is not None

    def test_direct_line_of_sight(self) -> None:
        """When start and end can see each other, path is a straight line."""
        mesh = _unit_cube_mesh()
        s = VisibilityGraphStrategy()
        # Both points on the same side of the cube.
        start = Vector3(0, 0, 2)
        end = Vector3(0.1, 0, 2)
        result = s.find_path(start, end, mesh, smooth=False)
        assert result.found
        assert len(result.points) == 2
        assert math.isclose(result.distance, start.distance_to(end), rel_tol=1e-6)


# ═══════════════════════════════════════════════════════════
#  Comparison with Dijkstra
# ═══════════════════════════════════════════════════════════

class TestVisGraphVsDijkstra:
    def test_both_find_valid_path(self) -> None:
        mesh = _unit_cube_mesh()
        d = DijkstraStrategy(bary_steps=1)
        v = VisibilityGraphStrategy()
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        dr = d.find_path(start, end, mesh, smooth=False)
        vr = v.find_path(start, end, mesh, smooth=False)
        assert dr.found and vr.found
        # Both should find reasonable paths (not necessarily identical
        # because they use different node sets).
        assert dr.distance > 0
        assert vr.distance > 0


# ═══════════════════════════════════════════════════════════
#  PathFinder integration
# ═══════════════════════════════════════════════════════════

class TestPathFinderWithVisGraph:
    def test_swap_to_visibility_graph(self) -> None:
        mesh = _unit_cube_mesh()
        pf = PathFinder(DijkstraStrategy(bary_steps=1))
        assert pf.algorithm_name == "Dijkstra"

        pf.set_strategy(VisibilityGraphStrategy())
        assert pf.algorithm_name == "Visibility Graph"

        result = pf.compute_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh)
        assert result.found
        assert result.algorithm_name == "Visibility Graph"
