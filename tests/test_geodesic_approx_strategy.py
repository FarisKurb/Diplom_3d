"""Tests for Stage 4 — Geodesic Approximation strategy.

Covers:
    - GeodesicApproxStrategy interface compliance
    - Dense edge subdivision produces more nodes than visibility graph
    - Iterative refinement mechanics
    - Path found around cube obstacle
    - Smoothing on/off
    - Same-point trivial case
    - Comparison with Visibility Graph (should find path of similar or better quality)
    - PathFinder integration with strategy swap
    - Internal helpers (_subdivide_edges, _vertices_near_path, etc.)
"""

from __future__ import annotations

import math
import pytest

from core.vector3 import Vector3
from mesh.mesh import Mesh
from mesh.mesh_topology import extract_edges
from pathfinding.geodesic_approx_strategy import (
    GeodesicApproxStrategy,
    _merge_samples,
)
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

class TestGeodesicApproxInterface:
    def test_name(self) -> None:
        s = GeodesicApproxStrategy()
        assert s.name == "Geodesic Approx"

    def test_is_pathfinding_strategy(self) -> None:
        from pathfinding.strategy import PathfindingStrategy
        assert isinstance(GeodesicApproxStrategy(), PathfindingStrategy)

    def test_default_parameters(self) -> None:
        s = GeodesicApproxStrategy()
        assert s._edge_divisions == 4
        assert s._refine_passes == 1

    def test_custom_parameters(self) -> None:
        s = GeodesicApproxStrategy(edge_divisions=6, refine_passes=3)
        assert s._edge_divisions == 6
        assert s._refine_passes == 3


# ═══════════════════════════════════════════════════════════
#  Edge subdivision
# ═══════════════════════════════════════════════════════════

class TestEdgeSubdivision:
    def test_subdivide_single_edge(self) -> None:
        """4 divisions should produce 4 interior points per edge."""
        mesh = _unit_cube_mesh()
        edges = [(0, 1)]  # one edge
        pts = GeodesicApproxStrategy._subdivide_edges(mesh, edges, 4)
        assert len(pts) == 4

    def test_subdivide_points_lie_on_segment(self) -> None:
        mesh = _unit_cube_mesh()
        a, b = mesh.vertices[0], mesh.vertices[1]
        pts = GeodesicApproxStrategy._subdivide_edges(mesh, [(0, 1)], 4)
        for p in pts:
            # p should lie between a and b — check parameterically
            ab = b - a
            ap = p - a
            t = ap.dot(ab) / ab.dot(ab)
            assert 0.0 < t < 1.0

    def test_subdivide_dedup(self) -> None:
        """Shared edges should not produce duplicate points."""
        mesh = _unit_cube_mesh()
        all_edges = extract_edges(mesh)
        pts = GeodesicApproxStrategy._subdivide_edges(mesh, all_edges, 2)
        coords = [(round(p.x, 9), round(p.y, 9), round(p.z, 9)) for p in pts]
        assert len(coords) == len(set(coords))

    def test_more_divisions_more_points(self) -> None:
        mesh = _unit_cube_mesh()
        edges = extract_edges(mesh)
        pts2 = GeodesicApproxStrategy._subdivide_edges(mesh, edges, 2)
        pts4 = GeodesicApproxStrategy._subdivide_edges(mesh, edges, 4)
        assert len(pts4) > len(pts2)


# ═══════════════════════════════════════════════════════════
#  Graph construction
# ═══════════════════════════════════════════════════════════

class TestGeodesicGraphConstruction:
    def test_graph_denser_than_visibility_graph(self) -> None:
        """Geodesic graph with 4 divisions should have more nodes than
        visibility graph (which uses 1 midpoint per edge)."""
        mesh = _unit_cube_mesh()
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)

        vg = VisibilityGraphStrategy()
        vg_graph, _, _ = vg._build_graph(start, end, mesh)

        gs = GeodesicApproxStrategy(edge_divisions=4, refine_passes=0)
        edges = extract_edges(mesh)
        samples = gs._subdivide_edges(mesh, edges, 4)
        gg, _, _ = gs._build_graph(start, end, mesh, samples)

        assert gg.num_nodes > vg_graph.num_nodes

    def test_start_end_in_graph(self) -> None:
        mesh = _unit_cube_mesh()
        start = Vector3(5, 5, 5)
        end = Vector3(-5, -5, -5)
        edges = extract_edges(mesh)
        samples = GeodesicApproxStrategy._subdivide_edges(mesh, edges, 2)
        graph, sid, eid = GeodesicApproxStrategy._build_graph(
            start, end, mesh, samples,
        )
        assert graph.nodes[sid].approx_equal(start)
        assert graph.nodes[eid].approx_equal(end)

    def test_num_samples_reported(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        result = s.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh)
        assert result.num_samples > 0


# ═══════════════════════════════════════════════════════════
#  Refinement helpers
# ═══════════════════════════════════════════════════════════

class TestRefinementHelpers:
    def test_vertices_near_path(self) -> None:
        mesh = _unit_cube_mesh()
        from mesh.mesh_topology import build_adjacency
        adj = build_adjacency(mesh)
        path = [Vector3(0.5, 0, 0)]  # near vertex 1 and 2
        near = GeodesicApproxStrategy._vertices_near_path(path, mesh, adj)
        # Should contain at least the closest vertex
        assert len(near) >= 1

    def test_edges_near_vertices(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        near_verts = {1, 2}
        result = GeodesicApproxStrategy._edges_near_vertices(near_verts, edges)
        # (0,1), (1,2), (2,3) all touch vertex 1 or 2
        assert (0, 1) in result
        assert (1, 2) in result
        assert (2, 3) in result
        assert (3, 4) not in result

    def test_merge_samples_dedup(self) -> None:
        a = [Vector3(1, 0, 0), Vector3(0, 1, 0)]
        b = [Vector3(0, 1, 0), Vector3(0, 0, 1)]  # one overlap
        merged = _merge_samples(a, b)
        assert len(merged) == 3

    def test_merge_samples_empty(self) -> None:
        a = [Vector3(1, 0, 0)]
        merged = _merge_samples(a, [])
        assert len(merged) == 1


# ═══════════════════════════════════════════════════════════
#  Path finding
# ═══════════════════════════════════════════════════════════

class TestGeodesicApproxPathfinding:
    def test_find_path_around_cube(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        result = s.find_path(start, end, mesh)
        assert result.found
        assert result.distance > 0
        assert len(result.points) >= 2
        assert result.algorithm_name == "Geodesic Approx"

    def test_find_path_with_refinement(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=1)
        result = s.find_path(
            Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=False,
        )
        assert result.found
        assert result.distance > 0

    def test_refinement_does_not_worsen_path(self) -> None:
        """With refinement the path should be equal or shorter."""
        mesh = _unit_cube_mesh()
        s0 = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        s1 = GeodesicApproxStrategy(edge_divisions=2, refine_passes=1)
        r0 = s0.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=False)
        r1 = s1.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=False)
        assert r0.found and r1.found
        # Refined path should be no longer than coarse
        assert r1.distance <= r0.distance + 1e-9

    def test_find_path_no_smooth(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        result = s.find_path(
            Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=False,
        )
        assert result.found
        assert not result.smoothed
        assert result.raw_points == []

    def test_find_path_smoothed(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        result = s.find_path(
            Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=True,
        )
        assert result.found
        assert result.smoothed
        assert len(result.raw_points) >= 2

    def test_same_point(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        p = Vector3(0, 0, 2)
        result = s.find_path(p, p, mesh)
        assert result.found
        assert result.distance == 0.0

    def test_result_has_graph(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        result = s.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh)
        assert result.graph is not None

    def test_direct_line_of_sight(self) -> None:
        mesh = _unit_cube_mesh()
        s = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        start = Vector3(0, 0, 2)
        end = Vector3(0.1, 0, 2)
        result = s.find_path(start, end, mesh, smooth=False)
        assert result.found
        assert len(result.points) == 2
        assert math.isclose(result.distance, start.distance_to(end), rel_tol=1e-6)


# ═══════════════════════════════════════════════════════════
#  Comparison with other strategies
# ═══════════════════════════════════════════════════════════

class TestGeodesicVsOtherStrategies:
    def test_vs_visibility_graph_both_find_path(self) -> None:
        mesh = _unit_cube_mesh()
        vg = VisibilityGraphStrategy()
        ga = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        vr = vg.find_path(start, end, mesh, smooth=False)
        gr = ga.find_path(start, end, mesh, smooth=False)
        assert vr.found and gr.found
        assert vr.distance > 0 and gr.distance > 0

    def test_denser_graph_equal_or_better(self) -> None:
        """More edge divisions should yield equal or shorter paths."""
        mesh = _unit_cube_mesh()
        s2 = GeodesicApproxStrategy(edge_divisions=2, refine_passes=0)
        s4 = GeodesicApproxStrategy(edge_divisions=4, refine_passes=0)
        r2 = s2.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=False)
        r4 = s4.find_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh, smooth=False)
        assert r2.found and r4.found
        assert r4.distance <= r2.distance + 1e-9


# ═══════════════════════════════════════════════════════════
#  PathFinder integration
# ═══════════════════════════════════════════════════════════

class TestPathFinderWithGeodesic:
    def test_swap_to_geodesic(self) -> None:
        mesh = _unit_cube_mesh()
        pf = PathFinder(DijkstraStrategy(bary_steps=1))
        assert pf.algorithm_name == "Dijkstra"

        pf.set_strategy(GeodesicApproxStrategy(edge_divisions=2, refine_passes=0))
        assert pf.algorithm_name == "Geodesic Approx"

        result = pf.compute_path(Vector3(0, 0, 2), Vector3(0, 0, -2), mesh)
        assert result.found
        assert result.algorithm_name == "Geodesic Approx"
