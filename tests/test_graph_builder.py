"""Unit tests for graph.graph_builder.GraphBuilder."""

import math
from typing import List

from core.vector3 import Vector3
from geometry.triangle import Triangle
from graph.graph_builder import GraphBuilder
from graph.dijkstra import dijkstra_path_points
from mesh.mesh import Mesh


def _always_visible(a: Vector3, b: Vector3, tris: List[Triangle]) -> bool:
    """Visibility function that always returns True (no obstacles)."""
    return True


def _never_visible(a: Vector3, b: Vector3, tris: List[Triangle]) -> bool:
    """Visibility function that always returns False."""
    return False


def _simple_mesh() -> Mesh:
    """Single triangle mesh."""
    verts = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)]
    faces = [(0, 1, 2)]
    return Mesh(verts, faces)


class TestGraphBuilder:
    def test_builds_graph_with_mesh_vertices(self) -> None:
        mesh = _simple_mesh()
        builder = GraphBuilder(mesh, _always_visible)
        graph, sid, eid = builder.build(Vector3(2, 0, 0), Vector3(3, 0, 0))
        # 3 mesh vertices + 2 query = 5 nodes
        assert graph.num_nodes == 5

    def test_all_visible_fully_connected(self) -> None:
        mesh = _simple_mesh()
        builder = GraphBuilder(mesh, _always_visible)
        graph, sid, eid = builder.build(Vector3(2, 0, 0), Vector3(3, 0, 0))
        # 5 nodes fully connected = 5*4/2 = 10 edges
        assert graph.num_edges == 10

    def test_none_visible_no_edges(self) -> None:
        mesh = _simple_mesh()
        builder = GraphBuilder(mesh, _never_visible)
        graph, sid, eid = builder.build(Vector3(2, 0, 0), Vector3(3, 0, 0))
        assert graph.num_edges == 0

    def test_with_sample_points(self) -> None:
        mesh = _simple_mesh()
        samples = [Vector3(0.25, 0.25, 0)]
        builder = GraphBuilder(mesh, _always_visible, sample_points=samples)
        graph, sid, eid = builder.build(Vector3(2, 0, 0), Vector3(3, 0, 0))
        # 3 + 1 + 2 = 6 nodes
        assert graph.num_nodes == 6

    def test_path_through_graph(self) -> None:
        mesh = _simple_mesh()
        builder = GraphBuilder(mesh, _always_visible)
        graph, sid, eid = builder.build(Vector3(-1, 0, 0), Vector3(2, 0, 0))
        result = dijkstra_path_points(graph, sid, eid)
        assert result is not None
        dist, pts = result
        # Direct route is shortest: distance = 3
        assert math.isclose(dist, 3.0, abs_tol=1e-6)
        assert len(pts) == 2  # start → end directly
