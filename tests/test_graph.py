"""Unit tests for graph.graph.Graph."""

import math
from core.vector3 import Vector3
from graph.graph import Graph


class TestGraphNodes:
    def test_add_node_auto_id(self) -> None:
        g = Graph()
        id0 = g.add_node(Vector3(0, 0, 0))
        id1 = g.add_node(Vector3(1, 0, 0))
        assert id0 == 0
        assert id1 == 1
        assert g.num_nodes == 2

    def test_add_node_explicit_id(self) -> None:
        g = Graph()
        nid = g.add_node(Vector3(0, 0, 0), node_id=10)
        assert nid == 10
        assert g.has_node(10)

    def test_has_node(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        assert g.has_node(0)
        assert not g.has_node(99)

    def test_node_ids(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_node(Vector3(1, 0, 0))
        assert g.node_ids() == {0, 1}

    def test_clear(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.clear()
        assert g.num_nodes == 0
        assert g.num_edges == 0


class TestGraphEdges:
    def test_add_edge_auto_weight(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_node(Vector3(3, 4, 0))
        g.add_edge(0, 1)
        assert g.num_edges == 1
        nbrs = g.neighbours(0)
        assert len(nbrs) == 1
        assert nbrs[0][0] == 1
        assert math.isclose(nbrs[0][1], 5.0)

    def test_add_edge_explicit_weight(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_node(Vector3(1, 0, 0))
        g.add_edge(0, 1, weight=42.0)
        assert math.isclose(g.neighbours(0)[0][1], 42.0)

    def test_no_self_loop(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_edge(0, 0)
        assert g.num_edges == 0

    def test_no_duplicate_edge(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_node(Vector3(1, 0, 0))
        g.add_edge(0, 1)
        g.add_edge(0, 1)
        assert g.num_edges == 1

    def test_bidirectional(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_node(Vector3(1, 0, 0))
        g.add_edge(0, 1)
        assert len(g.neighbours(0)) == 1
        assert len(g.neighbours(1)) == 1

    def test_repr(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        assert "Graph" in repr(g)
