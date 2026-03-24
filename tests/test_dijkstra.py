"""Unit tests for graph.dijkstra."""

import math
import pytest
from core.vector3 import Vector3
from graph.graph import Graph
from graph.dijkstra import dijkstra, dijkstra_path_points


def _line_graph() -> Graph:
    """0 -- 1 -- 2  (each 1 unit apart along X)."""
    g = Graph()
    g.add_node(Vector3(0, 0, 0))
    g.add_node(Vector3(1, 0, 0))
    g.add_node(Vector3(2, 0, 0))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


def _triangle_graph() -> Graph:
    """
    0 --- 1
     \\   |
      \\  |
       \\ |
        2
    0→1 = 1, 1→2 = 1, 0→2 = 10
    """
    g = Graph()
    g.add_node(Vector3(0, 0, 0))
    g.add_node(Vector3(1, 0, 0))
    g.add_node(Vector3(1, 1, 0))
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(0, 2, weight=10.0)
    return g


class TestDijkstra:
    def test_same_node(self) -> None:
        g = _line_graph()
        result = dijkstra(g, 0, 0)
        assert result is not None
        dist, path = result
        assert dist == 0.0
        assert path == [0]

    def test_direct_neighbour(self) -> None:
        g = _line_graph()
        result = dijkstra(g, 0, 1)
        assert result is not None
        dist, path = result
        assert math.isclose(dist, 1.0)
        assert path == [0, 1]

    def test_two_hops(self) -> None:
        g = _line_graph()
        result = dijkstra(g, 0, 2)
        assert result is not None
        dist, path = result
        assert math.isclose(dist, 2.0)
        assert path == [0, 1, 2]

    def test_shortest_via_intermediate(self) -> None:
        g = _triangle_graph()
        result = dijkstra(g, 0, 2)
        assert result is not None
        dist, path = result
        assert math.isclose(dist, 2.0)
        assert path == [0, 1, 2]

    def test_unreachable(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_node(Vector3(5, 5, 5))
        result = dijkstra(g, 0, 1)
        assert result is None

    def test_invalid_nodes(self) -> None:
        g = Graph()
        assert dijkstra(g, 0, 1) is None

    def test_reverse_direction(self) -> None:
        g = _line_graph()
        result = dijkstra(g, 2, 0)
        assert result is not None
        dist, path = result
        assert math.isclose(dist, 2.0)
        assert path == [2, 1, 0]


class TestDijkstraPathPoints:
    def test_returns_points(self) -> None:
        g = _line_graph()
        result = dijkstra_path_points(g, 0, 2)
        assert result is not None
        dist, pts = result
        assert math.isclose(dist, 2.0)
        assert len(pts) == 3
        assert pts[0].approx_equal(Vector3(0, 0, 0))
        assert pts[2].approx_equal(Vector3(2, 0, 0))

    def test_unreachable(self) -> None:
        g = Graph()
        g.add_node(Vector3(0, 0, 0))
        g.add_node(Vector3(1, 0, 0))
        assert dijkstra_path_points(g, 0, 1) is None
