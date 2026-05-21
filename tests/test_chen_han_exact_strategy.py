"""Tests for Chen-Han Exact strategy."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

from core.vector3 import Vector3
from geometry.intersection import segment_intersects_mesh
from main import AVAILABLE_STRATEGIES, Application
from mesh.obj_loader import ensure_default_cube, load_obj
from pathfinding.astar_strategy import AStarStrategy
from pathfinding.chen_han_exact_strategy import ChenHanExactStrategy
from pathfinding.dijkstra_strategy import DijkstraStrategy
from pathfinding.geodesic_approx_strategy import GeodesicApproxStrategy
from pathfinding.strategy import PathResult
from pathfinding.visibility_graph_strategy import VisibilityGraphStrategy


def _cube_mesh(tmp_path):
    obj_path = str(tmp_path / "cube.obj")
    ensure_default_cube(obj_path)
    return load_obj(obj_path)


def test_directly_visible_points_return_two_point_path(tmp_path):
    mesh = _cube_mesh(tmp_path)
    strategy = ChenHanExactStrategy()
    start = Vector3(-2.0, 2.0, 0.0)
    end = Vector3(2.0, 2.0, 0.0)

    result = strategy.find_path(start, end, mesh)

    assert result.found is True
    assert result.points == [start, end]
    assert result.graph is None
    assert result.distance == start.distance_to(end)


def test_cube_between_points_path_does_not_enter_cube(tmp_path):
    mesh = _cube_mesh(tmp_path)
    strategy = ChenHanExactStrategy()
    start = Vector3(-2.0, 0.0, 0.0)
    end = Vector3(2.0, 0.0, 0.0)

    result = strategy.find_path(start, end, mesh)

    assert result.found is True
    assert result.algorithm_name == "Chen-Han Exact"
    assert len(result.points) >= 2
    for a, b in zip(result.points, result.points[1:]):
        assert not segment_intersects_mesh(a, b, mesh.triangles)


def test_blocked_case_uses_window_predecessor_chain(tmp_path):
    mesh = _cube_mesh(tmp_path)
    strategy = ChenHanExactStrategy()
    start = Vector3(-2.0, 0.0, 0.0)
    end = Vector3(2.0, 0.0, 0.0)
    vertices, faces = strategy._build_convex_hull([*mesh.vertices, start, end])
    start_id = strategy._find_vertex(vertices, start)
    end_id = strategy._find_vertex(vertices, end)

    assert start_id is not None
    assert end_id is not None
    candidate = strategy._chen_han_shortest_path(vertices, faces, start_id, end_id)

    assert candidate is not None
    assert candidate.window.edge is not None
    assert candidate.window.predecessor is not None
    assert len(candidate.window.sequence) >= 2


def test_surface_points_inside_cube_faces_keep_hull_manifold(tmp_path):
    mesh = _cube_mesh(tmp_path)
    strategy = ChenHanExactStrategy()
    start = Vector3(0.45479940882771563, 0.5, 0.47467971577963786)
    end = Vector3(0.5000000000000004, -0.4782492757849155, -0.4790163908772156)

    result = strategy.find_path(start, end, mesh)

    assert result.found is True
    assert result.points[0].approx_equal(start)
    assert result.points[-1].approx_equal(end)
    for a, b in zip(result.points, result.points[1:]):
        assert not segment_intersects_mesh(a, b, mesh.triangles)

    vertices, faces = strategy._build_convex_hull([*mesh.vertices, start, end])
    edge_to_faces = defaultdict(list)
    for face_id, face in enumerate(faces):
        for edge in strategy._face_edges(face):
            edge_to_faces[tuple(sorted(edge))].append(face_id)

    assert all(len(incident) == 2 for incident in edge_to_faces.values())


def test_chen_han_exact_is_default_strategy(tmp_path):
    obj_path = str(tmp_path / "cube.obj")
    ensure_default_cube(obj_path)
    app = Application(mesh_path=obj_path)

    assert AVAILABLE_STRATEGIES[0] is ChenHanExactStrategy
    assert app.path_finder.algorithm_name == "Chen-Han Exact"


def test_legacy_strategies_remain_available():
    assert DijkstraStrategy in AVAILABLE_STRATEGIES
    assert AStarStrategy in AVAILABLE_STRATEGIES
    assert VisibilityGraphStrategy in AVAILABLE_STRATEGIES
    assert GeodesicApproxStrategy in AVAILABLE_STRATEGIES


def test_application_accepts_result_without_graph(tmp_path):
    obj_path = str(tmp_path / "cube.obj")
    ensure_default_cube(obj_path)
    app = Application(mesh_path=obj_path)
    result = PathResult(
        found=True,
        distance=1.0,
        points=[Vector3(0, 0, 0), Vector3(1, 0, 0)],
        graph=None,
        algorithm_name="Chen-Han Exact",
    )
    app.path_finder.compute_path = MagicMock(return_value=result)

    app._compute_and_display(Vector3(0, 0, 0), Vector3(1, 0, 0))

    assert app.path_renderer.sample_nodes == []
    assert app._last_result is result
