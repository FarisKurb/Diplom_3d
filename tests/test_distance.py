"""Unit tests for geometry.distance."""

import math
from core.vector3 import Vector3
from geometry.triangle import Triangle
from geometry.distance import point_to_triangle_distance, closest_point_on_triangle


def _xy_tri() -> Triangle:
    return Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0))


class TestPointToTriangleDistance:
    def test_point_above_interior(self) -> None:
        d = point_to_triangle_distance(Vector3(0.25, 0.25, 1), _xy_tri())
        assert math.isclose(d, 1.0, abs_tol=1e-6)

    def test_point_on_triangle(self) -> None:
        d = point_to_triangle_distance(Vector3(0.25, 0.25, 0), _xy_tri())
        assert math.isclose(d, 0.0, abs_tol=1e-6)

    def test_point_nearest_to_edge(self) -> None:
        d = point_to_triangle_distance(Vector3(0.5, -1, 0), _xy_tri())
        assert math.isclose(d, 1.0, abs_tol=1e-6)

    def test_point_nearest_to_vertex(self) -> None:
        d = point_to_triangle_distance(Vector3(-1, -1, 0), _xy_tri())
        expected = Vector3(-1, -1, 0).distance_to(Vector3(0, 0, 0))
        assert math.isclose(d, expected, abs_tol=1e-6)

    def test_degenerate_triangle(self) -> None:
        tri = Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0))
        d = point_to_triangle_distance(Vector3(1, 1, 0), tri)
        assert math.isclose(d, 1.0, abs_tol=1e-6)


class TestClosestPointOnTriangle:
    def test_projection_inside(self) -> None:
        cp = closest_point_on_triangle(Vector3(0.25, 0.25, 5), _xy_tri())
        assert cp.approx_equal(Vector3(0.25, 0.25, 0))

    def test_closest_on_edge(self) -> None:
        cp = closest_point_on_triangle(Vector3(0.5, -1, 0), _xy_tri())
        # Nearest point on bottom edge y=0.
        assert math.isclose(cp.y, 0.0, abs_tol=1e-6)

    def test_closest_at_vertex(self) -> None:
        cp = closest_point_on_triangle(Vector3(-1, -1, 0), _xy_tri())
        assert cp.approx_equal(Vector3(0, 0, 0), eps=1e-6)

    def test_degenerate(self) -> None:
        tri = Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0))
        cp = closest_point_on_triangle(Vector3(1, 1, 0), tri)
        assert math.isclose(cp.y, 0.0, abs_tol=1e-6)
