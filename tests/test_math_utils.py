"""Unit tests for core.math_utils."""

import math
import pytest
from core.vector3 import Vector3
from core.math_utils import (
    clamp,
    point_to_segment_distance,
    closest_point_on_segment,
    segments_closest_distance,
    barycentric_coordinates,
    triangle_area,
    triangle_normal,
)


# ── clamp ───────────────────────────────────────────────────


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below(self) -> None:
        assert clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above(self) -> None:
        assert clamp(2.0, 0.0, 1.0) == 1.0


# ── point_to_segment_distance ──────────────────────────────


class TestPointToSegmentDistance:
    def test_projection_onto_segment(self) -> None:
        d = point_to_segment_distance(
            Vector3(1, 1, 0), Vector3(0, 0, 0), Vector3(2, 0, 0)
        )
        assert math.isclose(d, 1.0)

    def test_closest_to_start(self) -> None:
        d = point_to_segment_distance(
            Vector3(-1, 0, 0), Vector3(0, 0, 0), Vector3(2, 0, 0)
        )
        assert math.isclose(d, 1.0)

    def test_closest_to_end(self) -> None:
        d = point_to_segment_distance(
            Vector3(3, 0, 0), Vector3(0, 0, 0), Vector3(2, 0, 0)
        )
        assert math.isclose(d, 1.0)

    def test_degenerate_segment(self) -> None:
        d = point_to_segment_distance(
            Vector3(1, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0)
        )
        assert math.isclose(d, 1.0)


# ── closest_point_on_segment ───────────────────────────────


class TestClosestPointOnSegment:
    def test_middle_projection(self) -> None:
        p = closest_point_on_segment(
            Vector3(1, 1, 0), Vector3(0, 0, 0), Vector3(2, 0, 0)
        )
        assert p.approx_equal(Vector3(1, 0, 0))

    def test_clamped_to_start(self) -> None:
        p = closest_point_on_segment(
            Vector3(-5, 0, 0), Vector3(0, 0, 0), Vector3(2, 0, 0)
        )
        assert p.approx_equal(Vector3(0, 0, 0))


# ── segments_closest_distance ──────────────────────────────


class TestSegmentsClosestDistance:
    def test_parallel_segments(self) -> None:
        d = segments_closest_distance(
            Vector3(0, 0, 0), Vector3(1, 0, 0),
            Vector3(0, 1, 0), Vector3(1, 1, 0),
        )
        assert math.isclose(d, 1.0, abs_tol=1e-6)

    def test_intersecting(self) -> None:
        d = segments_closest_distance(
            Vector3(0, 0, 0), Vector3(2, 0, 0),
            Vector3(1, -1, 0), Vector3(1, 1, 0),
        )
        assert math.isclose(d, 0.0, abs_tol=1e-6)

    def test_point_segments(self) -> None:
        d = segments_closest_distance(
            Vector3(0, 0, 0), Vector3(0, 0, 0),
            Vector3(1, 0, 0), Vector3(1, 0, 0),
        )
        assert math.isclose(d, 1.0, abs_tol=1e-6)


# ── barycentric_coordinates ────────────────────────────────


class TestBarycentricCoordinates:
    def test_vertex_a(self) -> None:
        a, b, c = Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)
        u, v, w = barycentric_coordinates(a, a, b, c)
        assert math.isclose(u, 1.0, abs_tol=1e-6)
        assert math.isclose(v, 0.0, abs_tol=1e-6)
        assert math.isclose(w, 0.0, abs_tol=1e-6)

    def test_centroid(self) -> None:
        a, b, c = Vector3(0, 0, 0), Vector3(3, 0, 0), Vector3(0, 3, 0)
        centroid = Vector3(1, 1, 0)
        u, v, w = barycentric_coordinates(centroid, a, b, c)
        assert math.isclose(u, 1 / 3, abs_tol=1e-6)
        assert math.isclose(v, 1 / 3, abs_tol=1e-6)
        assert math.isclose(w, 1 / 3, abs_tol=1e-6)


# ── triangle_area ──────────────────────────────────────────


class TestTriangleArea:
    def test_unit_right(self) -> None:
        area = triangle_area(
            Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)
        )
        assert math.isclose(area, 0.5)

    def test_degenerate(self) -> None:
        area = triangle_area(
            Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0)
        )
        assert math.isclose(area, 0.0, abs_tol=1e-12)


# ── triangle_normal ─────────────────────────────────────────


class TestTriangleNormal:
    def test_xy_plane(self) -> None:
        n = triangle_normal(
            Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)
        )
        assert n.approx_equal(Vector3(0, 0, 1))

    def test_degenerate_raises(self) -> None:
        with pytest.raises(ValueError):
            triangle_normal(
                Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0)
            )
