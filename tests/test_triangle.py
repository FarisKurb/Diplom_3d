"""Unit tests for geometry.triangle.Triangle."""

import math
import pytest
from core.vector3 import Vector3
from geometry.triangle import Triangle


# ── Helpers ─────────────────────────────────────────────────

def _xy_tri() -> Triangle:
    """Right triangle in the XY plane: (0,0,0)-(1,0,0)-(0,1,0)."""
    return Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0))


# ── Basic properties ────────────────────────────────────────


class TestTriangleProperties:
    def test_edges(self) -> None:
        t = _xy_tri()
        assert t.edge0() == Vector3(1, 0, 0)
        assert t.edge1() == Vector3(0, 1, 0)
        assert t.edge2() == Vector3(-1, 1, 0)

    def test_normal(self) -> None:
        n = _xy_tri().normal()
        assert n.approx_equal(Vector3(0, 0, 1))

    def test_area(self) -> None:
        assert math.isclose(_xy_tri().area(), 0.5)

    def test_centroid(self) -> None:
        c = _xy_tri().centroid()
        expected = Vector3(1 / 3, 1 / 3, 0)
        assert c.approx_equal(expected)

    def test_edge_midpoints(self) -> None:
        m01, m12, m02 = _xy_tri().edge_midpoints()
        assert m01.approx_equal(Vector3(0.5, 0.0, 0.0))
        assert m12.approx_equal(Vector3(0.5, 0.5, 0.0))
        assert m02.approx_equal(Vector3(0.0, 0.5, 0.0))


# ── Barycentric & containment ──────────────────────────────


class TestTriangleBarycentric:
    def test_vertex(self) -> None:
        t = _xy_tri()
        u, v, w = t.barycentric(t.v0)
        assert math.isclose(u, 1.0, abs_tol=1e-6)

    def test_centroid_bary(self) -> None:
        t = _xy_tri()
        u, v, w = t.barycentric(t.centroid())
        assert math.isclose(u, 1 / 3, abs_tol=1e-6)
        assert math.isclose(v, 1 / 3, abs_tol=1e-6)
        assert math.isclose(w, 1 / 3, abs_tol=1e-6)

    def test_contains_centroid(self) -> None:
        assert _xy_tri().contains_point(_xy_tri().centroid())

    def test_contains_vertex(self) -> None:
        t = _xy_tri()
        assert t.contains_point(t.v0)

    def test_outside_point(self) -> None:
        assert not _xy_tri().contains_point(Vector3(2, 2, 0))

    def test_off_plane(self) -> None:
        assert not _xy_tri().contains_point(Vector3(0.25, 0.25, 1.0))


# ── Degenerate triangle ────────────────────────────────────


class TestTriangleDegenerate:
    def test_degenerate_normal_raises(self) -> None:
        t = Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0))
        with pytest.raises(ValueError):
            t.normal()

    def test_degenerate_area_zero(self) -> None:
        t = Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0))
        assert math.isclose(t.area(), 0.0, abs_tol=1e-12)

    def test_repr(self) -> None:
        assert "Triangle" in repr(_xy_tri())
