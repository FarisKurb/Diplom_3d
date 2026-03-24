"""Unit tests for geometry.intersection (ray–tri, segment–tri, segment–mesh)."""

import math
import pytest
from core.vector3 import Vector3
from core.ray import Ray
from geometry.triangle import Triangle
from geometry.intersection import (
    ray_triangle_intersection,
    segment_triangle_intersection,
    segment_intersects_mesh,
    segment_mesh_closest_intersection,
)


# ── Helpers ─────────────────────────────────────────────────


def _xy_tri() -> Triangle:
    return Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0))


def _unit_cube_triangles() -> list[Triangle]:
    """12 triangles forming a unit cube [0,1]^3 (all faces)."""
    v = [
        Vector3(0, 0, 0),  # 0
        Vector3(1, 0, 0),  # 1
        Vector3(1, 1, 0),  # 2
        Vector3(0, 1, 0),  # 3
        Vector3(0, 0, 1),  # 4
        Vector3(1, 0, 1),  # 5
        Vector3(1, 1, 1),  # 6
        Vector3(0, 1, 1),  # 7
    ]
    faces = [
        # front  (z=1)
        (4, 5, 6), (4, 6, 7),
        # back   (z=0)
        (0, 3, 2), (0, 2, 1),
        # left   (x=0)
        (0, 4, 7), (0, 7, 3),
        # right  (x=1)
        (1, 2, 6), (1, 6, 5),
        # top    (y=1)
        (3, 7, 6), (3, 6, 2),
        # bottom (y=0)
        (0, 1, 5), (0, 5, 4),
    ]
    return [Triangle(v[a], v[b], v[c]) for a, b, c in faces]


# ── Ray–Triangle ────────────────────────────────────────────


class TestRayTriangle:
    def test_hit_center(self) -> None:
        ray = Ray(Vector3(0.25, 0.25, 1), Vector3(0, 0, -1))
        result = ray_triangle_intersection(ray, _xy_tri())
        assert result is not None
        t, u, v = result
        assert math.isclose(t, 1.0, abs_tol=1e-6)

    def test_miss_outside(self) -> None:
        ray = Ray(Vector3(2, 2, 1), Vector3(0, 0, -1))
        assert ray_triangle_intersection(ray, _xy_tri()) is None

    def test_parallel(self) -> None:
        ray = Ray(Vector3(0, 0, 0), Vector3(1, 0, 0))
        assert ray_triangle_intersection(ray, _xy_tri()) is None

    def test_behind_ray(self) -> None:
        ray = Ray(Vector3(0.25, 0.25, -1), Vector3(0, 0, -1))
        assert ray_triangle_intersection(ray, _xy_tri()) is None

    def test_backface_culled(self) -> None:
        ray = Ray(Vector3(0.25, 0.25, -1), Vector3(0, 0, 1))
        result = ray_triangle_intersection(ray, _xy_tri(), cull_backface=True)
        assert result is None

    def test_hit_vertex(self) -> None:
        ray = Ray(Vector3(0, 0, 1), Vector3(0, 0, -1))
        result = ray_triangle_intersection(ray, _xy_tri())
        assert result is not None

    def test_hit_edge(self) -> None:
        ray = Ray(Vector3(0.5, 0, 1), Vector3(0, 0, -1))
        result = ray_triangle_intersection(ray, _xy_tri())
        assert result is not None


# ── Segment–Triangle ───────────────────────────────────────


class TestSegmentTriangle:
    def test_through(self) -> None:
        result = segment_triangle_intersection(
            Vector3(0.25, 0.25, 1), Vector3(0.25, 0.25, -1), _xy_tri()
        )
        assert result is not None
        t, u, v = result
        assert math.isclose(t, 0.5, abs_tol=1e-6)

    def test_miss(self) -> None:
        result = segment_triangle_intersection(
            Vector3(2, 2, 1), Vector3(2, 2, -1), _xy_tri()
        )
        assert result is None

    def test_too_short(self) -> None:
        result = segment_triangle_intersection(
            Vector3(0.25, 0.25, 1), Vector3(0.25, 0.25, 0.5), _xy_tri()
        )
        assert result is None

    def test_starts_on_triangle(self) -> None:
        result = segment_triangle_intersection(
            Vector3(0.25, 0.25, 0), Vector3(0.25, 0.25, -1), _xy_tri()
        )
        assert result is not None
        t, _, _ = result
        assert math.isclose(t, 0.0, abs_tol=1e-6)

    def test_parallel_no_hit(self) -> None:
        result = segment_triangle_intersection(
            Vector3(0, 0, 1), Vector3(1, 0, 1), _xy_tri()
        )
        assert result is None


# ── Segment–Mesh (interior crossing) ───────────────────────


class TestSegmentMesh:
    def test_through_cube_interior(self) -> None:
        tris = _unit_cube_triangles()
        assert segment_intersects_mesh(
            Vector3(0.5, 0.5, -1), Vector3(0.5, 0.5, 2), tris
        )

    def test_outside_cube(self) -> None:
        tris = _unit_cube_triangles()
        assert not segment_intersects_mesh(
            Vector3(2, 2, 0), Vector3(3, 3, 0), tris
        )

    def test_touching_face_not_interior(self) -> None:
        tris = _unit_cube_triangles()
        # Segment along the top face y=1 should NOT count as interior.
        assert not segment_intersects_mesh(
            Vector3(0.2, 1.0, 0.2), Vector3(0.8, 1.0, 0.8), tris
        )

    def test_segment_between_vertices_on_surface(self) -> None:
        tris = _unit_cube_triangles()
        # Edge of the cube along x-axis at y=0, z=0.
        assert not segment_intersects_mesh(
            Vector3(0, 0, 0), Vector3(1, 0, 0), tris
        )

    def test_free_space_to_back_face_through_interior(self) -> None:
        """Segment from free space to a back-face point passes through interior."""
        tris = _unit_cube_triangles()
        # Start in front of cube, end on back face → must cross front face.
        assert segment_intersects_mesh(
            Vector3(0.5, 0.5, -1), Vector3(0.5, 0.5, 1.0), tris
        )

    def test_free_space_to_front_face_visible(self) -> None:
        """Segment from free space directly to a front-face point is visible."""
        tris = _unit_cube_triangles()
        # Start in front of cube, end on front face → no interior crossing.
        assert not segment_intersects_mesh(
            Vector3(0.5, 0.5, -1), Vector3(0.5, 0.5, 0.0), tris
        )


# ── Closest intersection ───────────────────────────────────


class TestSegmentMeshClosest:
    def test_closest_hit(self) -> None:
        tris = _unit_cube_triangles()
        result = segment_mesh_closest_intersection(
            Vector3(0.5, 0.5, -1), Vector3(0.5, 0.5, 2), tris
        )
        assert result is not None
        t, pt, idx = result
        # The nearest face is z=0 at t ≈ 1/3.
        assert math.isclose(pt.z, 0.0, abs_tol=1e-6)

    def test_no_hit(self) -> None:
        tris = _unit_cube_triangles()
        result = segment_mesh_closest_intersection(
            Vector3(5, 5, 5), Vector3(6, 6, 6), tris
        )
        assert result is None
