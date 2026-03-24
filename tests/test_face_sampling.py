"""Unit tests for pathfinding.face_sampling."""

import math
from core.vector3 import Vector3
from geometry.triangle import Triangle
from mesh.mesh import Mesh
from pathfinding.face_sampling import sample_triangle, sample_mesh_faces


def _xy_tri() -> Triangle:
    return Triangle(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0))


class TestSampleTriangle:
    def test_centroid_included(self) -> None:
        pts = sample_triangle(_xy_tri(), bary_steps=0, include_edge_midpoints=False)
        assert len(pts) == 1
        assert pts[0].approx_equal(Vector3(1 / 3, 1 / 3, 0))

    def test_edge_midpoints_included(self) -> None:
        pts = sample_triangle(_xy_tri(), bary_steps=0, include_centroid=False)
        assert len(pts) == 3

    def test_no_vertices_in_output(self) -> None:
        """Vertices of the triangle must NOT appear (they are mesh nodes)."""
        tri = _xy_tri()
        pts = sample_triangle(tri, bary_steps=5)
        verts = set(v.to_tuple() for v in tri.vertices())
        for p in pts:
            assert p.to_tuple() not in verts

    def test_bary_steps_2(self) -> None:
        pts = sample_triangle(
            _xy_tri(), bary_steps=2,
            include_centroid=False, include_edge_midpoints=False,
        )
        # bary_steps=2: interior points with (i,j,k) where none == 2
        # (0,1,1), (1,0,1), (1,1,0) → 3 edge midpoints already excluded
        # so only (0,1,1)=midpoint, etc. — but actually with vertices excluded
        # the only point is (1,1,0)/2 etc.  With bary_steps=2 and vertices
        # skipped, the remaining grid points are the 3 edge midpoints.
        assert len(pts) == 3

    def test_bary_steps_3_more_points(self) -> None:
        pts = sample_triangle(
            _xy_tri(), bary_steps=3,
            include_centroid=False, include_edge_midpoints=False,
        )
        # More points than with bary_steps=2
        assert len(pts) > 3

    def test_all_points_on_triangle(self) -> None:
        tri = _xy_tri()
        pts = sample_triangle(tri, bary_steps=4)
        for p in pts:
            assert tri.contains_point(p, eps=1e-6)

    def test_no_duplicates(self) -> None:
        pts = sample_triangle(_xy_tri(), bary_steps=4)
        keys = [(round(p.x, 9), round(p.y, 9), round(p.z, 9)) for p in pts]
        assert len(keys) == len(set(keys))


class TestSampleMeshFaces:
    def test_single_triangle_mesh(self) -> None:
        mesh = Mesh(
            [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)],
            [(0, 1, 2)],
        )
        pts = sample_mesh_faces(mesh, bary_steps=3)
        assert len(pts) > 0

    def test_deduplication_across_shared_edges(self) -> None:
        """Two triangles sharing an edge should not duplicate midpoints."""
        mesh = Mesh(
            [
                Vector3(0, 0, 0),
                Vector3(1, 0, 0),
                Vector3(1, 1, 0),
                Vector3(0, 1, 0),
            ],
            [(0, 1, 2), (0, 2, 3)],
        )
        pts = sample_mesh_faces(mesh, bary_steps=3)
        keys = [(round(p.x, 9), round(p.y, 9), round(p.z, 9)) for p in pts]
        assert len(keys) == len(set(keys))

    def test_empty_mesh(self) -> None:
        mesh = Mesh([], [])
        assert sample_mesh_faces(mesh) == []
