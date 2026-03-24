"""Unit tests for mesh.mesh_topology."""

import os
from mesh.obj_loader import load_obj
from mesh.mesh_topology import (
    extract_edges,
    build_adjacency,
    faces_sharing_edge,
    is_manifold,
)
from mesh.mesh import Mesh
from core.vector3 import Vector3

_CUBE_OBJ = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "cube.obj"
)


def _quad_mesh() -> Mesh:
    verts = [
        Vector3(0, 0, 0),
        Vector3(1, 0, 0),
        Vector3(1, 1, 0),
        Vector3(0, 1, 0),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    return Mesh(verts, faces)


class TestExtractEdges:
    def test_quad_edge_count(self) -> None:
        edges = extract_edges(_quad_mesh())
        # 4 outer edges + 1 diagonal = 5
        assert len(edges) == 5

    def test_cube_edge_count(self) -> None:
        m = load_obj(_CUBE_OBJ)
        edges = extract_edges(m)
        # A cube has 12 edges + 6 face diagonals from fan triangulation = 18
        assert len(edges) == 18

    def test_edges_are_sorted(self) -> None:
        for a, b in extract_edges(_quad_mesh()):
            assert a < b


class TestBuildAdjacency:
    def test_quad(self) -> None:
        adj = build_adjacency(_quad_mesh())
        assert len(adj) == 4
        # vertex 0 connects to 1, 2, 3
        assert adj[0] == {1, 2, 3}

    def test_cube(self) -> None:
        m = load_obj(_CUBE_OBJ)
        adj = build_adjacency(m)
        assert len(adj) == 8


class TestFacesSharingEdge:
    def test_shared_diagonal(self) -> None:
        ef = faces_sharing_edge(_quad_mesh())
        # The diagonal edge (0,2) is shared by both triangles
        assert len(ef[(0, 2)]) == 2

    def test_boundary_edge(self) -> None:
        ef = faces_sharing_edge(_quad_mesh())
        # Edge (0,1) belongs to only one face
        assert len(ef[(0, 1)]) == 1


class TestIsManifold:
    def test_cube_is_manifold(self) -> None:
        m = load_obj(_CUBE_OBJ)
        assert is_manifold(m)

    def test_open_quad_not_manifold(self) -> None:
        assert not is_manifold(_quad_mesh())
