"""Unit tests for mesh.mesh.Mesh."""

from core.vector3 import Vector3
from mesh.mesh import Mesh


def _simple_mesh() -> Mesh:
    """Two-triangle quad in the XY plane."""
    verts = [
        Vector3(0, 0, 0),
        Vector3(1, 0, 0),
        Vector3(1, 1, 0),
        Vector3(0, 1, 0),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    return Mesh(verts, faces)


class TestMesh:
    def test_num_vertices(self) -> None:
        assert _simple_mesh().num_vertices == 4

    def test_num_faces(self) -> None:
        assert _simple_mesh().num_faces == 2

    def test_triangles_count(self) -> None:
        assert len(_simple_mesh().triangles) == 2

    def test_triangles_cached(self) -> None:
        m = _simple_mesh()
        t1 = m.triangles
        t2 = m.triangles
        assert t1 is t2

    def test_invalidate_cache(self) -> None:
        m = _simple_mesh()
        t1 = m.triangles
        m.invalidate_cache()
        t2 = m.triangles
        assert t1 is not t2

    def test_bounding_box(self) -> None:
        lo, hi = _simple_mesh().bounding_box()
        assert lo.approx_equal(Vector3(0, 0, 0))
        assert hi.approx_equal(Vector3(1, 1, 0))

    def test_center(self) -> None:
        c = _simple_mesh().center()
        assert c.approx_equal(Vector3(0.5, 0.5, 0))

    def test_bounding_box_empty(self) -> None:
        m = Mesh([], [])
        lo, hi = m.bounding_box()
        assert lo == Vector3.zero()

    def test_repr(self) -> None:
        assert "Mesh" in repr(_simple_mesh())
