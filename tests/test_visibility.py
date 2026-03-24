"""Unit tests for geometry.visibility."""

from core.vector3 import Vector3
from geometry.triangle import Triangle
from geometry.visibility import is_visible


def _unit_cube_triangles() -> list[Triangle]:
    """12 triangles forming a unit cube [0,1]^3."""
    v = [
        Vector3(0, 0, 0), Vector3(1, 0, 0),
        Vector3(1, 1, 0), Vector3(0, 1, 0),
        Vector3(0, 0, 1), Vector3(1, 0, 1),
        Vector3(1, 1, 1), Vector3(0, 1, 1),
    ]
    faces = [
        (4, 5, 6), (4, 6, 7),
        (0, 3, 2), (0, 2, 1),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
        (3, 7, 6), (3, 6, 2),
        (0, 1, 5), (0, 5, 4),
    ]
    return [Triangle(v[a], v[b], v[c]) for a, b, c in faces]


class TestVisibility:
    def test_same_point(self) -> None:
        tris = _unit_cube_triangles()
        assert is_visible(Vector3(0, 0, 0), Vector3(0, 0, 0), tris)

    def test_both_outside_no_obstruction(self) -> None:
        tris = _unit_cube_triangles()
        # Both above the cube — clear line of sight.
        assert is_visible(Vector3(-1, 0.5, 0.5), Vector3(-2, 0.5, 0.5), tris)

    def test_through_cube_interior(self) -> None:
        tris = _unit_cube_triangles()
        # Segment passes right through the cube.
        assert not is_visible(Vector3(0.5, 0.5, -1), Vector3(0.5, 0.5, 2), tris)

    def test_along_surface(self) -> None:
        tris = _unit_cube_triangles()
        # Segment along the top face y=1 — should be visible.
        assert is_visible(Vector3(0.2, 1.0, 0.2), Vector3(0.8, 1.0, 0.8), tris)

    def test_along_edge(self) -> None:
        tris = _unit_cube_triangles()
        # Along the bottom front edge.
        assert is_visible(Vector3(0, 0, 0), Vector3(1, 0, 0), tris)

    def test_between_vertices_outside(self) -> None:
        tris = _unit_cube_triangles()
        # Two points on the same side, outside the cube.
        assert is_visible(Vector3(-1, -1, -1), Vector3(-2, -2, -2), tris)

    def test_tangent_to_face(self) -> None:
        tris = _unit_cube_triangles()
        # Segment that grazes the z=1 face.
        assert is_visible(Vector3(0.5, 0.5, 1.0), Vector3(2, 0.5, 1.0), tris)

    def test_free_space_to_far_surface_not_visible(self) -> None:
        tris = _unit_cube_triangles()
        # Free-space point to back face — goes through the interior.
        assert not is_visible(Vector3(0.5, 0.5, -1), Vector3(0.5, 0.5, 1.0), tris)

    def test_free_space_to_near_surface_visible(self) -> None:
        tris = _unit_cube_triangles()
        # Free-space point to nearest surface — no interior crossing.
        assert is_visible(Vector3(0.5, 0.5, -1), Vector3(0.5, 0.5, 0.0), tris)
