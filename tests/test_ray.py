"""Unit tests for core.ray.Ray."""

import math
from core.vector3 import Vector3
from core.ray import Ray


class TestRay:
    def test_direction_is_normalized(self) -> None:
        r = Ray(Vector3(0, 0, 0), Vector3(0, 0, 10))
        assert math.isclose(r.direction.length(), 1.0)

    def test_point_at_origin(self) -> None:
        r = Ray(Vector3(1, 2, 3), Vector3(1, 0, 0))
        assert r.point_at(0).approx_equal(Vector3(1, 2, 3))

    def test_point_at_positive(self) -> None:
        r = Ray(Vector3(0, 0, 0), Vector3(0, 1, 0))
        assert r.point_at(5).approx_equal(Vector3(0, 5, 0))

    def test_repr(self) -> None:
        r = Ray(Vector3(0, 0, 0), Vector3(1, 0, 0))
        assert "Ray" in repr(r)
