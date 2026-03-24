"""Unit tests for core.vector3.Vector3."""

import math
import pytest
from core.vector3 import Vector3


# ── Construction ────────────────────────────────────────────


class TestVector3Construction:
    def test_default_is_zero(self) -> None:
        v = Vector3()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_explicit_values(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_from_tuple(self) -> None:
        v = Vector3.from_tuple((4.0, 5.0, 6.0))
        assert v == Vector3(4.0, 5.0, 6.0)

    def test_to_tuple(self) -> None:
        assert Vector3(1.0, 2.0, 3.0).to_tuple() == (1.0, 2.0, 3.0)

    def test_zero_factory(self) -> None:
        assert Vector3.zero() == Vector3(0.0, 0.0, 0.0)

    def test_immutability(self) -> None:
        v = Vector3(1.0, 2.0, 3.0)
        with pytest.raises(AttributeError):
            v.x = 10.0  # type: ignore[misc]


# ── Arithmetic ──────────────────────────────────────────────


class TestVector3Arithmetic:
    def test_add(self) -> None:
        assert Vector3(1, 2, 3) + Vector3(4, 5, 6) == Vector3(5, 7, 9)

    def test_sub(self) -> None:
        assert Vector3(4, 5, 6) - Vector3(1, 2, 3) == Vector3(3, 3, 3)

    def test_neg(self) -> None:
        assert -Vector3(1, -2, 3) == Vector3(-1, 2, -3)

    def test_mul_scalar(self) -> None:
        assert Vector3(1, 2, 3) * 2 == Vector3(2, 4, 6)

    def test_rmul_scalar(self) -> None:
        assert 3 * Vector3(1, 2, 3) == Vector3(3, 6, 9)

    def test_div_scalar(self) -> None:
        v = Vector3(4, 6, 8) / 2
        assert v.approx_equal(Vector3(2, 3, 4))

    def test_div_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            Vector3(1, 2, 3) / 0


# ── Dot / Cross ─────────────────────────────────────────────


class TestVector3DotCross:
    def test_dot(self) -> None:
        assert Vector3(1, 0, 0).dot(Vector3(0, 1, 0)) == 0.0

    def test_dot_parallel(self) -> None:
        assert Vector3(2, 0, 0).dot(Vector3(3, 0, 0)) == 6.0

    def test_cross_basis(self) -> None:
        assert Vector3(1, 0, 0).cross(Vector3(0, 1, 0)) == Vector3(0, 0, 1)

    def test_cross_anti_commutative(self) -> None:
        a, b = Vector3(1, 2, 3), Vector3(4, 5, 6)
        assert a.cross(b).approx_equal(-b.cross(a))


# ── Length / Distance / Normalize ───────────────────────────


class TestVector3LengthNorm:
    def test_length(self) -> None:
        assert math.isclose(Vector3(3, 4, 0).length(), 5.0)

    def test_length_squared(self) -> None:
        assert Vector3(3, 4, 0).length_squared() == 25.0

    def test_normalized(self) -> None:
        n = Vector3(0, 0, 5).normalized()
        assert n.approx_equal(Vector3(0, 0, 1))

    def test_normalize_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            Vector3.zero().normalized()

    def test_distance(self) -> None:
        d = Vector3(0, 0, 0).distance_to(Vector3(3, 4, 0))
        assert math.isclose(d, 5.0)


# ── Miscellaneous ───────────────────────────────────────────


class TestVector3Misc:
    def test_lerp_start(self) -> None:
        a, b = Vector3(0, 0, 0), Vector3(10, 10, 10)
        assert a.lerp(b, 0.0) == a

    def test_lerp_end(self) -> None:
        a, b = Vector3(0, 0, 0), Vector3(10, 10, 10)
        assert a.lerp(b, 1.0) == b

    def test_lerp_mid(self) -> None:
        a, b = Vector3(0, 0, 0), Vector3(10, 0, 0)
        assert a.lerp(b, 0.5).approx_equal(Vector3(5, 0, 0))

    def test_repr(self) -> None:
        assert "Vector3" in repr(Vector3(1, 2, 3))

    def test_hash_equal(self) -> None:
        assert hash(Vector3(1, 2, 3)) == hash(Vector3(1, 2, 3))

    def test_eq_not_vector(self) -> None:
        assert Vector3(1, 2, 3) != (1, 2, 3)
