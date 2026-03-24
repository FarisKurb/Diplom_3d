"""Unit tests for mesh.obj_loader."""

import os
import math
import tempfile

import pytest
from core.vector3 import Vector3
from mesh.obj_loader import load_obj, ensure_default_cube


# ── Helpers ─────────────────────────────────────────────────

_CUBE_OBJ = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "cube.obj"
)


# ── load_obj ────────────────────────────────────────────────


class TestLoadObj:
    def test_load_cube(self) -> None:
        m = load_obj(_CUBE_OBJ)
        assert m.num_vertices == 8
        # 6 quad faces → 12 triangles after fan triangulation.
        assert m.num_faces == 12

    def test_cube_triangles(self) -> None:
        m = load_obj(_CUBE_OBJ)
        assert len(m.triangles) == 12

    def test_cube_bounding_box(self) -> None:
        m = load_obj(_CUBE_OBJ)
        lo, hi = m.bounding_box()
        assert math.isclose(lo.x, -0.5)
        assert math.isclose(hi.x, 0.5)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_obj("nonexistent_path.obj")

    def test_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".obj", delete=False, encoding="utf-8"
        ) as f:
            f.write("# empty\n")
            path = f.name
        try:
            with pytest.raises(ValueError):
                load_obj(path)
        finally:
            os.unlink(path)

    def test_triangle_face(self) -> None:
        content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".obj", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            m = load_obj(path)
            assert m.num_faces == 1
            assert m.num_vertices == 3
        finally:
            os.unlink(path)

    def test_face_with_texture_indices(self) -> None:
        content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1 2/2 3/3\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".obj", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            m = load_obj(path)
            assert m.num_faces == 1
        finally:
            os.unlink(path)

    def test_face_with_full_indices(self) -> None:
        content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".obj", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            m = load_obj(path)
            assert m.num_faces == 1
        finally:
            os.unlink(path)

    def test_negative_indices(self) -> None:
        content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".obj", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            m = load_obj(path)
            assert m.num_faces == 1
            assert m.faces[0] == (0, 1, 2)
        finally:
            os.unlink(path)

    def test_polygon_fan_triangulation(self) -> None:
        content = (
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0.5 1.5 0\nv 0 1 0\n"
            "f 1 2 3 4 5\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".obj", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            m = load_obj(path)
            # Pentagon → 3 triangles
            assert m.num_faces == 3
        finally:
            os.unlink(path)


# ── ensure_default_cube ─────────────────────────────────────


class TestEnsureDefaultCube:
    def test_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "cube.obj")
            result = ensure_default_cube(path)
            assert os.path.isfile(result)
            m = load_obj(result)
            assert m.num_vertices == 8
            assert m.num_faces == 12

    def test_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cube.obj")
            ensure_default_cube(path)
            mtime1 = os.path.getmtime(path)
            ensure_default_cube(path)
            mtime2 = os.path.getmtime(path)
            assert mtime1 == mtime2
