"""Tests for render.mesh_renderer and render.path_renderer.

OpenGL calls are mocked out so the tests run without a GPU / display.
"""

from __future__ import annotations

import math
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ── Stub out OpenGL before importing renderers ─────────────

_gl_stub = MagicMock()
sys.modules.setdefault("OpenGL", _gl_stub)
sys.modules.setdefault("OpenGL.GL", _gl_stub)
sys.modules.setdefault("OpenGL.GLU", _gl_stub)

from core.vector3 import Vector3
from mesh.mesh import Mesh
from render.mesh_renderer import MeshRenderer
from render.path_renderer import PathRenderer, draw_sphere_marker
from config import MESH_COLOR, WIREFRAME_COLOR, PATH_COLOR, POINT_COLOR, POINT_RADIUS, START_POINT_COLOR, END_POINT_COLOR


# ── helpers ─────────────────────────────────────────────────

def _make_triangle_mesh() -> Mesh:
    """Single-triangle mesh for testing."""
    return Mesh(
        vertices=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)],
        faces=[(0, 1, 2)],
    )


def _make_cube_mesh() -> Mesh:
    """A minimal cube-like mesh (8 verts, 12 faces)."""
    verts = [
        Vector3(-0.5, -0.5, 0.5), Vector3(0.5, -0.5, 0.5),
        Vector3(0.5, 0.5, 0.5),   Vector3(-0.5, 0.5, 0.5),
        Vector3(-0.5, -0.5, -0.5), Vector3(0.5, -0.5, -0.5),
        Vector3(0.5, 0.5, -0.5),  Vector3(-0.5, 0.5, -0.5),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),  # front
        (4, 7, 6), (4, 6, 5),  # back
        (0, 3, 7), (0, 7, 4),  # left
        (1, 5, 6), (1, 6, 2),  # right
        (3, 2, 6), (3, 6, 7),  # top
        (0, 4, 5), (0, 5, 1),  # bottom
    ]
    return Mesh(verts, faces)


# ═══════════════════════════════════════════════════════════
#  MeshRenderer Tests
# ═══════════════════════════════════════════════════════════

class TestMeshRendererInit:
    def test_defaults(self) -> None:
        mr = MeshRenderer()
        assert mr.mesh is None
        assert mr.face_color == MESH_COLOR
        assert mr.wireframe_color == WIREFRAME_COLOR
        assert mr.show_faces is True
        assert mr.show_wireframe is True

    def test_custom_colors(self) -> None:
        mr = MeshRenderer(face_color=(1.0, 0.0, 0.0), wireframe_color=(0.0, 1.0, 0.0))
        assert mr.face_color == (1.0, 0.0, 0.0)
        assert mr.wireframe_color == (0.0, 1.0, 0.0)

    def test_with_mesh(self) -> None:
        m = _make_triangle_mesh()
        mr = MeshRenderer(mesh=m)
        assert mr.mesh is m


class TestMeshRendererDraw:
    def test_draw_none_mesh_no_error(self) -> None:
        """draw() with no mesh assigned should silently do nothing."""
        mr = MeshRenderer()
        mr.draw()  # must not raise

    def test_draw_with_mesh_runs(self) -> None:
        """draw() with a real mesh should execute without error."""
        mr = MeshRenderer(mesh=_make_triangle_mesh())
        mr.draw()

    def test_draw_cube(self) -> None:
        mr = MeshRenderer(mesh=_make_cube_mesh())
        mr.draw()

    def test_draw_faces_only(self) -> None:
        mr = MeshRenderer(mesh=_make_triangle_mesh(), show_wireframe=False)
        mr.draw()

    def test_draw_wireframe_only(self) -> None:
        mr = MeshRenderer(mesh=_make_triangle_mesh(), show_faces=False)
        mr.draw()

    def test_draw_nothing_when_both_off(self) -> None:
        mr = MeshRenderer(mesh=_make_triangle_mesh(), show_faces=False, show_wireframe=False)
        mr.draw()


class TestMeshRendererMeshSwap:
    def test_replace_mesh(self) -> None:
        mr = MeshRenderer(mesh=_make_triangle_mesh())
        new_mesh = _make_cube_mesh()
        mr.mesh = new_mesh
        assert mr.mesh is new_mesh
        mr.draw()

    def test_set_mesh_to_none(self) -> None:
        mr = MeshRenderer(mesh=_make_triangle_mesh())
        mr.mesh = None
        mr.draw()  # should not raise


# ═══════════════════════════════════════════════════════════
#  PathRenderer Tests
# ═══════════════════════════════════════════════════════════

class TestPathRendererInit:
    def test_defaults(self) -> None:
        pr = PathRenderer()
        assert pr.path == []
        assert pr.path_color == PATH_COLOR
        assert pr.start_color == START_POINT_COLOR
        assert pr.end_color == END_POINT_COLOR
        assert pr.point_radius == POINT_RADIUS
        assert pr.line_width == 3.0
        assert pr.start_point is None
        assert pr.end_point is None

    def test_custom_init(self) -> None:
        pts = [Vector3(0, 0, 0), Vector3(1, 1, 1)]
        pr = PathRenderer(path=pts, line_width=5.0)
        assert len(pr.path) == 2
        assert pr.line_width == 5.0


class TestPathRendererSetPath:
    def test_set_path(self) -> None:
        pr = PathRenderer()
        pts = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0)]
        pr.set_path(pts)
        assert len(pr.path) == 3
        assert pr.path[1] == Vector3(1, 0, 0)

    def test_set_path_copies(self) -> None:
        """set_path should make a copy so the original list is independent."""
        pr = PathRenderer()
        original = [Vector3(0, 0, 0)]
        pr.set_path(original)
        original.append(Vector3(1, 1, 1))
        assert len(pr.path) == 1

    def test_clear(self) -> None:
        pr = PathRenderer()
        pr.start_point = Vector3(0, 0, 0)
        pr.end_point = Vector3(1, 1, 1)
        pr.set_path([Vector3(0, 0, 0), Vector3(1, 1, 1)])
        pr.clear()
        assert pr.path == []
        assert pr.start_point is None
        assert pr.end_point is None


class TestPathRendererDraw:
    def test_draw_empty(self) -> None:
        """draw() with no path or points should not raise."""
        pr = PathRenderer()
        pr.draw()

    def test_draw_with_path(self) -> None:
        pr = PathRenderer(path=[Vector3(0, 0, 0), Vector3(1, 0, 0)])
        pr.draw()

    def test_draw_single_point_path(self) -> None:
        """A path with only 1 point should not draw the line strip."""
        pr = PathRenderer(path=[Vector3(0, 0, 0)])
        pr.draw()

    def test_draw_with_markers(self) -> None:
        pr = PathRenderer()
        pr.start_point = Vector3(0, 0, 0)
        pr.end_point = Vector3(2, 3, 4)
        pr.draw()

    def test_draw_full(self) -> None:
        """Full draw with path + both markers."""
        pr = PathRenderer(path=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0)])
        pr.start_point = Vector3(0, 0, 0)
        pr.end_point = Vector3(1, 1, 0)
        pr.draw()


class TestDrawSphereMarker:
    def test_call_no_error(self) -> None:
        draw_sphere_marker(Vector3(0, 0, 0), 0.1, (1.0, 0.0, 0.0))

    def test_custom_slices_stacks(self) -> None:
        draw_sphere_marker(Vector3(1, 2, 3), 0.05, (0.0, 1.0, 0.0), slices=6, stacks=4)


class TestPathRendererMarkers:
    def test_start_only(self) -> None:
        pr = PathRenderer()
        pr.start_point = Vector3(1, 2, 3)
        assert pr.start_point == Vector3(1, 2, 3)
        assert pr.end_point is None
        pr.draw()

    def test_end_only(self) -> None:
        pr = PathRenderer()
        pr.end_point = Vector3(4, 5, 6)
        assert pr.end_point == Vector3(4, 5, 6)
        assert pr.start_point is None
        pr.draw()
