"""Tests for Stage 5 — Mesh scanner and mesh selection menu.

Covers:
    - scan_mesh_directory: returns .obj files, sorted, absolute paths, 
      handles empty/missing directories
    - mesh_display_name: extracts human-readable names
    - Application._cycle_mesh: swaps mesh and resets state
    - Application.mesh_name: returns display name from catalogue
    - M key triggers mesh cycling
    - HUD title shows mesh name
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from core.vector3 import Vector3
from mesh.mesh_scanner import scan_mesh_directory, mesh_display_name


# ═══════════════════════════════════════════════════════════
#  scan_mesh_directory
# ═══════════════════════════════════════════════════════════

class TestScanMeshDirectory:
    def test_finds_obj_files(self, tmp_path):
        (tmp_path / "cube.obj").write_text("# cube")
        (tmp_path / "sphere.obj").write_text("# sphere")
        (tmp_path / "readme.txt").write_text("not a mesh")
        result = scan_mesh_directory(str(tmp_path))
        assert len(result) == 2
        names = [os.path.basename(p) for p in result]
        assert "cube.obj" in names
        assert "sphere.obj" in names

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / "test.obj").write_text("# test")
        result = scan_mesh_directory(str(tmp_path))
        for p in result:
            assert os.path.isabs(p)

    def test_sorted(self, tmp_path):
        (tmp_path / "zebra.obj").write_text("#")
        (tmp_path / "apple.obj").write_text("#")
        (tmp_path / "mango.obj").write_text("#")
        result = scan_mesh_directory(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert names == sorted(names)

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / "model.OBJ").write_text("#")
        result = scan_mesh_directory(str(tmp_path))
        assert len(result) == 1

    def test_empty_directory(self, tmp_path):
        result = scan_mesh_directory(str(tmp_path))
        assert result == []

    def test_missing_directory(self):
        result = scan_mesh_directory("/nonexistent/path/xyz")
        assert result == []

    def test_no_obj_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        result = scan_mesh_directory(str(tmp_path))
        assert result == []


# ═══════════════════════════════════════════════════════════
#  mesh_display_name
# ═══════════════════════════════════════════════════════════

class TestMeshDisplayName:
    def test_simple(self):
        assert mesh_display_name("assets/cube.obj") == "cube"

    def test_with_spaces(self):
        assert mesh_display_name("assets/My Model.obj") == "My Model"

    def test_absolute_path(self):
        assert mesh_display_name("C:/users/test/Melon.obj") == "Melon"

    def test_no_extension(self):
        assert mesh_display_name("noext") == "noext"


# ═══════════════════════════════════════════════════════════
#  Application mesh selection integration
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def multi_mesh_dir(tmp_path):
    """Create a temp assets directory with multiple .obj cube files."""
    from mesh.obj_loader import _generate_cube_obj_content
    content = _generate_cube_obj_content()
    for name in ("Alpha.obj", "Beta.obj", "Gamma.obj"):
        (tmp_path / name).write_text(content)
    return tmp_path


@pytest.fixture
def app_with_meshes(multi_mesh_dir):
    """Application whose assets dir has 3 meshes."""
    from main import Application
    first_obj = str(multi_mesh_dir / "Alpha.obj")
    app = Application(mesh_path=first_obj)
    # Override the scanner result to point at our temp dir.
    app._mesh_paths = scan_mesh_directory(str(multi_mesh_dir))
    app._mesh_index = 0
    return app


class TestApplicationMeshCatalogue:
    def test_mesh_paths_populated(self, app_with_meshes):
        assert len(app_with_meshes._mesh_paths) == 3

    def test_mesh_name_property(self, app_with_meshes):
        assert app_with_meshes.mesh_name == "Alpha"

    def test_mesh_name_after_cycle(self, app_with_meshes):
        app_with_meshes.point_placer = MagicMock()
        app_with_meshes._cycle_mesh()
        assert app_with_meshes.mesh_name == "Beta"

    def test_cycle_wraps_around(self, app_with_meshes):
        app_with_meshes.point_placer = MagicMock()
        for _ in range(3):
            app_with_meshes._cycle_mesh()
        assert app_with_meshes.mesh_name == "Alpha"

    def test_cycle_updates_mesh_object(self, app_with_meshes):
        app_with_meshes.point_placer = MagicMock()
        old_mesh = app_with_meshes.mesh
        app_with_meshes._cycle_mesh()
        # Different Mesh instance (even though geometry is the same)
        assert app_with_meshes.mesh is not old_mesh

    def test_cycle_updates_mesh_renderer(self, app_with_meshes):
        app_with_meshes.point_placer = MagicMock()
        app_with_meshes._cycle_mesh()
        assert app_with_meshes.mesh_renderer.mesh is app_with_meshes.mesh

    def test_cycle_updates_point_placer(self, app_with_meshes):
        mock_placer = MagicMock()
        app_with_meshes.point_placer = mock_placer
        app_with_meshes._cycle_mesh()
        assert mock_placer.mesh is app_with_meshes.mesh

    def test_cycle_resets_path(self, app_with_meshes):
        app_with_meshes.point_placer = MagicMock()
        app_with_meshes.path_renderer.set_path(
            [Vector3(0, 0, 0), Vector3(1, 0, 0)]
        )
        app_with_meshes._cycle_mesh()
        assert app_with_meshes.path_renderer.path == []

    def test_cycle_no_crash_single_mesh(self, tmp_path):
        """When only one mesh exists, cycling prints a message but doesn't crash."""
        from mesh.obj_loader import _generate_cube_obj_content
        from main import Application
        content = _generate_cube_obj_content()
        obj_path = str(tmp_path / "only.obj")
        (tmp_path / "only.obj").write_text(content)
        app = Application(mesh_path=obj_path)
        app._mesh_paths = [obj_path]
        app._mesh_index = 0
        app._cycle_mesh()  # should not raise


class TestMKeyBinding:
    def test_m_key_calls_cycle(self, app_with_meshes):
        app_with_meshes._cycle_mesh = MagicMock()

        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_ESCAPE = 256
            mock_glfw.KEY_Q = 81
            mock_glfw.KEY_R = 82
            mock_glfw.KEY_S = 83
            mock_glfw.KEY_H = 72
            mock_glfw.KEY_M = 77

            app_with_meshes._on_key(77, 0, 1, 0)

        app_with_meshes._cycle_mesh.assert_called_once()


class TestHudShowsMeshName:
    def test_title_contains_mesh_name(self, app_with_meshes):
        app_with_meshes._update_hud()
        texts = [line[0] for line in app_with_meshes.hud_renderer.lines]
        title = texts[0]
        assert "Alpha" in title
        assert "3D Shortest Path" in title
