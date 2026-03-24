"""Tests for Stage 10 — Application integration (main.py).

Covers:
    - Application construction and component wiring
    - Mesh loading (default cube + custom path)
    - Render callback invokes sub-renderers
    - Mouse-button callback forwarding to PointPlacer
    - Key callback (reset and quit)
    - _on_both_points_placed triggers path solver + path renderer
    - Reset clears placer and path renderer
    - CLI entry-point argument parsing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import os

import pytest

from core.vector3 import Vector3
from mesh.mesh import Mesh
from mesh.obj_loader import ensure_default_cube
from pathfinding.path_solver import PathResult
from main import Application


# ── helpers ─────────────────────────────────────────────────

def _unit_cube_mesh() -> Mesh:
    """Axis-aligned unit cube centred at origin."""
    verts = [
        Vector3(-0.5, -0.5,  0.5), Vector3( 0.5, -0.5,  0.5),
        Vector3( 0.5,  0.5,  0.5), Vector3(-0.5,  0.5,  0.5),
        Vector3(-0.5, -0.5, -0.5), Vector3( 0.5, -0.5, -0.5),
        Vector3( 0.5,  0.5, -0.5), Vector3(-0.5,  0.5, -0.5),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 7, 6), (4, 6, 5),
        (0, 3, 7), (0, 7, 4),
        (1, 5, 6), (1, 6, 2),
        (3, 2, 6), (3, 6, 7),
        (0, 4, 5), (0, 5, 1),
    ]
    return Mesh(verts, faces)


@pytest.fixture
def cube_obj_path(tmp_path):
    """Write a cube.obj to a temp directory and return its path."""
    obj_path = str(tmp_path / "cube.obj")
    return ensure_default_cube(obj_path)


@pytest.fixture
def app(cube_obj_path):
    """Create an Application with the temp cube mesh (no OpenGL)."""
    return Application(mesh_path=cube_obj_path)


# ═══════════════════════════════════════════════════════════
#  Construction and component wiring
# ═══════════════════════════════════════════════════════════

class TestApplicationConstruction:
    def test_mesh_loaded(self, app: Application):
        assert app.mesh is not None
        assert len(app.mesh.vertices) == 8
        assert len(app.mesh.faces) == 12

    def test_renderer_created(self, app: Application):
        assert app.renderer is not None

    def test_mesh_renderer_has_mesh(self, app: Application):
        assert app.mesh_renderer.mesh is app.mesh

    def test_path_renderer_created_empty(self, app: Application):
        assert app.path_renderer is not None
        assert app.path_renderer.path == []

    def test_path_finder_created(self, app: Application):
        assert app.path_finder is not None
        assert app.path_finder.algorithm_name == "Dijkstra"

    def test_point_placer_not_created_before_run(self, app: Application):
        # PointPlacer requires a GLFW window, so it's None before run().
        assert app.point_placer is None


# ═══════════════════════════════════════════════════════════
#  Mesh loading
# ═══════════════════════════════════════════════════════════

class TestMeshLoading:
    def test_load_custom_obj(self, cube_obj_path):
        app = Application(mesh_path=cube_obj_path)
        assert len(app.mesh.vertices) == 8

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Application(mesh_path=str(tmp_path / "nonexistent.obj"))

    def test_default_cube_auto_created(self, tmp_path, monkeypatch):
        """When using the default mesh path and the file is absent,
        the app auto-generates it."""
        fake_default = str(tmp_path / "auto_cube.obj")
        monkeypatch.setattr("main.DEFAULT_MESH_PATH", fake_default)
        app = Application(mesh_path=fake_default)
        assert len(app.mesh.vertices) == 8
        assert os.path.isfile(fake_default)


# ═══════════════════════════════════════════════════════════
#  Render callback
# ═══════════════════════════════════════════════════════════

class TestRenderCallback:
    def test_on_render_draws_mesh_and_path(self, app: Application):
        app.mesh_renderer.draw = MagicMock()
        app.path_renderer.draw = MagicMock()

        app._on_render()

        app.mesh_renderer.draw.assert_called_once()
        app.path_renderer.draw.assert_called_once()


# ═══════════════════════════════════════════════════════════
#  Mouse-button callback
# ═══════════════════════════════════════════════════════════

class TestMouseButtonCallback:
    def test_forward_to_point_placer(self, app: Application):
        app.point_placer = MagicMock()
        app.point_placer.start_point = None
        app.point_placer.end_point = None

        app._on_mouse_button(0, 1, 0, 0)

        app.point_placer.on_click.assert_called_once_with(0, 1, 0)

    def test_updates_path_renderer_markers(self, app: Application):
        start = Vector3(1, 0, 0)
        app.point_placer = MagicMock()
        app.point_placer.start_point = start
        app.point_placer.end_point = None

        app._on_mouse_button(0, 1, 0, 0)

        assert app.path_renderer.start_point is start

    def test_no_crash_without_placer(self, app: Application):
        # Before run(), point_placer is None.
        app._on_mouse_button(0, 1, 0, 0)  # should not raise


# ═══════════════════════════════════════════════════════════
#  Key callback
# ═══════════════════════════════════════════════════════════

class TestKeyCallback:
    def test_r_key_resets(self, app: Application):
        app.point_placer = MagicMock()
        app.path_renderer.set_path([Vector3(0, 0, 0), Vector3(1, 0, 0)])
        app.path_renderer.start_point = Vector3(0, 0, 0)

        # glfw.KEY_R = 82, glfw.PRESS = 1
        app._on_key(82, 0, 1, 0)

        app.point_placer.reset.assert_called_once()
        assert app.path_renderer.path == []
        assert app.path_renderer.start_point is None
        assert app.path_renderer.end_point is None

    def test_escape_closes_window(self, app: Application):
        mock_window = MagicMock()
        app.renderer._window = mock_window

        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_ESCAPE = 256
            mock_glfw.KEY_Q = 81
            mock_glfw.KEY_R = 82

            app._on_key(256, 0, 1, 0)

            mock_glfw.set_window_should_close.assert_called_once_with(
                mock_window, True
            )

    def test_q_key_closes_window(self, app: Application):
        mock_window = MagicMock()
        app.renderer._window = mock_window

        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_ESCAPE = 256
            mock_glfw.KEY_Q = 81
            mock_glfw.KEY_R = 82

            app._on_key(81, 0, 1, 0)

            mock_glfw.set_window_should_close.assert_called_once_with(
                mock_window, True
            )

    def test_release_ignored(self, app: Application):
        """Key release events should not trigger any action."""
        app.point_placer = MagicMock()

        # action=0 → release; KEY_R = 82
        app._on_key(82, 0, 0, 0)

        app.point_placer.reset.assert_not_called()


# ═══════════════════════════════════════════════════════════
#  Path solving integration
# ═══════════════════════════════════════════════════════════

class TestPathSolvingIntegration:
    def test_on_both_points_placed_updates_path_renderer(self, app: Application):
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        fake_result = PathResult(
            found=True,
            distance=5.0,
            points=[start, Vector3(1, 0, 0), end],
            num_samples=10,
        )
        app.path_finder.compute_path = MagicMock(return_value=fake_result)

        app._on_both_points_placed(start, end)

        app.path_finder.compute_path.assert_called_once_with(start, end, app.mesh, smooth=True)
        assert len(app.path_renderer.path) == 3
        assert app.path_renderer.path[0] == start
        assert app.path_renderer.path[-1] == end

    def test_on_both_points_placed_no_path(self, app: Application):
        start = Vector3(0, 0, 2)
        end = Vector3(0, 0, -2)
        fake_result = PathResult(found=False, num_samples=10)
        app.path_finder.compute_path = MagicMock(return_value=fake_result)

        app._on_both_points_placed(start, end)

        assert app.path_renderer.path == []


# ═══════════════════════════════════════════════════════════
#  Reset
# ═══════════════════════════════════════════════════════════

class TestReset:
    def test_reset_clears_all(self, app: Application):
        app.point_placer = MagicMock()
        app.path_renderer.start_point = Vector3(0, 0, 0)
        app.path_renderer.end_point = Vector3(1, 0, 0)
        app.path_renderer.set_path([Vector3(0, 0, 0), Vector3(1, 0, 0)])

        app._reset()

        app.point_placer.reset.assert_called_once()
        assert app.path_renderer.path == []
        assert app.path_renderer.start_point is None
        assert app.path_renderer.end_point is None

    def test_reset_safe_without_placer(self, app: Application):
        """Reset should not crash if point_placer is None."""
        app.point_placer = None
        app._reset()  # no exception


# ═══════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════

class TestCLIEntryPoint:
    def test_main_with_default_args(self, cube_obj_path, monkeypatch):
        """main() creates Application and calls run()."""
        monkeypatch.setattr("sys.argv", ["main.py"])
        monkeypatch.setattr("main.DEFAULT_MESH_PATH", cube_obj_path)

        with patch("main.Application") as MockApp:
            instance = MockApp.return_value
            from main import main
            main()

            MockApp.assert_called_once_with(cube_obj_path, algorithm=None)
            instance.run.assert_called_once()

    def test_main_with_custom_path(self, cube_obj_path, monkeypatch):
        """main() passes CLI argument to Application."""
        monkeypatch.setattr("sys.argv", ["main.py", cube_obj_path])

        with patch("main.Application") as MockApp:
            instance = MockApp.return_value
            from main import main
            main()

            MockApp.assert_called_once_with(cube_obj_path, algorithm=None)
            instance.run.assert_called_once()
