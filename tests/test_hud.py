"""Tests for Stage 12 — HUD overlay and application integration.

Covers:
    - HudRenderer construction and defaults
    - set_lines / set_viewport_size_fn
    - show_help toggle
    - draw does nothing without viewport_size_fn
    - _ensure_glut initialisation guard
    - Application._update_hud builds correct status lines
    - H key toggles HUD help
    - _last_result tracking across solve and reset
    - HUD config constants exist
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
from typing import List, Tuple

import pytest

from core.vector3 import Vector3
from pathfinding.path_solver import PathResult
from interaction.point_placer import PlacementState


# ── helpers ─────────────────────────────────────────────────

@pytest.fixture
def app(tmp_path):
    """Create an Application with a temp cube mesh (no OpenGL)."""
    from mesh.obj_loader import ensure_default_cube
    from main import Application
    obj_path = str(tmp_path / "cube.obj")
    ensure_default_cube(obj_path)
    return Application(mesh_path=obj_path)


# ═══════════════════════════════════════════════════════════
#  HudRenderer unit tests
# ═══════════════════════════════════════════════════════════

class TestHudRendererConstruction:
    def test_defaults(self):
        from render.hud_renderer import HudRenderer
        hud = HudRenderer()
        assert hud.lines == []
        assert hud.show_help is True
        assert hud._viewport_size_fn is None

    def test_set_lines(self):
        from render.hud_renderer import HudRenderer
        hud = HudRenderer()
        lines = [("test", (1.0, 1.0, 1.0))]
        hud.set_lines(lines)
        assert hud.lines == lines
        # Should be a copy, not the same list
        assert hud.lines is not lines

    def test_set_viewport_size_fn(self):
        from render.hud_renderer import HudRenderer
        hud = HudRenderer()
        fn = MagicMock(return_value=(800, 600))
        hud.set_viewport_size_fn(fn)
        assert hud._viewport_size_fn is fn

    def test_draw_without_viewport_fn_does_nothing(self):
        from render.hud_renderer import HudRenderer
        hud = HudRenderer()
        hud.set_lines([("hello", (1, 1, 1))])
        # Should not raise
        hud.draw()

    def test_draw_with_zero_size_does_nothing(self):
        from render.hud_renderer import HudRenderer
        hud = HudRenderer()
        hud.set_viewport_size_fn(lambda: (0, 0))
        hud.set_lines([("hello", (1, 1, 1))])
        # Should not raise
        hud.draw()

    def test_show_help_toggle(self):
        from render.hud_renderer import HudRenderer
        hud = HudRenderer()
        assert hud.show_help is True
        hud.show_help = False
        assert hud.show_help is False


# ═══════════════════════════════════════════════════════════
#  _ensure_glut
# ═══════════════════════════════════════════════════════════

class TestEnsureGlut:
    def test_glut_init_called_once(self):
        import render.hud_renderer as hud_mod
        old_val = hud_mod._glut_initialised
        hud_mod._glut_initialised = False
        try:
            with patch.object(hud_mod, "glutInit") as mock_init:
                hud_mod._ensure_glut()
                mock_init.assert_called_once()
                assert hud_mod._glut_initialised is True

                # Second call should not re-init.
                hud_mod._ensure_glut()
                mock_init.assert_called_once()
        finally:
            hud_mod._glut_initialised = old_val

    def test_already_initialised_skips(self):
        import render.hud_renderer as hud_mod
        old_val = hud_mod._glut_initialised
        hud_mod._glut_initialised = True
        try:
            with patch.object(hud_mod, "glutInit") as mock_init:
                hud_mod._ensure_glut()
                mock_init.assert_not_called()
        finally:
            hud_mod._glut_initialised = old_val


# ═══════════════════════════════════════════════════════════
#  Application HUD integration
# ═══════════════════════════════════════════════════════════

class TestApplicationHud:
    def test_hud_renderer_created(self, app):
        assert app.hud_renderer is not None

    def test_last_result_initially_none(self, app):
        assert app._last_result is None

    def test_on_render_calls_hud_draw(self, app):
        app.mesh_renderer.draw = MagicMock()
        app.path_renderer.draw = MagicMock()
        app.hud_renderer.draw = MagicMock()
        app._update_hud = MagicMock()

        app._on_render()

        app.mesh_renderer.draw.assert_called_once()
        app.path_renderer.draw.assert_called_once()
        app._update_hud.assert_called_once()
        app.hud_renderer.draw.assert_called_once()


class TestHudStatusLines:
    def test_initial_state_place_start(self, app):
        """Before run(), point_placer is None, so no placement line."""
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("3D Shortest Path" in t for t in texts)
        assert any("Smoothing: ON" in t for t in texts)

    def test_place_start_state(self, app):
        app.point_placer = MagicMock()
        app.point_placer.state = PlacementState.PLACE_START

        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert "Click mesh to place START point" in texts

    def test_place_end_state(self, app):
        app.point_placer = MagicMock()
        app.point_placer.state = PlacementState.PLACE_END

        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert "Click mesh to place END point" in texts

    def test_done_state(self, app):
        app.point_placer = MagicMock()
        app.point_placer.state = PlacementState.DONE

        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert "Both points placed" in texts

    def test_path_found_lines(self, app):
        app.point_placer = MagicMock()
        app.point_placer.state = PlacementState.DONE
        app._last_result = PathResult(
            found=True,
            distance=3.1416,
            points=[Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0)],
            raw_points=[Vector3(0, 0, 0), Vector3(0.5, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0)],
            smoothed=True,
            num_samples=42,
        )

        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert "Distance: 3.1416" in texts
        assert "Waypoints: 3" in texts
        assert "Raw waypoints: 4" in texts
        assert "Samples: 42" in texts

    def test_no_path_found_line(self, app):
        app.point_placer = MagicMock()
        app.point_placer.state = PlacementState.DONE
        app._last_result = PathResult(found=False, num_samples=10)

        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert "No path found" in texts

    def test_smoothing_off(self, app):
        app.smooth_enabled = False
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("Smoothing: OFF" in t for t in texts)

    def test_no_raw_waypoints_when_unsmoothed(self, app):
        app._last_result = PathResult(
            found=True,
            distance=1.0,
            points=[Vector3(0, 0, 0), Vector3(1, 0, 0)],
            raw_points=[],
            smoothed=False,
            num_samples=5,
        )

        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert not any("Raw waypoints" in t for t in texts)


# ═══════════════════════════════════════════════════════════
#  H key toggle
# ═══════════════════════════════════════════════════════════

class TestHKeyToggle:
    def test_h_key_toggles_help(self, app):
        assert app.hud_renderer.show_help is True

        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_ESCAPE = 256
            mock_glfw.KEY_Q = 81
            mock_glfw.KEY_R = 82
            mock_glfw.KEY_S = 83
            mock_glfw.KEY_H = 72

            app._on_key(72, 0, 1, 0)
            assert app.hud_renderer.show_help is False

            app._on_key(72, 0, 1, 0)
            assert app.hud_renderer.show_help is True


# ═══════════════════════════════════════════════════════════
#  _last_result tracking
# ═══════════════════════════════════════════════════════════

class TestLastResultTracking:
    def test_set_on_path_found(self, app):
        fake = PathResult(found=True, distance=2.0,
                          points=[Vector3(0, 0, 0), Vector3(1, 0, 0)])
        app.path_finder.compute_path = MagicMock(return_value=fake)
        app._on_both_points_placed(Vector3(0, 0, 0), Vector3(1, 0, 0))
        assert app._last_result is fake

    def test_set_on_no_path(self, app):
        fake = PathResult(found=False, num_samples=5)
        app.path_finder.compute_path = MagicMock(return_value=fake)
        app._on_both_points_placed(Vector3(0, 0, 0), Vector3(1, 0, 0))
        assert app._last_result is fake

    def test_reset_clears_last_result(self, app):
        app._last_result = PathResult(found=True)
        app.point_placer = MagicMock()
        app._reset()
        assert app._last_result is None


# ═══════════════════════════════════════════════════════════
#  HUD config
# ═══════════════════════════════════════════════════════════

class TestHudConfig:
    def test_hud_color_exists(self):
        from config import HUD_COLOR
        assert len(HUD_COLOR) == 3
        assert all(0 <= c <= 1 for c in HUD_COLOR)

    def test_hud_title_color_exists(self):
        from config import HUD_TITLE_COLOR
        assert len(HUD_TITLE_COLOR) == 3
        assert all(0 <= c <= 1 for c in HUD_TITLE_COLOR)
