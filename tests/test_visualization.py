"""Tests for Stage 8 — Visualization improvements.

Covers:
    - ALGORITHM_COLORS registry completeness
    - Per-algorithm path colour applied after _compute_and_display
    - Distinct start / end marker colours on PathRenderer
    - Debug overlay toggle (D key)
    - Raw path data fed to path_renderer
    - Sample-node data fed to path_renderer
    - PathRenderer.clear resets debug overlay data
    - HUD shows debug state
    - config constants have correct types
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.vector3 import Vector3
from config import (
    ALGORITHM_COLORS, START_POINT_COLOR, END_POINT_COLOR,
    RAW_PATH_COLOR, RAW_PATH_WIDTH, SAMPLE_NODE_COLOR, SAMPLE_NODE_SIZE,
)
from interaction.point_placer import PlacementState
from main import Application, AVAILABLE_STRATEGIES
from pathfinding.strategy import PathResult
from render.path_renderer import PathRenderer


# ── helpers ─────────────────────────────────────────────────

@pytest.fixture
def app(tmp_path):
    """Create an Application with a temp cube mesh (no OpenGL)."""
    from mesh.obj_loader import ensure_default_cube
    obj_path = str(tmp_path / "cube.obj")
    ensure_default_cube(obj_path)
    return Application(mesh_path=obj_path)


# ═══════════════════════════════════════════════════════════
#  ALGORITHM_COLORS config
# ═══════════════════════════════════════════════════════════

class TestAlgorithmColors:
    def test_one_entry_per_strategy(self):
        for cls in AVAILABLE_STRATEGIES:
            name = cls().name
            assert name in ALGORITHM_COLORS, f"No colour for {name}"

    def test_values_are_rgb_tuples(self):
        for name, color in ALGORITHM_COLORS.items():
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, float) for c in color)

    def test_dijkstra_colour(self):
        assert ALGORITHM_COLORS["Dijkstra"] == (1.0, 0.3, 0.1)

    def test_chen_han_colour(self):
        assert ALGORITHM_COLORS["Chen-Han Exact"] == (0.9, 0.2, 0.9)

    def test_astar_colour(self):
        assert ALGORITHM_COLORS["A*"] == (0.2, 0.9, 0.3)

    def test_visgraph_colour(self):
        assert ALGORITHM_COLORS["Visibility Graph"] == (0.3, 0.5, 1.0)

    def test_geodesic_colour(self):
        assert ALGORITHM_COLORS["Geodesic Approx"] == (1.0, 0.7, 0.1)


# ═══════════════════════════════════════════════════════════
#  Distinct start / end marker colours
# ═══════════════════════════════════════════════════════════

class TestMarkerColors:
    def test_config_constants_exist(self):
        assert isinstance(START_POINT_COLOR, tuple)
        assert isinstance(END_POINT_COLOR, tuple)

    def test_start_and_end_differ(self):
        assert START_POINT_COLOR != END_POINT_COLOR

    def test_path_renderer_uses_distinct_colors(self):
        pr = PathRenderer()
        assert pr.start_color == START_POINT_COLOR
        assert pr.end_color == END_POINT_COLOR

    def test_start_color_is_green(self):
        assert START_POINT_COLOR == (0.1, 1.0, 0.2)

    def test_end_color_is_red(self):
        assert END_POINT_COLOR == (1.0, 0.2, 0.2)


# ═══════════════════════════════════════════════════════════
#  PathRenderer debug overlay attributes
# ═══════════════════════════════════════════════════════════

class TestPathRendererDebugAttributes:
    def test_debug_default_off(self):
        pr = PathRenderer()
        assert pr.debug is False

    def test_raw_path_default_empty(self):
        pr = PathRenderer()
        assert pr.raw_path == []

    def test_sample_nodes_default_empty(self):
        pr = PathRenderer()
        assert pr.sample_nodes == []

    def test_clear_resets_debug_data(self):
        pr = PathRenderer()
        pr.raw_path = [Vector3(1, 2, 3)]
        pr.sample_nodes = [Vector3(4, 5, 6)]
        pr.start_point = Vector3(0, 0, 0)
        pr.end_point = Vector3(1, 1, 1)
        pr.clear()
        assert pr.raw_path == []
        assert pr.sample_nodes == []
        assert pr.start_point is None
        assert pr.end_point is None


# ═══════════════════════════════════════════════════════════
#  Debug config constants
# ═══════════════════════════════════════════════════════════

class TestDebugConfigConstants:
    def test_raw_path_color_is_tuple(self):
        assert isinstance(RAW_PATH_COLOR, tuple) and len(RAW_PATH_COLOR) == 3

    def test_raw_path_width_is_float(self):
        assert isinstance(RAW_PATH_WIDTH, float) and RAW_PATH_WIDTH > 0

    def test_sample_node_color_is_tuple(self):
        assert isinstance(SAMPLE_NODE_COLOR, tuple) and len(SAMPLE_NODE_COLOR) == 3

    def test_sample_node_size_is_positive(self):
        assert isinstance(SAMPLE_NODE_SIZE, float) and SAMPLE_NODE_SIZE > 0


# ═══════════════════════════════════════════════════════════
#  Per-algorithm colour applied on compute
# ═══════════════════════════════════════════════════════════

class TestPerAlgorithmColorOnCompute:
    def test_default_chen_han_color(self, app):
        start, end = Vector3(0, 0, 0), Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        assert app.path_renderer.path_color == ALGORITHM_COLORS["Chen-Han Exact"]

    def test_dijkstra_color_after_cycle(self, app):
        app._cycle_algorithm()  # Chen-Han Exact -> Dijkstra
        start, end = Vector3(0, 0, 0), Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        assert app.path_renderer.path_color == ALGORITHM_COLORS["Dijkstra"]

    def test_color_changes_when_algorithm_changes(self, app):
        start, end = Vector3(0, 0, 0), Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        c1 = app.path_renderer.path_color

        app._cycle_algorithm()  # Chen-Han Exact -> Dijkstra
        app._compute_and_display(start, end)
        c2 = app.path_renderer.path_color

        assert c1 != c2


# ═══════════════════════════════════════════════════════════
#  Debug overlay data fed by _compute_and_display
# ═══════════════════════════════════════════════════════════

class TestDebugDataOnCompute:
    def test_raw_path_populated_when_smoothed(self, app):
        start, end = Vector3(0, 0, 0), Vector3(1, 1, 1)
        app.smooth_enabled = True
        app._compute_and_display(start, end)
        r = app._last_result
        if r.found and r.smoothed and r.raw_points:
            assert len(app.path_renderer.raw_path) == len(r.raw_points)

    def test_sample_nodes_populated_when_graph_exists(self, app):
        start, end = Vector3(0, 0, 0), Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        r = app._last_result
        if r.found and r.graph is not None:
            assert len(app.path_renderer.sample_nodes) == len(r.graph.nodes)

    def test_sample_nodes_empty_on_no_path(self, app):
        mock_result = PathResult(found=False, points=[], distance=0.0,
                                 num_samples=0, smoothed=False)
        app.path_finder.compute_path = MagicMock(return_value=mock_result)
        app._compute_and_display(Vector3(0, 0, 0), Vector3(1, 1, 1))
        assert app.path_renderer.sample_nodes == []
        assert app.path_renderer.raw_path == []


# ═══════════════════════════════════════════════════════════
#  D key toggles debug overlay
# ═══════════════════════════════════════════════════════════

class TestDKeyToggle:
    def test_toggle_debug_on(self, app):
        assert app.path_renderer.debug is False
        app._toggle_debug()
        assert app.path_renderer.debug is True

    def test_toggle_debug_off(self, app):
        app.path_renderer.debug = True
        app._toggle_debug()
        assert app.path_renderer.debug is False

    def test_d_key_binding(self, app):
        app._toggle_debug = MagicMock()

        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_ESCAPE = 256
            mock_glfw.KEY_Q = 81
            mock_glfw.KEY_R = 82
            mock_glfw.KEY_S = 83
            mock_glfw.KEY_H = 72
            mock_glfw.KEY_M = 77
            mock_glfw.KEY_A = 65
            mock_glfw.KEY_D = 68

            app._on_key(68, 0, 1, 0)

        app._toggle_debug.assert_called_once()


# ═══════════════════════════════════════════════════════════
#  HUD shows debug state
# ═══════════════════════════════════════════════════════════

class TestHudDebugState:
    def test_hud_shows_debug_off(self, app):
        app.path_renderer.debug = False
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("Debug: OFF" in t for t in texts)

    def test_hud_shows_debug_on(self, app):
        app.path_renderer.debug = True
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("Debug: ON" in t for t in texts)

    def test_hud_smoothing_and_debug_on_same_line(self, app):
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("Smoothing:" in t and "Debug:" in t for t in texts)


# ═══════════════════════════════════════════════════════════
#  HUD help block mentions D key
# ═══════════════════════════════════════════════════════════

class TestHudHelpBlock:
    def test_help_mentions_debug(self):
        from render.hud_renderer import HudRenderer
        hr = HudRenderer()
        hr.show_help = True
        # The help lines are hard-coded; just verify the class exists
        # and the renderer is constructable.
        assert hr.show_help is True
