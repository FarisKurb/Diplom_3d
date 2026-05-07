"""Tests for Stage 6 — Algorithm selection menu.

Covers:
    - AVAILABLE_STRATEGIES list completeness and ordering
    - Application._cycle_algorithm cycles through all strategies
    - A key binding triggers algorithm cycling
    - HUD title updates with new algorithm name
    - PathFinder strategy is swapped correctly
    - Cycling wraps around to first strategy
    - _strategy_index tracks current position
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.vector3 import Vector3
from main import Application, AVAILABLE_STRATEGIES
from pathfinding.chen_han_exact_strategy import ChenHanExactStrategy
from pathfinding.dijkstra_strategy import DijkstraStrategy
from pathfinding.astar_strategy import AStarStrategy
from pathfinding.visibility_graph_strategy import VisibilityGraphStrategy
from pathfinding.geodesic_approx_strategy import GeodesicApproxStrategy


# ── helpers ─────────────────────────────────────────────────

@pytest.fixture
def app(tmp_path):
    """Create an Application with a temp cube mesh (no OpenGL)."""
    from mesh.obj_loader import ensure_default_cube
    obj_path = str(tmp_path / "cube.obj")
    ensure_default_cube(obj_path)
    return Application(mesh_path=obj_path)


# ═══════════════════════════════════════════════════════════
#  AVAILABLE_STRATEGIES registry
# ═══════════════════════════════════════════════════════════

class TestAvailableStrategies:
    def test_contains_all_strategies(self):
        assert len(AVAILABLE_STRATEGIES) == 5

    def test_order(self):
        assert AVAILABLE_STRATEGIES[0] is ChenHanExactStrategy
        assert AVAILABLE_STRATEGIES[1] is DijkstraStrategy
        assert AVAILABLE_STRATEGIES[2] is AStarStrategy
        assert AVAILABLE_STRATEGIES[3] is VisibilityGraphStrategy
        assert AVAILABLE_STRATEGIES[4] is GeodesicApproxStrategy

    def test_all_are_strategy_subclasses(self):
        from pathfinding.strategy import PathfindingStrategy
        for cls in AVAILABLE_STRATEGIES:
            assert issubclass(cls, PathfindingStrategy)


# ═══════════════════════════════════════════════════════════
#  Algorithm cycling
# ═══════════════════════════════════════════════════════════

class TestCycleAlgorithm:
    def test_initial_strategy_is_chen_han_exact(self, app):
        assert app.path_finder.algorithm_name == "Chen-Han Exact"
        assert app._strategy_index == 0

    def test_cycle_to_dijkstra(self, app):
        app._cycle_algorithm()
        assert app.path_finder.algorithm_name == "Dijkstra"
        assert app._strategy_index == 1

    def test_cycle_to_astar(self, app):
        app._cycle_algorithm()
        app._cycle_algorithm()
        assert app.path_finder.algorithm_name == "A*"
        assert app._strategy_index == 2

    def test_cycle_to_visibility_graph(self, app):
        for _ in range(3):
            app._cycle_algorithm()
        assert app.path_finder.algorithm_name == "Visibility Graph"
        assert app._strategy_index == 3

    def test_cycle_to_geodesic(self, app):
        for _ in range(4):
            app._cycle_algorithm()
        assert app.path_finder.algorithm_name == "Geodesic Approx"
        assert app._strategy_index == 4

    def test_cycle_wraps_to_chen_han_exact(self, app):
        for _ in range(5):
            app._cycle_algorithm()
        assert app.path_finder.algorithm_name == "Chen-Han Exact"
        assert app._strategy_index == 0

    def test_full_cycle_names(self, app):
        expected = ["Dijkstra", "A*", "Visibility Graph", "Geodesic Approx", "Chen-Han Exact"]
        actual = []
        for _ in range(5):
            app._cycle_algorithm()
            actual.append(app.path_finder.algorithm_name)
        assert actual == expected


# ═══════════════════════════════════════════════════════════
#  A key binding
# ═══════════════════════════════════════════════════════════

class TestAKeyBinding:
    def test_a_key_calls_cycle(self, app):
        app._cycle_algorithm = MagicMock()

        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_ESCAPE = 256
            mock_glfw.KEY_Q = 81
            mock_glfw.KEY_R = 82
            mock_glfw.KEY_S = 83
            mock_glfw.KEY_H = 72
            mock_glfw.KEY_M = 77
            mock_glfw.KEY_A = 65

            app._on_key(65, 0, 1, 0)

        app._cycle_algorithm.assert_called_once()

    def test_a_key_release_ignored(self, app):
        app._cycle_algorithm = MagicMock()

        with patch("main.glfw") as mock_glfw:
            mock_glfw.PRESS = 1
            mock_glfw.KEY_A = 65

            # action=0 is RELEASE, should be ignored
            app._on_key(65, 0, 0, 0)

        app._cycle_algorithm.assert_not_called()


# ═══════════════════════════════════════════════════════════
#  HUD integration
# ═══════════════════════════════════════════════════════════

class TestHudAlgorithmName:
    def test_title_shows_algorithm(self, app):
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("Chen-Han Exact" in t for t in texts)

    def test_title_updates_after_cycle(self, app):
        app._cycle_algorithm()
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("Dijkstra" in t for t in texts)

    def test_title_shows_all_algorithms_on_cycle(self, app):
        expected_names = ["Dijkstra", "A*", "Visibility Graph", "Geodesic Approx", "Chen-Han Exact"]
        for name in expected_names:
            app._cycle_algorithm()
            app._update_hud()
            texts = [line[0] for line in app.hud_renderer.lines]
            assert any(name in t for t in texts), f"{name} not found in HUD"
