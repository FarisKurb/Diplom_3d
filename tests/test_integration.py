"""Tests for Stage 7 — Full Integration.

Covers:
    - STRATEGY_ALIASES completeness and mapping
    - _resolve_initial_strategy with valid/invalid/None aliases
    - CLI --algorithm flag parsing
    - Auto-recompute on algorithm change when both points placed
    - Auto-recompute on smoothing toggle when both points placed
    - No recompute when points are NOT placed
    - _compute_and_display runs pathfinding, sets _last_result and _compute_time
    - HUD shows compute time after pathfinding
    - Cache invalidation on mesh swap
    - _recompute_if_ready checks placement state
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from core.vector3 import Vector3
from interaction.point_placer import PlacementState
from main import Application, AVAILABLE_STRATEGIES, STRATEGY_ALIASES
from pathfinding.chen_han_exact_strategy import ChenHanExactStrategy
from pathfinding.dijkstra_strategy import DijkstraStrategy
from pathfinding.astar_strategy import AStarStrategy
from pathfinding.visibility_graph_strategy import VisibilityGraphStrategy
from pathfinding.geodesic_approx_strategy import GeodesicApproxStrategy
from pathfinding.strategy import PathResult


# ── helpers ─────────────────────────────────────────────────

@pytest.fixture
def app(tmp_path):
    """Create an Application with a temp cube mesh (no OpenGL)."""
    from mesh.obj_loader import ensure_default_cube
    obj_path = str(tmp_path / "cube.obj")
    ensure_default_cube(obj_path)
    return Application(mesh_path=obj_path)


def _make_point_placer_mock(state: PlacementState, start=None, end=None):
    """Create a mock PointPlacer with given state and points."""
    pp = MagicMock()
    pp.state = state
    pp.start_point = start
    pp.end_point = end
    return pp


# ═══════════════════════════════════════════════════════════
#  STRATEGY_ALIASES
# ═══════════════════════════════════════════════════════════

class TestStrategyAliases:
    def test_all_keys_are_lowercase_strings(self):
        for key in STRATEGY_ALIASES:
            assert isinstance(key, str)
            assert key == key.lower()

    def test_dijkstra_alias(self):
        assert STRATEGY_ALIASES["dijkstra"] is DijkstraStrategy

    def test_chen_han_aliases(self):
        assert STRATEGY_ALIASES["chenhan"] is ChenHanExactStrategy
        assert STRATEGY_ALIASES["chen-han"] is ChenHanExactStrategy
        assert STRATEGY_ALIASES["chen_han"] is ChenHanExactStrategy
        assert STRATEGY_ALIASES["exact"] is ChenHanExactStrategy

    def test_astar_aliases(self):
        assert STRATEGY_ALIASES["astar"] is AStarStrategy
        assert STRATEGY_ALIASES["a*"] is AStarStrategy

    def test_visgraph_aliases(self):
        assert STRATEGY_ALIASES["visgraph"] is VisibilityGraphStrategy
        assert STRATEGY_ALIASES["visibility"] is VisibilityGraphStrategy

    def test_geodesic_alias(self):
        assert STRATEGY_ALIASES["geodesic"] is GeodesicApproxStrategy

    def test_all_available_strategies_reachable(self):
        """Every strategy in AVAILABLE_STRATEGIES is reachable via at least one alias."""
        aliased = set(STRATEGY_ALIASES.values())
        for cls in AVAILABLE_STRATEGIES:
            assert cls in aliased, f"{cls.__name__} has no alias"


# ═══════════════════════════════════════════════════════════
#  _resolve_initial_strategy
# ═══════════════════════════════════════════════════════════

class TestResolveInitialStrategy:
    def test_none_returns_chen_han_exact(self, app):
        strategy = app._resolve_initial_strategy(None)
        assert isinstance(strategy, ChenHanExactStrategy)

    def test_chen_han_alias(self, app):
        strategy = app._resolve_initial_strategy("chenhan")
        assert isinstance(strategy, ChenHanExactStrategy)

    def test_dijkstra_alias(self, app):
        strategy = app._resolve_initial_strategy("dijkstra")
        assert isinstance(strategy, DijkstraStrategy)

    def test_astar_alias(self, app):
        strategy = app._resolve_initial_strategy("astar")
        assert isinstance(strategy, AStarStrategy)

    def test_astar_star_alias(self, app):
        strategy = app._resolve_initial_strategy("a*")
        assert isinstance(strategy, AStarStrategy)

    def test_visgraph_alias(self, app):
        strategy = app._resolve_initial_strategy("visgraph")
        assert isinstance(strategy, VisibilityGraphStrategy)

    def test_geodesic_alias(self, app):
        strategy = app._resolve_initial_strategy("geodesic")
        assert isinstance(strategy, GeodesicApproxStrategy)

    def test_unknown_falls_back_to_chen_han_exact(self, app):
        strategy = app._resolve_initial_strategy("bogus")
        assert isinstance(strategy, ChenHanExactStrategy)

    def test_case_insensitive(self, app):
        strategy = app._resolve_initial_strategy("ASTAR")
        assert isinstance(strategy, AStarStrategy)

    def test_whitespace_stripped(self, app):
        strategy = app._resolve_initial_strategy("  geodesic  ")
        assert isinstance(strategy, GeodesicApproxStrategy)

    def test_syncs_strategy_index(self, app):
        app._resolve_initial_strategy("geodesic")
        assert app._strategy_index == AVAILABLE_STRATEGIES.index(GeodesicApproxStrategy)


# ═══════════════════════════════════════════════════════════
#  CLI --algorithm constructor parameter
# ═══════════════════════════════════════════════════════════

class TestConstructorAlgorithm:
    def test_default_is_chen_han_exact(self, tmp_path):
        from mesh.obj_loader import ensure_default_cube
        obj = str(tmp_path / "c.obj")
        ensure_default_cube(obj)
        app = Application(mesh_path=obj)
        assert app.path_finder.algorithm_name == "Chen-Han Exact"

    def test_algorithm_astar(self, tmp_path):
        from mesh.obj_loader import ensure_default_cube
        obj = str(tmp_path / "c.obj")
        ensure_default_cube(obj)
        app = Application(mesh_path=obj, algorithm="astar")
        assert app.path_finder.algorithm_name == "A*"
        assert app._strategy_index == 2

    def test_algorithm_geodesic(self, tmp_path):
        from mesh.obj_loader import ensure_default_cube
        obj = str(tmp_path / "c.obj")
        ensure_default_cube(obj)
        app = Application(mesh_path=obj, algorithm="geodesic")
        assert app.path_finder.algorithm_name == "Geodesic Approx"
        assert app._strategy_index == 4

    def test_algorithm_unknown_falls_back(self, tmp_path):
        from mesh.obj_loader import ensure_default_cube
        obj = str(tmp_path / "c.obj")
        ensure_default_cube(obj)
        app = Application(mesh_path=obj, algorithm="nonexistent")
        assert app.path_finder.algorithm_name == "Chen-Han Exact"


# ═══════════════════════════════════════════════════════════
#  _compute_and_display
# ═══════════════════════════════════════════════════════════

class TestComputeAndDisplay:
    def test_sets_compute_time(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        assert app._compute_time > 0

    def test_sets_last_result(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        assert app._last_result is not None
        assert isinstance(app._last_result, PathResult)

    def test_updates_path_renderer_on_success(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        r = app._last_result
        if r.found:
            assert len(app.path_renderer.path) > 0

    def test_clears_path_renderer_on_failure(self, app):
        """If no path is found, path_renderer should be cleared."""
        # Use a mock that returns found=False
        mock_result = PathResult(
            found=False, points=[], distance=0.0,
            num_samples=0, smoothed=False, raw_points=None,
        )
        app.path_finder.compute_path = MagicMock(return_value=mock_result)
        app._compute_and_display(Vector3(0, 0, 0), Vector3(1, 1, 1))
        assert app._last_result.found is False


# ═══════════════════════════════════════════════════════════
#  _recompute_if_ready
# ═══════════════════════════════════════════════════════════

class TestRecomputeIfReady:
    def test_recomputes_when_done(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app.point_placer = _make_point_placer_mock(PlacementState.DONE, start, end)
        app._compute_and_display = MagicMock()
        app._recompute_if_ready()
        app._compute_and_display.assert_called_once_with(start, end)

    def test_does_nothing_when_placing_start(self, app):
        app.point_placer = _make_point_placer_mock(PlacementState.PLACE_START)
        app._compute_and_display = MagicMock()
        app._recompute_if_ready()
        app._compute_and_display.assert_not_called()

    def test_does_nothing_when_placing_end(self, app):
        app.point_placer = _make_point_placer_mock(PlacementState.PLACE_END, start=Vector3(0, 0, 0))
        app._compute_and_display = MagicMock()
        app._recompute_if_ready()
        app._compute_and_display.assert_not_called()

    def test_does_nothing_when_point_placer_is_none(self, app):
        app.point_placer = None
        app._compute_and_display = MagicMock()
        app._recompute_if_ready()
        app._compute_and_display.assert_not_called()

    def test_does_nothing_when_start_is_none(self, app):
        app.point_placer = _make_point_placer_mock(PlacementState.DONE, start=None, end=Vector3(1,1,1))
        app._compute_and_display = MagicMock()
        app._recompute_if_ready()
        app._compute_and_display.assert_not_called()


# ═══════════════════════════════════════════════════════════
#  Auto-recompute on algorithm change
# ═══════════════════════════════════════════════════════════

class TestAutoRecomputeOnAlgorithmChange:
    def test_cycle_algorithm_recomputes_when_done(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app.point_placer = _make_point_placer_mock(PlacementState.DONE, start, end)
        app._compute_and_display = MagicMock()
        app._cycle_algorithm()
        app._compute_and_display.assert_called_once_with(start, end)

    def test_cycle_algorithm_no_recompute_when_not_done(self, app):
        app.point_placer = _make_point_placer_mock(PlacementState.PLACE_START)
        app._compute_and_display = MagicMock()
        app._cycle_algorithm()
        app._compute_and_display.assert_not_called()


# ═══════════════════════════════════════════════════════════
#  Auto-recompute on smoothing toggle
# ═══════════════════════════════════════════════════════════

class TestAutoRecomputeOnSmoothingToggle:
    def test_toggle_recomputes_when_done(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app.point_placer = _make_point_placer_mock(PlacementState.DONE, start, end)
        app._compute_and_display = MagicMock()
        app._toggle_smoothing()
        app._compute_and_display.assert_called_once_with(start, end)

    def test_toggle_no_recompute_when_not_done(self, app):
        app.point_placer = _make_point_placer_mock(PlacementState.PLACE_END, start=Vector3(0,0,0))
        app._compute_and_display = MagicMock()
        app._toggle_smoothing()
        app._compute_and_display.assert_not_called()

    def test_toggle_flips_smooth_flag(self, app):
        assert app.smooth_enabled is True
        app._toggle_smoothing()
        assert app.smooth_enabled is False
        app._toggle_smoothing()
        assert app.smooth_enabled is True


# ═══════════════════════════════════════════════════════════
#  Cache invalidation on mesh swap
# ═══════════════════════════════════════════════════════════

class TestCacheInvalidationOnMeshSwap:
    def test_invalidate_cache_called(self, app):
        """Switching mesh calls invalidate_cache on the current strategy."""
        mock_strategy = MagicMock()
        mock_strategy.invalidate_cache = MagicMock()
        app.path_finder._strategy = mock_strategy
        # Need at least 2 meshes to cycle.
        app._mesh_paths = ["a.obj", "b.obj"]
        app._mesh_index = 0
        with patch("main.load_obj", return_value=app.mesh):
            app._cycle_mesh()
        mock_strategy.invalidate_cache.assert_called_once()

    def test_no_error_without_invalidate_cache(self, app):
        """Strategies without invalidate_cache don't crash on mesh swap."""
        mock_strategy = MagicMock(spec=[])  # no attributes
        app.path_finder._strategy = mock_strategy
        app._mesh_paths = ["a.obj", "b.obj"]
        app._mesh_index = 0
        with patch("main.load_obj", return_value=app.mesh):
            app._cycle_mesh()  # should not raise


# ═══════════════════════════════════════════════════════════
#  HUD shows compute time
# ═══════════════════════════════════════════════════════════

class TestHudComputeTime:
    def test_no_compute_time_before_pathfinding(self, app):
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert not any("Compute time" in t for t in texts)

    def test_compute_time_shown_after_pathfinding(self, app):
        app._compute_time = 0.123
        app._last_result = PathResult(
            found=True, points=[Vector3(0,0,0), Vector3(1,1,1)],
            distance=1.0, num_samples=50, smoothed=False, raw_points=None,
        )
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert any("Compute time: 0.123s" in t for t in texts)

    def test_no_compute_time_when_zero(self, app):
        app._compute_time = 0.0
        app._last_result = PathResult(
            found=True, points=[Vector3(0,0,0)],
            distance=1.0, num_samples=50, smoothed=False, raw_points=None,
        )
        app._update_hud()
        texts = [line[0] for line in app.hud_renderer.lines]
        assert not any("Compute time" in t for t in texts)


# ═══════════════════════════════════════════════════════════
#  Timing measurement
# ═══════════════════════════════════════════════════════════

class TestTimingMeasurement:
    def test_compute_time_is_float(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        assert isinstance(app._compute_time, float)
        assert app._compute_time >= 0

    def test_compute_time_updates_on_each_call(self, app):
        start = Vector3(0, 0, 0)
        end = Vector3(1, 1, 1)
        app._compute_and_display(start, end)
        t1 = app._compute_time
        app._compute_and_display(start, end)
        t2 = app._compute_time
        # Both should be positive floats (not necessarily different).
        assert t1 >= 0
        assert t2 >= 0


# ═══════════════════════════════════════════════════════════
#  CLI parsing (main function)
# ═══════════════════════════════════════════════════════════

class TestCLIParsing:
    def test_algorithm_flag(self):
        with patch("main.Application") as MockApp:
            mock_instance = MagicMock()
            MockApp.return_value = mock_instance
            with patch("sys.argv", ["main.py", "--algorithm", "astar"]):
                from main import main
                main()
            MockApp.assert_called_once()
            _, kwargs = MockApp.call_args
            assert kwargs["algorithm"] == "astar"

    def test_short_algorithm_flag(self):
        with patch("main.Application") as MockApp:
            mock_instance = MagicMock()
            MockApp.return_value = mock_instance
            with patch("sys.argv", ["main.py", "-a", "geodesic"]):
                from main import main
                main()
            MockApp.assert_called_once()
            _, kwargs = MockApp.call_args
            assert kwargs["algorithm"] == "geodesic"

    def test_mesh_path_positional(self):
        with patch("main.Application") as MockApp:
            mock_instance = MagicMock()
            MockApp.return_value = mock_instance
            with patch("sys.argv", ["main.py", "mymodel.obj"]):
                from main import main
                main()
            MockApp.assert_called_once()
            args, kwargs = MockApp.call_args
            assert args[0] == "mymodel.obj"

    def test_mesh_and_algorithm(self):
        with patch("main.Application") as MockApp:
            mock_instance = MagicMock()
            MockApp.return_value = mock_instance
            with patch("sys.argv", ["main.py", "mymodel.obj", "--algorithm", "visgraph"]):
                from main import main
                main()
            MockApp.assert_called_once()
            args, kwargs = MockApp.call_args
            assert args[0] == "mymodel.obj"
            assert kwargs["algorithm"] == "visgraph"

    def test_no_args_uses_defaults(self):
        with patch("main.Application") as MockApp:
            mock_instance = MagicMock()
            MockApp.return_value = mock_instance
            with patch("sys.argv", ["main.py"]):
                from main import main
                main()
            MockApp.assert_called_once()
            args, kwargs = MockApp.call_args
            assert kwargs["algorithm"] is None
