"""Main application – integrates all modules into an interactive 3-D viewer.

Usage::

    python main.py                              # loads default cube
    python main.py path/to/model.obj            # loads custom mesh
    python main.py --algorithm astar            # start with A*
    python main.py model.obj --algorithm geodesic

Controls:
    Left-click on mesh     Place start point, then end point
    Left-drag              Orbit camera
    Middle/right-drag      Pan camera
    Scroll                 Zoom
    R                      Reset points and path
    S                      Toggle smoothing (auto-recomputes)
    A                      Cycle algorithm (auto-recomputes)
    M                      Cycle mesh
    D                      Toggle debug overlay (sample nodes + raw path)
    H                      Toggle HUD help
    ESC / Q                Quit
"""

from __future__ import annotations

import os
import sys
import time

import glfw  # type: ignore

from config import DEFAULT_MESH_PATH
from mesh.obj_loader import load_obj, ensure_default_cube
from mesh.mesh import Mesh
from mesh.mesh_scanner import scan_mesh_directory, mesh_display_name
from render.renderer import Renderer
from render.mesh_renderer import MeshRenderer
from render.path_renderer import PathRenderer
from render.hud_renderer import HudRenderer
from interaction.point_placer import PointPlacer, PlacementState
from pathfinding.strategy import PathfindingStrategy, PathResult
from pathfinding.path_finder import PathFinder
from pathfinding.dijkstra_strategy import DijkstraStrategy
from pathfinding.astar_strategy import AStarStrategy
from pathfinding.visibility_graph_strategy import VisibilityGraphStrategy
from pathfinding.geodesic_approx_strategy import GeodesicApproxStrategy
from core.vector3 import Vector3
from config import HUD_COLOR, HUD_TITLE_COLOR, ALGORITHM_COLORS


# Ordered list of available strategies (cycled with the A key).
AVAILABLE_STRATEGIES: list[type[PathfindingStrategy]] = [
    DijkstraStrategy,
    AStarStrategy,
    VisibilityGraphStrategy,
    GeodesicApproxStrategy,
]

# Map of CLI-friendly names to strategy classes.
STRATEGY_ALIASES: dict[str, type[PathfindingStrategy]] = {
    "dijkstra": DijkstraStrategy,
    "astar": AStarStrategy,
    "a*": AStarStrategy,
    "visgraph": VisibilityGraphStrategy,
    "visibility": VisibilityGraphStrategy,
    "geodesic": GeodesicApproxStrategy,
}


class Application:
    """Top-level controller that wires together rendering, interaction,
    and pathfinding.

    Attributes:
        mesh:           The loaded obstacle mesh.
        renderer:       GLFW window / OpenGL context manager.
        mesh_renderer:  Draws the mesh (faces + wireframe).
        path_renderer:  Draws path line strip and markers.
        point_placer:   Handles click → point-placement state machine.
        path_finder:    Strategy-pattern context for pathfinding algorithms.
    """

    def __init__(self, mesh_path: str = DEFAULT_MESH_PATH, algorithm: str | None = None) -> None:
        # ── Load mesh ───────────────────────────────────────
        self.mesh = self._load_mesh(mesh_path)

        # ── Renderer / camera ───────────────────────────────
        self.renderer = Renderer()

        # ── Sub-renderers ───────────────────────────────────
        self.mesh_renderer = MeshRenderer(mesh=self.mesh)
        self.path_renderer = PathRenderer()
        self.hud_renderer = HudRenderer()

        # ── Pathfinding (Strategy Pattern) ───────────────────
        self._strategy_index: int = 0
        initial_strategy = self._resolve_initial_strategy(algorithm)
        self.path_finder = PathFinder(initial_strategy)
        self.smooth_enabled: bool = True
        self._last_result: PathResult | None = None
        self._compute_time: float = 0.0

        # ── Mesh catalogue ──────────────────────────────────
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        self._mesh_paths: list[str] = scan_mesh_directory(assets_dir)
        self._mesh_index: int = self._find_mesh_index(mesh_path)

        # ── Point placer (initialised after renderer.init) ──
        self.point_placer: PointPlacer | None = None

    # ── public API ──────────────────────────────────────────

    def run(self) -> None:
        """Initialise the window and enter the main loop."""
        self.renderer.init()

        # Create point placer now that the window exists.
        self.point_placer = PointPlacer(
            mesh=self.mesh,
            camera=self.renderer.camera,
            get_viewport_size=self.renderer.get_framebuffer_size,
            get_cursor_pos=self.renderer.get_cursor_pos,
        )
        self.point_placer.on_both_placed = self._on_both_points_placed

        # Bind HUD viewport size now that the window exists.
        self.hud_renderer.set_viewport_size_fn(self.renderer.get_framebuffer_size)

        # Wire callbacks.
        self.renderer.set_render_callback(self._on_render)
        self.renderer.set_mouse_button_callback(self._on_mouse_button)
        self.renderer.set_key_callback(self._on_key)

        self._print_instructions()
        self.renderer.run()

    # ── callbacks ───────────────────────────────────────────

    def _on_render(self) -> None:
        """Called every frame – draw mesh, path, and HUD."""
        self.mesh_renderer.draw()
        self.path_renderer.draw()
        self._update_hud()
        self.hud_renderer.draw()

    def _on_mouse_button(self, button: int, action: int, mods: int, _extra: int) -> None:
        """Forward mouse clicks to the point placer."""
        if self.point_placer is not None:
            self.point_placer.on_click(button, action, mods)
            # Update path_renderer markers live.
            self.path_renderer.start_point = self.point_placer.start_point
            self.path_renderer.end_point = self.point_placer.end_point

    def _on_key(self, key: int, _scancode: int, action: int, _mods: int) -> None:
        """Handle keyboard shortcuts."""
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(self.renderer.get_window(), True)

        elif key == glfw.KEY_R:
            self._reset()

        elif key == glfw.KEY_S:
            self._toggle_smoothing()

        elif key == glfw.KEY_H:
            self.hud_renderer.show_help = not self.hud_renderer.show_help

        elif key == glfw.KEY_M:
            self._cycle_mesh()

        elif key == glfw.KEY_D:
            self._toggle_debug()

        elif key == glfw.KEY_A:
            self._cycle_algorithm()

    def _on_both_points_placed(self, start: Vector3, end: Vector3) -> None:
        """Called when both start and end points are placed on the mesh."""
        self._compute_and_display(start, end)

    def _compute_and_display(self, start: Vector3, end: Vector3) -> None:
        """Run the current pathfinding algorithm and update the display."""
        algo = self.path_finder.algorithm_name
        print(f"[{algo}] Computing shortest path from {start} to {end} ...")

        t0 = time.perf_counter()
        result = self.path_finder.compute_path(start, end, self.mesh, smooth=self.smooth_enabled)
        self._compute_time = time.perf_counter() - t0

        # Update path colour to match the current algorithm.
        algo_color = ALGORITHM_COLORS.get(algo)
        if algo_color is not None:
            self.path_renderer.path_color = algo_color

        if result.found:
            sm = " (smoothed)" if result.smoothed else ""
            raw_info = ""
            if result.smoothed and result.raw_points:
                raw_info = f", raw waypoints: {len(result.raw_points)}"
            print(f"Path found{sm}! Distance: {result.distance:.4f}, "
                  f"waypoints: {len(result.points)}{raw_info}, "
                  f"samples: {result.num_samples}, "
                  f"time: {self._compute_time:.3f}s")
            self.path_renderer.set_path(result.points)

            # Feed debug overlay data.
            self.path_renderer.raw_path = list(result.raw_points) if result.raw_points else []
            if result.graph is not None:
                self.path_renderer.sample_nodes = list(result.graph.nodes.values())
            else:
                self.path_renderer.sample_nodes = []

            self._last_result = result
        else:
            print("No path found.")
            self.path_renderer.set_path([])
            self.path_renderer.raw_path = []
            self.path_renderer.sample_nodes = []
            self._last_result = result

    # ── helpers ─────────────────────────────────────────────

    def _reset(self) -> None:
        """Clear points and path, restart placement."""
        if self.point_placer is not None:
            self.point_placer.reset()
        self.path_renderer.clear()
        self._last_result = None
        print("Reset – click to place start point.")

    def _toggle_debug(self) -> None:
        """Toggle the debug overlay (sample nodes + raw path)."""
        self.path_renderer.debug = not self.path_renderer.debug
        state = "ON" if self.path_renderer.debug else "OFF"
        print(f"Debug overlay: {state}")

    def _toggle_smoothing(self) -> None:
        """Toggle path smoothing on/off and recompute if points are placed."""
        self.smooth_enabled = not self.smooth_enabled
        state = "ON" if self.smooth_enabled else "OFF"
        print(f"Path smoothing: {state}")
        self._recompute_if_ready()

    def _cycle_algorithm(self) -> None:
        """Switch to the next pathfinding algorithm and recompute if points are placed."""
        self._strategy_index = (self._strategy_index + 1) % len(AVAILABLE_STRATEGIES)
        strategy = AVAILABLE_STRATEGIES[self._strategy_index]()
        self.path_finder.set_strategy(strategy)
        name = self.path_finder.algorithm_name
        print(f"Algorithm switched to: {name}")
        self._recompute_if_ready()

    def _recompute_if_ready(self) -> None:
        """Re-run pathfinding if both points are already placed."""
        if (
            self.point_placer is not None
            and self.point_placer.state == PlacementState.DONE
            and self.point_placer.start_point is not None
            and self.point_placer.end_point is not None
        ):
            self._compute_and_display(
                self.point_placer.start_point,
                self.point_placer.end_point,
            )

    def _resolve_initial_strategy(self, algorithm: str | None) -> PathfindingStrategy:
        """Create the initial strategy, optionally from a CLI alias."""
        if algorithm is not None:
            key = algorithm.lower().strip()
            if key in STRATEGY_ALIASES:
                cls = STRATEGY_ALIASES[key]
                # Sync _strategy_index to match.
                if cls in AVAILABLE_STRATEGIES:
                    self._strategy_index = AVAILABLE_STRATEGIES.index(cls)
                return cls()
            print(f"Unknown algorithm '{algorithm}', falling back to Dijkstra.")
        return AVAILABLE_STRATEGIES[0]()

    @staticmethod
    def _load_mesh(mesh_path: str) -> Mesh:
        """Load a mesh from *mesh_path*, creating the default cube if needed."""
        abs_path = os.path.abspath(mesh_path)
        if mesh_path == DEFAULT_MESH_PATH and not os.path.isfile(abs_path):
            abs_path = ensure_default_cube(mesh_path)
        return load_obj(abs_path)

    def _find_mesh_index(self, mesh_path: str) -> int:
        """Return index of *mesh_path* in the catalogue, or -1."""
        abs_path = os.path.abspath(mesh_path)
        for i, p in enumerate(self._mesh_paths):
            if os.path.normcase(p) == os.path.normcase(abs_path):
                return i
        return max(0, 0)  # default to first entry

    @property
    def mesh_name(self) -> str:
        """Human-readable name of the currently loaded mesh."""
        if 0 <= self._mesh_index < len(self._mesh_paths):
            return mesh_display_name(self._mesh_paths[self._mesh_index])
        return "unknown"

    def _cycle_mesh(self) -> None:
        """Switch to the next mesh in the assets catalogue."""
        if len(self._mesh_paths) < 2:
            print("No other meshes available in assets/.")
            return

        self._mesh_index = (self._mesh_index + 1) % len(self._mesh_paths)
        new_path = self._mesh_paths[self._mesh_index]
        name = mesh_display_name(new_path)
        print(f"Loading mesh: {name} ...")
        self.mesh = load_obj(new_path)

        # Update sub-systems that hold a reference to the mesh.
        self.mesh_renderer.mesh = self.mesh
        if self.point_placer is not None:
            self.point_placer.mesh = self.mesh

        # Invalidate any strategy-level sample caches.
        strategy = self.path_finder.strategy
        if hasattr(strategy, "invalidate_cache"):
            strategy.invalidate_cache()

        self._reset()
        print(f"Mesh switched to: {name}")

    def _update_hud(self) -> None:
        """Build the HUD status lines from current application state."""
        lines = []

        # Title with algorithm name and mesh name.
        algo = self.path_finder.algorithm_name
        mesh_label = self.mesh_name
        lines.append((f"3D Shortest Path  [{algo}]  Mesh: {mesh_label}", HUD_TITLE_COLOR))

        # Placement state.
        if self.point_placer is not None:
            state = self.point_placer.state
            if state == PlacementState.PLACE_START:
                lines.append(("Click mesh to place START point", HUD_COLOR))
            elif state == PlacementState.PLACE_END:
                lines.append(("Click mesh to place END point", HUD_COLOR))
            elif state == PlacementState.DONE:
                lines.append(("Both points placed", HUD_COLOR))

        # Smoothing / debug state.
        sm = "ON" if self.smooth_enabled else "OFF"
        dbg = "ON" if self.path_renderer.debug else "OFF"
        lines.append((f"Smoothing: {sm}  |  Debug: {dbg}", HUD_COLOR))

        # Path result.
        r = self._last_result
        if r is not None:
            if r.found:
                lines.append((f"Distance: {r.distance:.4f}", HUD_COLOR))
                lines.append((f"Waypoints: {len(r.points)}", HUD_COLOR))
                if r.smoothed and r.raw_points:
                    lines.append((f"Raw waypoints: {len(r.raw_points)}", HUD_COLOR))
                lines.append((f"Samples: {r.num_samples}", HUD_COLOR))
                if self._compute_time > 0:
                    lines.append((f"Compute time: {self._compute_time:.3f}s", HUD_COLOR))
            else:
                lines.append(("No path found", (1.0, 0.3, 0.3)))

        self.hud_renderer.set_lines(lines)

    @staticmethod
    def _print_instructions() -> None:
        print("──────────────────────────────────────")
        print("  Left-click on mesh  → place points")
        print("  Left-drag           → orbit camera")
        print("  Middle/right-drag   → pan camera")
        print("  Scroll              → zoom")
        print("  R                   → reset")
        print("  S                   → toggle smoothing")
        print("  D                   → toggle debug overlay")
        print("  A                   → cycle algorithm")
        print("  M                   → cycle mesh")
        print("  H                   → toggle HUD help")
        print("  ESC / Q             → quit")
        print("──────────────────────────────────────")


def main() -> None:
    """Entry point – parse CLI args and launch the application."""
    mesh_path = DEFAULT_MESH_PATH
    algorithm: str | None = None

    # Simple argument parsing (positional .obj path + optional --algorithm).
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--algorithm", "-a") and i + 1 < len(args):
            algorithm = args[i + 1]
            i += 2
        else:
            mesh_path = args[i]
            i += 1

    app = Application(mesh_path, algorithm=algorithm)
    app.run()


if __name__ == "__main__":
    main()
