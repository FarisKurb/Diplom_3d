"""Context class for the Strategy Pattern (pathfinding).

:class:`PathFinder` holds a reference to a :class:`PathfindingStrategy`
and delegates path computation to it.  The strategy can be swapped at
runtime via :meth:`set_strategy`.
"""

from __future__ import annotations

from core.vector3 import Vector3
from mesh.mesh import Mesh
from pathfinding.strategy import PathfindingStrategy, PathResult


class PathFinder:
    """Strategy-pattern context that delegates to a concrete algorithm.

    Usage::

        finder = PathFinder(DijkstraStrategy())
        result = finder.compute_path(start, end, mesh)

        # swap algorithm at runtime
        finder.set_strategy(AStarStrategy())
        result = finder.compute_path(start, end, mesh)
    """

    def __init__(self, strategy: PathfindingStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> PathfindingStrategy:
        """The currently active pathfinding strategy."""
        return self._strategy

    @property
    def algorithm_name(self) -> str:
        """Convenience shortcut for ``strategy.name``."""
        return self._strategy.name

    def set_strategy(self, strategy: PathfindingStrategy) -> None:
        """Replace the active strategy."""
        self._strategy = strategy

    def compute_path(
        self,
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
        *,
        smooth: bool = True,
    ) -> PathResult:
        """Delegate path computation to the current strategy.

        Args:
            start:  Start point.
            end:    End point.
            mesh:   The obstacle mesh.
            smooth: Whether to apply path smoothing.

        Returns:
            A :class:`PathResult` produced by the active strategy.
        """
        return self._strategy.find_path(start, end, mesh, smooth=smooth)
