"""Abstract base class for pathfinding strategies (Strategy Pattern).

Every concrete algorithm (Dijkstra, A*, visibility graph, geodesic
approximation, …) implements :class:`PathfindingStrategy` so that the
application can swap algorithms at runtime without changing calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from core.vector3 import Vector3
from graph.graph import Graph
from mesh.mesh import Mesh


@dataclass
class PathResult:
    """Result returned by every pathfinding strategy.

    Attributes:
        found:          Whether a valid path was discovered.
        distance:       Total path length (``0.0`` if not found).
        points:         Ordered 3-D waypoints from start to end.
        raw_points:     Waypoints before smoothing (empty when smoothing is off).
        smoothed:       Whether the result has been smoothed.
        graph:          The graph that was constructed (if applicable).
        num_samples:    Number of face-sample points used.
        algorithm_name: Human-readable name of the algorithm that produced
                        this result.
    """
    found: bool = False
    distance: float = 0.0
    points: List[Vector3] = field(default_factory=list)
    raw_points: List[Vector3] = field(default_factory=list)
    smoothed: bool = False
    graph: Optional[Graph] = None
    num_samples: int = 0
    algorithm_name: str = ""


class PathfindingStrategy(ABC):
    """Interface that every pathfinding algorithm must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the algorithm (e.g. ``"Dijkstra"``)."""

    @abstractmethod
    def find_path(
        self,
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
        *,
        smooth: bool = True,
    ) -> PathResult:
        """Compute the shortest path from *start* to *end* around *mesh*.

        Args:
            start:  Start point.
            end:    End point.
            mesh:   The obstacle mesh.
            smooth: Whether to apply post-processing path smoothing.

        Returns:
            A :class:`PathResult` with the computed path and metadata.
        """
