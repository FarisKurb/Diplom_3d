"""High-level shortest-path solver.

Orchestrates face sampling → visibility graph → Dijkstra to compute
the shortest path between two surface points on a mesh obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.vector3 import Vector3
from geometry.visibility import is_visible
from graph.graph import Graph
from graph.graph_builder import GraphBuilder
from graph.dijkstra import dijkstra_path_points
from mesh.mesh import Mesh
from pathfinding.face_sampling import sample_mesh_faces
from pathfinding.path_smoother import smooth_path, compute_path_length
from config import FACE_SAMPLE_BARY_STEPS


@dataclass
class PathResult:
    """Result of a shortest-path computation.

    Attributes:
        found:       Whether a path was found.
        distance:    Total path length (``0.0`` if not found).
        points:      Ordered 3-D waypoints from start to end (empty if not found).
        raw_points:  Waypoints before smoothing (empty if smoothing was off).
        smoothed:    Whether the path was smoothed.
        graph:       The visibility graph that was constructed.
        num_samples: Number of face-sample points used.
    """
    found: bool = False
    distance: float = 0.0
    points: List[Vector3] = field(default_factory=list)
    raw_points: List[Vector3] = field(default_factory=list)
    smoothed: bool = False
    graph: Optional[Graph] = None
    num_samples: int = 0


class PathSolver:
    """Computes shortest surface paths between two points around a mesh.

    Usage::

        solver = PathSolver(mesh)
        result = solver.solve(start, end)
        if result.found:
            draw(result.points)

    Args:
        mesh:       The obstacle mesh.
        bary_steps: Barycentric grid resolution for face sampling.
    """

    def __init__(self, mesh: Mesh, bary_steps: int = FACE_SAMPLE_BARY_STEPS) -> None:
        self.mesh = mesh
        self.bary_steps = bary_steps
        self._sample_cache: Optional[List[Vector3]] = None

    @property
    def sample_points(self) -> List[Vector3]:
        """Lazily compute and cache face sample points."""
        if self._sample_cache is None:
            self._sample_cache = sample_mesh_faces(
                self.mesh,
                bary_steps=self.bary_steps,
            )
        return self._sample_cache

    def invalidate_cache(self) -> None:
        """Clear cached sample points (call if the mesh changes)."""
        self._sample_cache = None

    def solve(self, start: Vector3, end: Vector3, *, smooth: bool = True) -> PathResult:
        """Compute the shortest path from *start* to *end*.

        Steps:
            1. Generate face-sample points (cached).
            2. Build a visibility graph (mesh vertices + samples + start/end).
            3. Run Dijkstra to find the shortest path.
            4. Optionally smooth the path by removing redundant waypoints.

        Args:
            start:  Start point (should be on or near the mesh surface).
            end:    End point (should be on or near the mesh surface).
            smooth: If *True* (default), apply greedy shortcutting to
                    remove redundant intermediate waypoints.

        Returns:
            A :class:`PathResult` containing the path and metadata.
        """
        samples = self.sample_points

        builder = GraphBuilder(
            mesh=self.mesh,
            visibility_fn=is_visible,
            sample_points=samples,
        )
        graph, start_id, end_id = builder.build(start, end)

        result = dijkstra_path_points(graph, start_id, end_id)
        if result is None:
            return PathResult(
                found=False,
                graph=graph,
                num_samples=len(samples),
            )

        distance, points = result

        raw_points: List[Vector3] = []
        if smooth:
            raw_points = list(points)
            points = smooth_path(points, self.mesh.triangles)
            distance = compute_path_length(points)

        return PathResult(
            found=True,
            distance=distance,
            points=points,
            raw_points=raw_points,
            smoothed=smooth,
            graph=graph,
            num_samples=len(samples),
        )
