"""A* pathfinding strategy.

Uses the same visibility-graph construction as Dijkstra but runs A*
with Euclidean distance as the heuristic, typically expanding fewer
nodes while still guaranteeing an optimal path.
"""

from __future__ import annotations

from typing import List, Optional

from core.vector3 import Vector3
from geometry.visibility import is_visible
from graph.graph_builder import GraphBuilder
from graph.astar import astar_path_points
from mesh.mesh import Mesh
from pathfinding.strategy import PathfindingStrategy, PathResult
from pathfinding.face_sampling import sample_mesh_faces
from pathfinding.path_smoother import smooth_path, compute_path_length
from config import FACE_SAMPLE_BARY_STEPS


class AStarStrategy(PathfindingStrategy):
    """A* shortest-path on a visibility graph with Euclidean heuristic.

    Pipeline:
        1. Sample faces of the mesh.
        2. Build a visibility graph (mesh vertices + samples + start/end).
        3. Run A* with ``h(n) = euclidean_distance(n, goal)``.
        4. Optionally smooth the result.

    Args:
        bary_steps: Barycentric grid resolution for face sampling.
    """

    def __init__(self, bary_steps: int = FACE_SAMPLE_BARY_STEPS) -> None:
        self._bary_steps = bary_steps
        self._sample_cache: dict[int, List[Vector3]] = {}

    @property
    def name(self) -> str:
        return "A*"

    def _get_samples(self, mesh: Mesh) -> List[Vector3]:
        key = id(mesh)
        if key not in self._sample_cache:
            self._sample_cache[key] = sample_mesh_faces(
                mesh, bary_steps=self._bary_steps,
            )
        return self._sample_cache[key]

    def invalidate_cache(self) -> None:
        """Clear cached sample points."""
        self._sample_cache.clear()

    def find_path(
        self,
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
        *,
        smooth: bool = True,
    ) -> PathResult:
        samples = self._get_samples(mesh)

        builder = GraphBuilder(
            mesh=mesh,
            visibility_fn=is_visible,
            sample_points=samples,
        )
        graph, start_id, end_id = builder.build(start, end)

        result = astar_path_points(graph, start_id, end_id)
        if result is None:
            return PathResult(
                found=False,
                graph=graph,
                num_samples=len(samples),
                algorithm_name=self.name,
            )

        distance, points = result

        raw_points: List[Vector3] = []
        if smooth:
            raw_points = list(points)
            points = smooth_path(points, mesh.triangles)
            distance = compute_path_length(points)

        return PathResult(
            found=True,
            distance=distance,
            points=points,
            raw_points=raw_points,
            smoothed=smooth,
            graph=graph,
            num_samples=len(samples),
            algorithm_name=self.name,
        )
