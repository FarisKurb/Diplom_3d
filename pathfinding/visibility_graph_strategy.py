"""Visibility Graph pathfinding strategy.

Implements the classic 3-D visibility graph approach: graph nodes are the
mesh boundary features (vertices and edge midpoints) plus the start/end
query points.  Edges connect mutually visible pairs.  A* is used for the
graph search.

This differs from the Dijkstra / A* strategies which use barycentric
face-interior sampling.  The visibility graph approach relies on the
observation that the shortest path around a convex polyhedral obstacle
must pass through its vertices or edge points (Alexandrov / shortest
path on polyhedra theory).
"""

from __future__ import annotations

from typing import List

from core.vector3 import Vector3
from geometry.visibility import is_visible
from graph.graph import Graph
from graph.astar import astar_path_points
from mesh.mesh import Mesh
from mesh.mesh_topology import extract_edges
from pathfinding.strategy import PathfindingStrategy, PathResult
from pathfinding.path_smoother import smooth_path, compute_path_length


class VisibilityGraphStrategy(PathfindingStrategy):
    """3-D visibility graph: vertices + edge midpoints, visibility edges.

    Pipeline:
        1. Collect mesh vertices and edge midpoints as graph nodes.
        2. Add start and end query points.
        3. Connect all mutually visible node pairs.
        4. Run A* with Euclidean heuristic to find the shortest path.
        5. Optionally smooth the result.

    This strategy produces fewer nodes than face sampling, making it
    faster on dense meshes, while still capturing the key boundary
    features that shortest paths travel through.
    """

    @property
    def name(self) -> str:
        return "Visibility Graph"

    def find_path(
        self,
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
        *,
        smooth: bool = True,
    ) -> PathResult:
        graph, start_id, end_id = self._build_graph(start, end, mesh)

        result = astar_path_points(graph, start_id, end_id)
        if result is None:
            return PathResult(
                found=False,
                graph=graph,
                num_samples=0,
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
            num_samples=0,
            algorithm_name=self.name,
        )

    @staticmethod
    def _build_graph(
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
    ) -> tuple[Graph, int, int]:
        """Build a visibility graph from mesh boundary features."""
        g = Graph()
        triangles = mesh.triangles

        # 1. Add mesh vertices.
        for v in mesh.vertices:
            g.add_node(v)

        # 2. Add edge midpoints.
        edges = extract_edges(mesh)
        for i, j in edges:
            mid = (mesh.vertices[i] + mesh.vertices[j]) * 0.5
            g.add_node(mid)

        # 3. Add start and end.
        start_id = g.add_node(start)
        end_id = g.add_node(end)

        # 4. Connect visible pairs.
        all_ids = list(g.node_ids())
        n = len(all_ids)
        for i in range(n):
            for j in range(i + 1, n):
                a_id = all_ids[i]
                b_id = all_ids[j]
                if is_visible(g.nodes[a_id], g.nodes[b_id], triangles):
                    g.add_edge(a_id, b_id)

        return g, start_id, end_id
