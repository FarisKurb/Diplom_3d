"""Geodesic approximation via adaptive edge-subdivision refinement.

Produces a higher-quality shortest-path approximation than the plain
visibility graph by densely sampling mesh edges and then *iteratively
refining* around the neighbourhood of the current best path.

Conceptual pipeline:

    1.  Place multiple interior sample points along **every** mesh edge
        (not just midpoints), giving the search denser boundary coverage.
    2.  Build a visibility graph from mesh vertices + edge samples +
        start/end, and run A* to find an initial path.
    3.  Identify mesh vertices that are *close* to the current path, and
        collect the edges incident to those vertices.
    4.  Subdivide those edges at an even higher resolution (adaptive
        refinement), add the new nodes to the graph, and re-solve.
    5.  Repeat step 3-4 for a configurable number of passes.
    6.  Optionally smooth the result.

This converges toward the true geodesic (shortest free-space path around
the polyhedral obstacle) because refinement is concentrated where the
path actually travels.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from core.vector3 import Vector3
from geometry.visibility import is_visible
from graph.graph import Graph
from graph.astar import astar_path_points
from mesh.mesh import Mesh
from mesh.mesh_topology import extract_edges, build_adjacency
from pathfinding.strategy import PathfindingStrategy, PathResult
from pathfinding.path_smoother import smooth_path, compute_path_length


class GeodesicApproxStrategy(PathfindingStrategy):
    """Adaptive edge-subdivision geodesic approximation.

    Parameters:
        edge_divisions: Number of interior sample points placed along each
                        mesh edge in the initial graph.  ``4`` yields points
                        at *t* = 0.2, 0.4, 0.6, 0.8.
        refine_passes:  How many iterative refinement rounds to perform.
                        Each round doubles the subdivision density for edges
                        near the current path.
    """

    def __init__(
        self,
        edge_divisions: int = 4,
        refine_passes: int = 1,
    ) -> None:
        self._edge_divisions = edge_divisions
        self._refine_passes = refine_passes

    @property
    def name(self) -> str:
        return "Geodesic Approx"

    # ── public interface ────────────────────────────────────

    def find_path(
        self,
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
        *,
        smooth: bool = True,
    ) -> PathResult:
        edges = extract_edges(mesh)
        adjacency = build_adjacency(mesh)
        triangles = mesh.triangles

        # Phase 1 — initial dense-edge graph.
        edge_samples = self._subdivide_edges(
            mesh, edges, self._edge_divisions,
        )
        graph, start_id, end_id = self._build_graph(
            start, end, mesh, edge_samples,
        )

        result = astar_path_points(graph, start_id, end_id)
        if result is None:
            return PathResult(
                found=False,
                graph=graph,
                num_samples=len(edge_samples),
                algorithm_name=self.name,
            )

        distance, points = result

        # Phase 2 — iterative refinement.
        for _ in range(self._refine_passes):
            near_verts = self._vertices_near_path(points, mesh, adjacency)
            near_edges = self._edges_near_vertices(near_verts, edges)
            refined = self._subdivide_edges(
                mesh, near_edges, self._edge_divisions * 2,
            )
            all_samples = _merge_samples(edge_samples, refined)

            graph, start_id, end_id = self._build_graph(
                start, end, mesh, all_samples,
            )
            new_result = astar_path_points(graph, start_id, end_id)
            if new_result is not None:
                distance, points = new_result
                edge_samples = all_samples

        # Phase 3 — optional smoothing.
        raw_points: List[Vector3] = []
        if smooth:
            raw_points = list(points)
            points = smooth_path(points, triangles)
            distance = compute_path_length(points)

        return PathResult(
            found=True,
            distance=distance,
            points=points,
            raw_points=raw_points,
            smoothed=smooth,
            graph=graph,
            num_samples=len(edge_samples),
            algorithm_name=self.name,
        )

    # ── internal helpers ────────────────────────────────────

    @staticmethod
    def _subdivide_edges(
        mesh: Mesh,
        edges: List[Tuple[int, int]],
        divisions: int,
    ) -> List[Vector3]:
        """Return interior sample points along each edge.

        For an edge A–B with *divisions* = N, the points are placed at
        ``t = k / (N+1)`` for *k* = 1 … N (excluding the endpoints
        themselves, which are already mesh vertices).
        """
        points: List[Vector3] = []
        seen: set[tuple[float, float, float]] = set()
        for i, j in edges:
            a = mesh.vertices[i]
            b = mesh.vertices[j]
            for k in range(1, divisions + 1):
                t = k / (divisions + 1)
                p = a * (1.0 - t) + b * t
                key = (round(p.x, 9), round(p.y, 9), round(p.z, 9))
                if key not in seen:
                    seen.add(key)
                    points.append(p)
        return points

    @staticmethod
    def _build_graph(
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
        edge_samples: List[Vector3],
    ) -> tuple[Graph, int, int]:
        """Build a visibility graph from vertices + edge samples + endpoints."""
        g = Graph()
        triangles = mesh.triangles

        for v in mesh.vertices:
            g.add_node(v)
        for sp in edge_samples:
            g.add_node(sp)
        start_id = g.add_node(start)
        end_id = g.add_node(end)

        all_ids = list(g.node_ids())
        n = len(all_ids)
        for i in range(n):
            for j in range(i + 1, n):
                a_id = all_ids[i]
                b_id = all_ids[j]
                if is_visible(g.nodes[a_id], g.nodes[b_id], triangles):
                    g.add_edge(a_id, b_id)

        return g, start_id, end_id

    @staticmethod
    def _vertices_near_path(
        path_points: List[Vector3],
        mesh: Mesh,
        adjacency: Dict[int, Set[int]],
    ) -> Set[int]:
        """Return mesh vertex indices that lie close to any path waypoint.

        For each waypoint the closest mesh vertex is found; that vertex
        and its 1-ring topological neighbours are included.
        """
        near: Set[int] = set()
        for pp in path_points:
            best_idx = 0
            best_dist = float("inf")
            for vi, v in enumerate(mesh.vertices):
                d = pp.distance_to(v)
                if d < best_dist:
                    best_dist = d
                    best_idx = vi
            near.add(best_idx)
            near.update(adjacency.get(best_idx, set()))
        return near

    @staticmethod
    def _edges_near_vertices(
        near_verts: Set[int],
        edges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Return edges where at least one endpoint is in *near_verts*."""
        return [
            (i, j) for i, j in edges
            if i in near_verts or j in near_verts
        ]


# ── module-level utility ────────────────────────────────────

def _merge_samples(
    existing: List[Vector3],
    new: List[Vector3],
) -> List[Vector3]:
    """Merge two sample lists, removing positional duplicates."""
    seen: set[tuple[float, float, float]] = set()
    merged: List[Vector3] = []
    for p in existing:
        key = (round(p.x, 9), round(p.y, 9), round(p.z, 9))
        if key not in seen:
            seen.add(key)
            merged.append(p)
    for p in new:
        key = (round(p.x, 9), round(p.y, 9), round(p.z, 9))
        if key not in seen:
            seen.add(key)
            merged.append(p)
    return merged
