"""Dijkstra's shortest-path algorithm on weighted undirected graphs."""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from core.vector3 import Vector3
from graph.graph import Graph


def dijkstra(
    graph: Graph,
    start_id: int,
    end_id: int,
) -> Optional[Tuple[float, List[int]]]:
    """Run Dijkstra's algorithm from *start_id* to *end_id*.

    Args:
        graph:    The weighted graph.
        start_id: Source node ID.
        end_id:   Target node ID.

    Returns:
        ``(distance, path)`` where *path* is the ordered list of node IDs
        from *start_id* to *end_id*, or ``None`` if no path exists.
    """
    if start_id not in graph.nodes or end_id not in graph.nodes:
        return None

    if start_id == end_id:
        return (0.0, [start_id])

    dist: Dict[int, float] = {start_id: 0.0}
    prev: Dict[int, Optional[int]] = {start_id: None}
    visited: set[int] = set()

    # Min-heap entries: (distance, node_id)
    heap: List[Tuple[float, int]] = [(0.0, start_id)]

    while heap:
        d, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        if u == end_id:
            break

        for v, w in graph.neighbours(u):
            if v in visited:
                continue
            new_dist = d + w
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    if end_id not in prev:
        return None

    # Reconstruct path.
    path: List[int] = []
    cur: Optional[int] = end_id
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return (dist[end_id], path)


def dijkstra_path_points(
    graph: Graph,
    start_id: int,
    end_id: int,
) -> Optional[Tuple[float, List[Vector3]]]:
    """Run Dijkstra and return the path as 3-D points instead of node IDs.

    Returns:
        ``(distance, points)`` or ``None`` if unreachable.
    """
    result = dijkstra(graph, start_id, end_id)
    if result is None:
        return None
    distance, id_path = result
    return (distance, [graph.nodes[nid] for nid in id_path])
