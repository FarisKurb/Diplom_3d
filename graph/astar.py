"""A* shortest-path algorithm on weighted undirected graphs.

Uses the Euclidean distance from each node to the goal as an admissible
heuristic, guaranteeing optimal paths while typically expanding fewer
nodes than Dijkstra.
"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional, Tuple

from core.vector3 import Vector3
from graph.graph import Graph


def astar(
    graph: Graph,
    start_id: int,
    end_id: int,
    heuristic: Optional[Callable[[Vector3, Vector3], float]] = None,
) -> Optional[Tuple[float, List[int]]]:
    """Run A* from *start_id* to *end_id*.

    Args:
        graph:     The weighted graph.
        start_id:  Source node ID.
        end_id:    Target node ID.
        heuristic: ``h(node_pos, goal_pos) -> float``.  Defaults to
                   Euclidean distance.

    Returns:
        ``(distance, path)`` where *path* is the ordered list of node IDs,
        or ``None`` if no path exists.
    """
    if start_id not in graph.nodes or end_id not in graph.nodes:
        return None

    if start_id == end_id:
        return (0.0, [start_id])

    if heuristic is None:
        heuristic = _euclidean_heuristic

    goal_pos = graph.nodes[end_id]

    g_score: Dict[int, float] = {start_id: 0.0}
    prev: Dict[int, Optional[int]] = {start_id: None}
    visited: set[int] = set()

    # Min-heap entries: (f_score, node_id)
    h_start = heuristic(graph.nodes[start_id], goal_pos)
    heap: List[Tuple[float, int]] = [(h_start, start_id)]

    while heap:
        _f, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        if u == end_id:
            break

        for v, w in graph.neighbours(u):
            if v in visited:
                continue
            tentative_g = g_score[u] + w
            if v not in g_score or tentative_g < g_score[v]:
                g_score[v] = tentative_g
                prev[v] = u
                f = tentative_g + heuristic(graph.nodes[v], goal_pos)
                heapq.heappush(heap, (f, v))

    if end_id not in prev:
        return None

    # Reconstruct path.
    path: List[int] = []
    cur: Optional[int] = end_id
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return (g_score[end_id], path)


def astar_path_points(
    graph: Graph,
    start_id: int,
    end_id: int,
    heuristic: Optional[Callable[[Vector3, Vector3], float]] = None,
) -> Optional[Tuple[float, List[Vector3]]]:
    """Run A* and return the path as 3-D points instead of node IDs.

    Returns:
        ``(distance, points)`` or ``None`` if unreachable.
    """
    result = astar(graph, start_id, end_id, heuristic=heuristic)
    if result is None:
        return None
    distance, id_path = result
    return (distance, [graph.nodes[nid] for nid in id_path])


def _euclidean_heuristic(a: Vector3, b: Vector3) -> float:
    """Admissible heuristic: straight-line Euclidean distance."""
    return a.distance_to(b)
