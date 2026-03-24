"""Build a visibility graph from mesh data and query points."""

from __future__ import annotations

from typing import Callable, List, Optional

from core.vector3 import Vector3
from geometry.triangle import Triangle
from graph.graph import Graph
from mesh.mesh import Mesh


class GraphBuilder:
    """Constructs a :class:`Graph` whose nodes are mesh vertices, sampled
    face points, and user-placed query points, and whose edges connect
    mutually visible node pairs.

    The builder is decoupled from the specific visibility test so that
    it can be used with different strategies.

    Args:
        mesh:            The obstacle mesh.
        visibility_fn:   ``(Vector3, Vector3, List[Triangle]) -> bool``
                         returning *True* when two points can see each other
                         (segment does NOT pass through the mesh interior).
        sample_points:   Additional sample points on mesh faces (produced by
                         face sampling).  May be empty.
    """

    def __init__(
        self,
        mesh: Mesh,
        visibility_fn: Callable[[Vector3, Vector3, List[Triangle]], bool],
        sample_points: Optional[List[Vector3]] = None,
    ) -> None:
        self._mesh = mesh
        self._visibility = visibility_fn
        self._sample_points = sample_points or []

    def build(
        self,
        start: Vector3,
        end: Vector3,
    ) -> Graph:
        """Build the full visibility graph including *start* and *end*.

        Returns:
            A :class:`Graph` ready for shortest-path queries.
        """
        g = Graph()
        triangles = self._mesh.triangles

        # 1. Add mesh vertices.
        vertex_ids: List[int] = []
        for v in self._mesh.vertices:
            nid = g.add_node(v)
            vertex_ids.append(nid)

        # 2. Add sampled face points.
        sample_ids: List[int] = []
        for sp in self._sample_points:
            nid = g.add_node(sp)
            sample_ids.append(nid)

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
                pa = g.nodes[a_id]
                pb = g.nodes[b_id]
                if self._visibility(pa, pb, triangles):
                    g.add_edge(a_id, b_id)

        return g, start_id, end_id
