"""Weighted undirected graph for shortest-path computation."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from core.vector3 import Vector3


class Graph:
    """Weighted undirected graph where each node stores a 3-D position.

    Nodes are identified by integer IDs.  Edges are stored as adjacency
    lists with weights equal to Euclidean distance (or an explicit value).

    Attributes:
        nodes: Maps node ID → 3-D position.
        adj:   Maps node ID → list of (neighbour_id, weight).
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, Vector3] = {}
        self.adj: Dict[int, List[Tuple[int, float]]] = {}
        self._next_id: int = 0

    # ── node management ─────────────────────────────────────

    def add_node(self, position: Vector3, node_id: Optional[int] = None) -> int:
        """Add a node and return its ID.

        Args:
            position: 3-D position of the node.
            node_id:  Explicit ID to use.  If ``None``, an auto-incrementing
                      ID is assigned.

        Returns:
            The ID of the newly added node.
        """
        if node_id is None:
            node_id = self._next_id
        self.nodes[node_id] = position
        self.adj.setdefault(node_id, [])
        if node_id >= self._next_id:
            self._next_id = node_id + 1
        return node_id

    def has_node(self, node_id: int) -> bool:
        return node_id in self.nodes

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of undirected edges (each stored twice in adj)."""
        return sum(len(nbrs) for nbrs in self.adj.values()) // 2

    # ── edge management ─────────────────────────────────────

    def add_edge(self, a: int, b: int, weight: Optional[float] = None) -> None:
        """Add an undirected edge between nodes *a* and *b*.

        If *weight* is ``None`` the Euclidean distance between the two
        node positions is used.

        Does nothing if the edge already exists.
        """
        if a == b:
            return
        # Check for duplicate.
        for nbr, _ in self.adj[a]:
            if nbr == b:
                return
        if weight is None:
            weight = self.nodes[a].distance_to(self.nodes[b])
        self.adj[a].append((b, weight))
        self.adj[b].append((a, weight))

    def neighbours(self, node_id: int) -> List[Tuple[int, float]]:
        """Return the list of ``(neighbour_id, weight)`` for *node_id*."""
        return self.adj.get(node_id, [])

    def node_ids(self) -> Set[int]:
        """Return the set of all node IDs."""
        return set(self.nodes.keys())

    # ── utilities ───────────────────────────────────────────

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self.nodes.clear()
        self.adj.clear()
        self._next_id = 0

    def __repr__(self) -> str:
        return f"Graph(nodes={self.num_nodes}, edges={self.num_edges})"
