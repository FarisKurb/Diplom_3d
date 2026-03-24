"""Mesh topology utilities — edge extraction, adjacency, and connectivity."""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

from mesh.mesh import Mesh


def extract_edges(mesh: Mesh) -> List[Tuple[int, int]]:
    """Return a list of unique undirected edges as sorted (i, j) pairs.

    Each edge appears exactly once regardless of how many faces share it.
    """
    edge_set: Set[Tuple[int, int]] = set()
    for i0, i1, i2 in mesh.faces:
        for a, b in ((i0, i1), (i1, i2), (i0, i2)):
            edge_set.add((min(a, b), max(a, b)))
    return sorted(edge_set)


def build_adjacency(mesh: Mesh) -> Dict[int, Set[int]]:
    """Build a vertex adjacency map: vertex_index → set of neighbour indices."""
    adj: Dict[int, Set[int]] = {}
    for i0, i1, i2 in mesh.faces:
        for a, b in ((i0, i1), (i1, i2), (i0, i2)):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def faces_sharing_edge(
    mesh: Mesh,
) -> Dict[Tuple[int, int], List[int]]:
    """Map each undirected edge (min, max) to the list of face indices that use it."""
    edge_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi, (i0, i1, i2) in enumerate(mesh.faces):
        for a, b in ((i0, i1), (i1, i2), (i0, i2)):
            key = (min(a, b), max(a, b))
            edge_faces.setdefault(key, []).append(fi)
    return edge_faces


def is_manifold(mesh: Mesh) -> bool:
    """Return *True* if every edge is shared by exactly two faces (closed manifold)."""
    ef = faces_sharing_edge(mesh)
    return all(len(fl) == 2 for fl in ef.values())
