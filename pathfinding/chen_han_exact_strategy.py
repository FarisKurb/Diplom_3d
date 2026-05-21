"""Exact Chen-Han window propagation for one convex polyhedral obstacle.

For one convex obstacle the free-space shortest path is either the direct
segment or a geodesic on the convex hull of ``vertices(P) + start + end``.
This module implements the second case with Chen-Han style windows:

* a window is an interval on an edge of an unfolded triangle;
* it stores the pseudo-source image, distance to that pseudo-source, unfolded
  face sequence, and a predecessor window;
* windows are propagated through adjacent triangles by unfolding the next face;
* dominated windows on the same directed edge are removed;
* the final path is recovered by following predecessor windows backward.

No visibility graph, Dijkstra/A*, vertex/edge search, or sampling is used.
"""

from __future__ import annotations

from collections import deque
import itertools
import math
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from core.math_utils import EPSILON, barycentric_coordinates
from core.vector3 import Vector3
from geometry.intersection import segment_intersects_mesh
from mesh.mesh import Mesh
from pathfinding.strategy import PathResult, PathfindingStrategy


Point2 = Tuple[float, float]
Face = Tuple[int, int, int]
Edge = Tuple[int, int]


@dataclass
class _Window:
    """Chen-Han propagation window on one edge of one unfolded face."""

    face_id: int
    edge: Optional[Edge]
    left: float
    right: float
    pseudo_source: Point2
    source_distance: float
    face_coords: Dict[int, Point2]
    sequence: List[int]
    predecessor: Optional["_Window"] = None
    id: int = field(default=0)


@dataclass(frozen=True)
class _TargetCandidate:
    window: _Window
    distance: float


class ChenHanExactStrategy(PathfindingStrategy):
    """Exact surface shortest path strategy for one convex obstacle."""

    def __init__(self, *, max_windows: int = 10000) -> None:
        self.max_windows = max_windows
        self._next_window_id = 1

    @property
    def name(self) -> str:
        return "Chen-Han Exact"

    def find_path(
        self,
        start: Vector3,
        end: Vector3,
        mesh: Mesh,
        *,
        smooth: bool = True,
    ) -> PathResult:
        if start.approx_equal(end):
            return PathResult(
                found=True,
                distance=0.0,
                points=[start],
                algorithm_name=self.name,
            )

        if not segment_intersects_mesh(start, end, mesh.triangles):
            return PathResult(
                found=True,
                distance=start.distance_to(end),
                points=[start, end],
                algorithm_name=self.name,
            )

        hull_vertices, hull_faces = self._build_convex_hull([*mesh.vertices, start, end])
        start_id = self._find_vertex(hull_vertices, start)
        end_id = self._find_vertex(hull_vertices, end)
        if start_id is None or end_id is None:
            return PathResult(found=False, algorithm_name=self.name)

        candidate = self._chen_han_shortest_path(
            hull_vertices,
            hull_faces,
            start_id,
            end_id,
        )
        if candidate is None:
            return PathResult(found=False, algorithm_name=self.name)

        points = self._recover_path(candidate.window, end_id, hull_vertices, hull_faces)
        distance = self._path_length(points)
        return PathResult(
            found=True,
            distance=distance,
            points=points,
            raw_points=list(points),
            smoothed=False,
            graph=None,
            num_samples=0,
            algorithm_name=self.name,
        )

    @staticmethod
    def _find_vertex(vertices: Sequence[Vector3], point: Vector3) -> Optional[int]:
        for i, vertex in enumerate(vertices):
            if vertex.approx_equal(point, eps=1e-7):
                return i
        return None

    def _build_convex_hull(self, points: Sequence[Vector3]) -> Tuple[List[Vector3], List[Face]]:
        vertices = self._unique_points(points)
        if len(vertices) < 4:
            if len(vertices) == 3:
                return vertices, [(0, 1, 2)]
            return vertices, []

        center = self._centroid(vertices)
        plane_groups: Dict[Tuple[int, int, int, int], set[int]] = {}

        for i, j, k in itertools.combinations(range(len(vertices)), 3):
            normal = (vertices[j] - vertices[i]).cross(vertices[k] - vertices[i])
            if normal.length() < EPSILON:
                continue

            signs: list[float] = []
            for m, point in enumerate(vertices):
                if m in (i, j, k):
                    continue
                signs.append((point - vertices[i]).dot(normal))

            has_pos = any(s > 1e-8 for s in signs)
            has_neg = any(s < -1e-8 for s in signs)
            if has_pos and has_neg:
                continue

            outward = normal
            if (center - vertices[i]).dot(outward) > 0.0:
                outward = -outward

            coplanar = {
                idx
                for idx, point in enumerate(vertices)
                if abs((point - vertices[i]).dot(outward)) < 1e-7
            }
            key = self._plane_key(outward, vertices[i])
            plane_groups.setdefault(key, set()).update(coplanar)

        faces: list[Face] = []
        seen: set[Face] = set()
        for indices in plane_groups.values():
            for face in self._triangulate_coplanar_face(vertices, sorted(indices), center):
                if self._face_area(vertices, face) < EPSILON:
                    continue
                oriented = self._orient_outward(vertices, face, center)
                key = tuple(sorted(oriented))
                if key not in seen:
                    seen.add(key)
                    faces.append(oriented)

        return vertices, faces

    @staticmethod
    def _unique_points(points: Sequence[Vector3]) -> List[Vector3]:
        result: list[Vector3] = []
        for point in points:
            if not any(point.approx_equal(existing, eps=1e-8) for existing in result):
                result.append(point)
        return result

    @staticmethod
    def _centroid(points: Sequence[Vector3]) -> Vector3:
        total = Vector3.zero()
        for point in points:
            total = total + point
        return total / float(len(points))

    @staticmethod
    def _plane_key(normal: Vector3, point: Vector3) -> Tuple[int, int, int, int]:
        n = normal.normalized()
        d = -n.dot(point)
        scale = 1_000_000
        return (
            round(n.x * scale),
            round(n.y * scale),
            round(n.z * scale),
            round(d * scale),
        )

    def _triangulate_coplanar_face(
        self,
        vertices: Sequence[Vector3],
        indices: Sequence[int],
        center: Vector3,
    ) -> List[Face]:
        if len(indices) < 3:
            return []

        face_center = self._centroid([vertices[i] for i in indices])
        normal: Optional[Vector3] = None
        for i, j, k in itertools.combinations(indices, 3):
            candidate = (vertices[j] - vertices[i]).cross(vertices[k] - vertices[i])
            if candidate.length() >= EPSILON:
                normal = candidate.normalized()
                break
        if normal is None:
            return []
        if (center - face_center).dot(normal) > 0.0:
            normal = -normal

        axis_u: Optional[Vector3] = None
        for idx in indices:
            rel = vertices[idx] - face_center
            if rel.length() >= EPSILON:
                axis_u = rel.normalized()
                break
        if axis_u is None:
            return []
        axis_v = normal.cross(axis_u).normalized()
        coords = {
            idx: (
                (vertices[idx] - face_center).dot(axis_u),
                (vertices[idx] - face_center).dot(axis_v),
            )
            for idx in indices
        }

        boundary = self._convex_hull_2d(indices, coords)
        if len(boundary) < 3:
            return []

        triangles: list[Face] = [
            (boundary[0], a, b)
            for a, b in zip(boundary[1:-1], boundary[2:])
        ]
        for idx in indices:
            if idx not in boundary:
                triangles = self._insert_point_in_2d_triangulation(triangles, idx, coords)

        return [self._orient_2d_face(face, coords) for face in triangles]

    @classmethod
    def _convex_hull_2d(
        cls,
        indices: Sequence[int],
        coords: Dict[int, Point2],
    ) -> List[int]:
        ordered = sorted(indices, key=lambda idx: (coords[idx][0], coords[idx][1], idx))

        def build_half(points: Sequence[int]) -> list[int]:
            half: list[int] = []
            for idx in points:
                while len(half) >= 2:
                    a, b = half[-2], half[-1]
                    if cls._orient_2d(coords[a], coords[b], coords[idx]) > 1e-10:
                        break
                    half.pop()
                half.append(idx)
            return half

        lower = build_half(ordered)
        upper = build_half(list(reversed(ordered)))
        return lower[:-1] + upper[:-1]

    def _insert_point_in_2d_triangulation(
        self,
        triangles: Sequence[Face],
        point_id: int,
        coords: Dict[int, Point2],
    ) -> List[Face]:
        edge_hit: Optional[Edge] = None
        for tri in triangles:
            for edge in self._face_edges(tri):
                if self._point_on_segment_2d(
                    coords[point_id],
                    coords[edge[0]],
                    coords[edge[1]],
                ):
                    edge_hit = tuple(sorted(edge))
                    break
            if edge_hit is not None:
                break

        if edge_hit is not None:
            result: list[Face] = []
            for tri in triangles:
                if edge_hit[0] in tri and edge_hit[1] in tri:
                    split = self._split_triangle_edge_2d(tri, edge_hit, point_id, coords)
                    result.extend(split)
                else:
                    result.append(tri)
            return self._filter_degenerate_2d(result, coords)

        for tri in triangles:
            if self._point_in_triangle_2d(coords[point_id], tri, coords):
                result = [old for old in triangles if old != tri]
                a, b, c = tri
                result.extend(
                    [
                        self._orient_2d_face((a, b, point_id), coords),
                        self._orient_2d_face((b, c, point_id), coords),
                        self._orient_2d_face((c, a, point_id), coords),
                    ]
                )
                return self._filter_degenerate_2d(result, coords)

        return list(triangles)

    def _split_triangle_edge_2d(
        self,
        tri: Face,
        edge: Edge,
        point_id: int,
        coords: Dict[int, Point2],
    ) -> List[Face]:
        ordered_edge: Optional[Edge] = None
        for i, candidate in enumerate(self._face_edges(tri)):
            if tuple(sorted(candidate)) == edge:
                ordered_edge = candidate
                opposite = tri[(i + 2) % 3]
                break
        if ordered_edge is None:
            return [tri]

        a, b = ordered_edge
        return [
            self._orient_2d_face((a, point_id, opposite), coords),
            self._orient_2d_face((point_id, b, opposite), coords),
        ]

    @staticmethod
    def _orient_2d_face(face: Face, coords: Dict[int, Point2]) -> Face:
        a, b, c = face
        if ChenHanExactStrategy._orient_2d(coords[a], coords[b], coords[c]) < 0.0:
            return (a, c, b)
        return face

    @staticmethod
    def _filter_degenerate_2d(
        triangles: Sequence[Face],
        coords: Dict[int, Point2],
    ) -> List[Face]:
        result: list[Face] = []
        seen: set[Face] = set()
        for face in triangles:
            a, b, c = face
            if len({a, b, c}) < 3:
                continue
            if abs(ChenHanExactStrategy._orient_2d(coords[a], coords[b], coords[c])) < 1e-10:
                continue
            key = tuple(sorted(face))
            if key in seen:
                continue
            seen.add(key)
            result.append(face)
        return result

    @staticmethod
    def _point_on_segment_2d(point: Point2, a: Point2, b: Point2) -> bool:
        cross = abs(ChenHanExactStrategy._orient_2d(a, b, point))
        if cross > 1e-10:
            return False
        dot = (point[0] - a[0]) * (point[0] - b[0]) + (point[1] - a[1]) * (point[1] - b[1])
        return dot <= 1e-10

    @staticmethod
    def _point_in_triangle_2d(
        point: Point2,
        tri: Face,
        coords: Dict[int, Point2],
    ) -> bool:
        a, b, c = (coords[idx] for idx in tri)
        return (
            ChenHanExactStrategy._orient_2d(a, b, point) >= -1e-10
            and ChenHanExactStrategy._orient_2d(b, c, point) >= -1e-10
            and ChenHanExactStrategy._orient_2d(c, a, point) >= -1e-10
        )

    @staticmethod
    def _face_area(vertices: Sequence[Vector3], face: Face) -> float:
        a, b, c = face
        return 0.5 * (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]).length()

    @staticmethod
    def _orient_outward(vertices: Sequence[Vector3], face: Face, center: Vector3) -> Face:
        a, b, c = face
        normal = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a])
        face_center = (vertices[a] + vertices[b] + vertices[c]) / 3.0
        if (center - face_center).dot(normal) > 0.0:
            return (a, c, b)
        return face

    def _chen_han_shortest_path(
        self,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
        start_id: int,
        end_id: int,
    ) -> Optional[_TargetCandidate]:
        edge_to_faces = self._edge_to_faces(faces)
        start_faces = [i for i, face in enumerate(faces) if start_id in face]
        end_faces = {i for i, face in enumerate(faces) if end_id in face}
        if not start_faces or not end_faces:
            return None

        self._next_window_id = 1
        queue: Deque[_Window] = deque()
        active: Dict[Tuple[int, Edge], List[_Window]] = {}
        best: Optional[_TargetCandidate] = None

        for face_id in start_faces:
            face_coords = self._initial_face_coords(vertices, faces[face_id])
            source = face_coords[start_id]
            seed = self._new_window(
                face_id=face_id,
                edge=None,
                left=0.0,
                right=1.0,
                pseudo_source=source,
                source_distance=0.0,
                face_coords=face_coords,
                sequence=[face_id],
                predecessor=None,
            )
            queue.append(seed)

        processed = 0
        while queue and processed < self.max_windows:
            window = queue.popleft()
            processed += 1

            if window.face_id in end_faces:
                reached = self._try_reach_target(window, vertices, faces, end_id)
                if reached is not None and (
                    best is None or reached.distance < best.distance - 1e-8
                ):
                    best = reached

            for child in self._propagate_window(window, vertices, faces, edge_to_faces):
                if best is not None:
                    optimistic = self._lower_bound_to_target(child, vertices, faces, end_id)
                    if optimistic >= best.distance - 1e-8:
                        continue
                key = (child.face_id, child.edge)
                if child.edge is None:
                    continue
                kept = active.setdefault(key, [])
                if self._is_dominated(child, kept):
                    continue
                kept[:] = [old for old in kept if not self._dominates(child, old)]
                kept.append(child)
                queue.append(child)

        return best

    def _new_window(
        self,
        *,
        face_id: int,
        edge: Optional[Edge],
        left: float,
        right: float,
        pseudo_source: Point2,
        source_distance: float,
        face_coords: Dict[int, Point2],
        sequence: List[int],
        predecessor: Optional[_Window],
    ) -> _Window:
        window = _Window(
            face_id=face_id,
            edge=edge,
            left=max(0.0, min(left, right)),
            right=min(1.0, max(left, right)),
            pseudo_source=pseudo_source,
            source_distance=source_distance,
            face_coords=face_coords,
            sequence=sequence,
            predecessor=predecessor,
            id=self._next_window_id,
        )
        self._next_window_id += 1
        return window

    @staticmethod
    def _edge_to_faces(faces: Sequence[Face]) -> Dict[Edge, List[int]]:
        edge_to_faces: Dict[Edge, List[int]] = {}
        for face_id, face in enumerate(faces):
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_to_faces.setdefault(tuple(sorted(edge)), []).append(face_id)
        return edge_to_faces

    def _propagate_window(
        self,
        window: _Window,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
        edge_to_faces: Dict[Edge, List[int]],
    ) -> List[_Window]:
        face = faces[window.face_id]
        outgoing_edges = [tuple(sorted(edge)) for edge in self._face_edges(face)]
        if window.edge is not None:
            outgoing_edges = [edge for edge in outgoing_edges if edge != window.edge]

        children: list[_Window] = []
        for out_edge in outgoing_edges:
            out_interval = self._project_window_to_edge(window, out_edge)
            if out_interval is None:
                continue

            neighbor = self._neighbor_across_edge(edge_to_faces, window.face_id, out_edge)
            if neighbor is None or neighbor in window.sequence:
                continue

            neighbor_coords = self._unfold_neighbor_face(
                vertices,
                faces,
                window.face_id,
                neighbor,
                out_edge,
                window.face_coords,
            )
            if neighbor_coords is None:
                continue

            child = self._new_window(
                face_id=neighbor,
                edge=out_edge,
                left=out_interval[0],
                right=out_interval[1],
                pseudo_source=window.pseudo_source,
                source_distance=window.source_distance,
                face_coords=neighbor_coords,
                sequence=[*window.sequence, neighbor],
                predecessor=window,
            )
            children.append(child)
        return children

    @staticmethod
    def _face_edges(face: Face) -> Tuple[Edge, Edge, Edge]:
        return ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))

    @staticmethod
    def _neighbor_across_edge(
        edge_to_faces: Dict[Edge, List[int]],
        face_id: int,
        edge: Edge,
    ) -> Optional[int]:
        incident = edge_to_faces.get(edge, [])
        if len(incident) != 2:
            return None
        return incident[1] if incident[0] == face_id else incident[0]

    def _project_window_to_edge(self, window: _Window, out_edge: Edge) -> Optional[Tuple[float, float]]:
        if window.edge is None:
            return (0.0, 1.0)

        a = window.face_coords[out_edge[0]]
        b = window.face_coords[out_edge[1]]
        samples = [0.0, 1.0]
        entry_a = self._edge_point_2d(window, window.left)
        entry_b = self._edge_point_2d(window, window.right)

        for boundary in (entry_a, entry_b):
            hit = self._line_line_intersection(window.pseudo_source, boundary, a, b)
            if hit is not None:
                _ray_t, edge_t = hit
                if -1e-8 <= edge_t <= 1.0 + 1e-8:
                    samples.append(max(0.0, min(1.0, edge_t)))

        samples = sorted(set(round(t, 12) for t in samples if -1e-8 <= t <= 1.0 + 1e-8))
        intervals: list[tuple[float, float]] = []
        for lo, hi in zip(samples, samples[1:]):
            mid = (lo + hi) * 0.5
            if self._edge_point_sees_window(window, out_edge, mid):
                intervals.append((lo, hi))

        for t in samples:
            if self._edge_point_sees_window(window, out_edge, t):
                intervals.append((t, t))

        if not intervals:
            return None

        left = min(lo for lo, _hi in intervals)
        right = max(hi for _lo, hi in intervals)
        if right - left < 1e-10:
            return None
        return (left, right)

    def _edge_point_sees_window(self, window: _Window, edge: Edge, edge_t: float) -> bool:
        point = self._point_on_edge_2d(window.face_coords, edge, edge_t)
        hit = self._segment_intersection_2d(
            window.pseudo_source,
            point,
            window.face_coords[window.edge[0]],
            window.face_coords[window.edge[1]],
        )
        if hit is None:
            return False
        line_t, entry_t = hit
        return (
            -1e-8 <= line_t <= 1.0 + 1e-8
            and window.left - 1e-8 <= entry_t <= window.right + 1e-8
        )

    def _try_reach_target(
        self,
        window: _Window,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
        end_id: int,
    ) -> Optional[_TargetCandidate]:
        end_2d = window.face_coords.get(end_id)
        if end_2d is None:
            return None
        if window.edge is not None:
            hit = self._segment_intersection_2d(
                window.pseudo_source,
                end_2d,
                window.face_coords[window.edge[0]],
                window.face_coords[window.edge[1]],
            )
            if hit is None:
                return None
            line_t, entry_t = hit
            if not (
                -1e-8 <= line_t <= 1.0 + 1e-8
                and window.left - 1e-8 <= entry_t <= window.right + 1e-8
            ):
                return None

        distance = window.source_distance + self._distance_2d(window.pseudo_source, end_2d)
        return _TargetCandidate(window=window, distance=distance)

    def _lower_bound_to_target(
        self,
        window: _Window,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
        end_id: int,
    ) -> float:
        if end_id in window.face_coords:
            return window.source_distance + self._distance_2d(
                window.pseudo_source,
                window.face_coords[end_id],
            )
        return window.source_distance

    def _is_dominated(self, candidate: _Window, existing: Sequence[_Window]) -> bool:
        return any(self._dominates(old, candidate) for old in existing)

    def _dominates(self, a: _Window, b: _Window) -> bool:
        if a.edge != b.edge or a.face_id != b.face_id:
            return False
        if a.left > b.left + 1e-8 or a.right < b.right - 1e-8:
            return False
        checks = (b.left, (b.left + b.right) * 0.5, b.right)
        return all(self._window_distance_at(a, t) <= self._window_distance_at(b, t) + 1e-8 for t in checks)

    def _window_distance_at(self, window: _Window, edge_t: float) -> float:
        assert window.edge is not None
        point = self._point_on_edge_2d(window.face_coords, window.edge, edge_t)
        return window.source_distance + self._distance_2d(window.pseudo_source, point)

    def _recover_path(
        self,
        window: _Window,
        end_id: int,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
    ) -> List[Vector3]:
        end = vertices[end_id]
        points = self._recover_to_point(window, end, vertices, faces)
        if not points[-1].approx_equal(end, eps=1e-7):
            points.append(end)
        return self._dedupe_points(points)

    def _recover_to_point(
        self,
        window: _Window,
        target: Vector3,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
    ) -> List[Vector3]:
        if window.edge is None:
            start_vertex = self._source_vertex_from_seed(window, vertices, faces)
            return [start_vertex, target]

        target_2d = self._point_to_face_2d(vertices, faces[window.face_id], window.face_coords, target)
        hit = self._segment_intersection_2d(
            window.pseudo_source,
            target_2d,
            window.face_coords[window.edge[0]],
            window.face_coords[window.edge[1]],
        )
        if hit is None:
            edge_t = (window.left + window.right) * 0.5
        else:
            _line_t, edge_t = hit
            edge_t = max(window.left, min(window.right, edge_t))

        crossing = self._point_on_edge_3d(vertices, window.edge, edge_t)
        if window.predecessor is None:
            return [crossing, target]
        points = self._recover_to_point(window.predecessor, crossing, vertices, faces)
        points.append(target)
        return points

    @staticmethod
    def _source_vertex_from_seed(
        window: _Window,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
    ) -> Vector3:
        for idx in faces[window.face_id]:
            if window.face_coords[idx] == window.pseudo_source:
                return vertices[idx]
        return vertices[faces[window.face_id][0]]

    def _unfold_neighbor_face(
        self,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
        current_id: int,
        neighbor_id: int,
        shared_edge: Edge,
        current_coords: Dict[int, Point2],
    ) -> Optional[Dict[int, Point2]]:
        current_face = faces[current_id]
        neighbor_face = faces[neighbor_id]
        prev_vertex = next(idx for idx in current_face if idx not in shared_edge)
        new_vertex = next(idx for idx in neighbor_face if idx not in shared_edge)

        a2 = current_coords[shared_edge[0]]
        b2 = current_coords[shared_edge[1]]
        prev2 = current_coords[prev_vertex]
        da = vertices[new_vertex].distance_to(vertices[shared_edge[0]])
        db = vertices[new_vertex].distance_to(vertices[shared_edge[1]])
        candidates = self._circle_intersections(a2, da, b2, db)
        if not candidates:
            return None

        side_prev = self._orient_2d(a2, b2, prev2)
        chosen = candidates[0]
        for candidate in candidates:
            side_candidate = self._orient_2d(a2, b2, candidate)
            if side_prev * side_candidate < 0.0:
                chosen = candidate
                break
        return {shared_edge[0]: a2, shared_edge[1]: b2, new_vertex: chosen}

    @staticmethod
    def _initial_face_coords(vertices: Sequence[Vector3], face: Face) -> Dict[int, Point2]:
        a, b, c = face
        ab = vertices[a].distance_to(vertices[b])
        ac = vertices[a].distance_to(vertices[c])
        bc = vertices[b].distance_to(vertices[c])
        x = (ac * ac + ab * ab - bc * bc) / (2.0 * ab)
        y = max(0.0, ac * ac - x * x)
        return {a: (0.0, 0.0), b: (ab, 0.0), c: (x, math.sqrt(y))}

    def _point_to_face_2d(
        self,
        vertices: Sequence[Vector3],
        face: Face,
        coords: Dict[int, Point2],
        point: Vector3,
    ) -> Point2:
        a, b, c = face
        u, v, w = barycentric_coordinates(point, vertices[a], vertices[b], vertices[c])
        a2, b2, c2 = coords[a], coords[b], coords[c]
        return (
            u * a2[0] + v * b2[0] + w * c2[0],
            u * a2[1] + v * b2[1] + w * c2[1],
        )

    def _edge_point_2d(self, window: _Window, edge_t: float) -> Point2:
        assert window.edge is not None
        return self._point_on_edge_2d(window.face_coords, window.edge, edge_t)

    @staticmethod
    def _point_on_edge_2d(coords: Dict[int, Point2], edge: Edge, t: float) -> Point2:
        a = coords[edge[0]]
        b = coords[edge[1]]
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    @staticmethod
    def _point_on_edge_3d(vertices: Sequence[Vector3], edge: Edge, t: float) -> Vector3:
        return vertices[edge[0]].lerp(vertices[edge[1]], t)

    @staticmethod
    def _circle_intersections(a: Point2, ra: float, b: Point2, rb: float) -> List[Point2]:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        d = math.hypot(dx, dy)
        if d < EPSILON:
            return []
        x = (ra * ra - rb * rb + d * d) / (2.0 * d)
        h_sq = max(0.0, ra * ra - x * x)
        h = math.sqrt(h_sq)
        ux = dx / d
        uy = dy / d
        base = (a[0] + ux * x, a[1] + uy * x)
        perp = (-uy * h, ux * h)
        return [(base[0] + perp[0], base[1] + perp[1]), (base[0] - perp[0], base[1] - perp[1])]

    @staticmethod
    def _line_line_intersection(
        p: Point2,
        q: Point2,
        a: Point2,
        b: Point2,
    ) -> Optional[Tuple[float, float]]:
        rx = q[0] - p[0]
        ry = q[1] - p[1]
        sx = b[0] - a[0]
        sy = b[1] - a[1]
        denom = rx * sy - ry * sx
        if abs(denom) < 1e-10:
            return None
        ax = a[0] - p[0]
        ay = a[1] - p[1]
        return ((ax * sy - ay * sx) / denom, (ax * ry - ay * rx) / denom)

    def _segment_intersection_2d(
        self,
        p: Point2,
        q: Point2,
        a: Point2,
        b: Point2,
    ) -> Optional[Tuple[float, float]]:
        hit = self._line_line_intersection(p, q, a, b)
        if hit is None:
            return None
        line_t, edge_t = hit
        if -1e-8 <= line_t <= 1.0 + 1e-8 and -1e-8 <= edge_t <= 1.0 + 1e-8:
            return (max(0.0, min(1.0, line_t)), max(0.0, min(1.0, edge_t)))
        return None

    @staticmethod
    def _orient_2d(a: Point2, b: Point2, c: Point2) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def _distance_2d(a: Point2, b: Point2) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _path_length(points: Sequence[Vector3]) -> float:
        return sum(a.distance_to(b) for a, b in zip(points, points[1:]))

    @staticmethod
    def _dedupe_points(points: Sequence[Vector3]) -> List[Vector3]:
        result: list[Vector3] = []
        for point in points:
            if not result or not point.approx_equal(result[-1], eps=1e-7):
                result.append(point)
        return result
