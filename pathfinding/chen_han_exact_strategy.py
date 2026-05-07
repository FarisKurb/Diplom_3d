"""Exact Chen-Han style shortest path on one convex polyhedral obstacle.

For one convex obstacle the free-space shortest path is either the direct
segment, or a geodesic on the convex hull of ``obstacle vertices + start +
end``.  This strategy builds that hull and searches unfolded face strips:
each candidate strip is flattened across its shared edges, the straight line
in the unfolding is tested against the propagated edge intervals, and the
shortest valid unfolding is returned.

The implementation deliberately does not build a visibility graph and does
not restrict the path to mesh vertices or edges; waypoints are crossing points
on hull edges between unfolded faces, while the path pieces inside faces are
straight geodesic segments.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.math_utils import EPSILON
from core.vector3 import Vector3
from geometry.intersection import segment_intersects_mesh
from mesh.mesh import Mesh
from pathfinding.strategy import PathResult, PathfindingStrategy


Point2 = Tuple[float, float]
Face = Tuple[int, int, int]
Edge = Tuple[int, int]


@dataclass(frozen=True)
class _UnfoldedStrip:
    points: List[Vector3]
    distance: float


class ChenHanExactStrategy(PathfindingStrategy):
    """Exact surface shortest path strategy for one convex obstacle.

    The direct segment is returned immediately when it does not enter the
    obstacle interior.  Otherwise the algorithm builds the convex hull of the
    obstacle vertices plus the query points and propagates Chen-Han style
    windows by unfolding adjacent triangle strips on the hull surface.
    """

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

        best = self._shortest_unfolded_path(hull_vertices, hull_faces, start_id, end_id)
        if best is None:
            return PathResult(found=False, algorithm_name=self.name)

        return PathResult(
            found=True,
            distance=best.distance,
            points=best.points,
            raw_points=list(best.points),
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
            polygon = self._ordered_face_polygon(vertices, sorted(indices), center)
            if len(polygon) < 3:
                continue
            anchor = polygon[0]
            for a, b in zip(polygon[1:-1], polygon[2:]):
                face = (anchor, a, b)
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

    def _ordered_face_polygon(
        self,
        vertices: Sequence[Vector3],
        indices: Sequence[int],
        center: Vector3,
    ) -> List[int]:
        face_center = self._centroid([vertices[i] for i in indices])
        normal = (vertices[indices[1]] - vertices[indices[0]]).cross(
            vertices[indices[2]] - vertices[indices[0]]
        ).normalized()
        if (center - face_center).dot(normal) > 0.0:
            normal = -normal

        axis_u = (vertices[indices[0]] - face_center).normalized()
        axis_v = normal.cross(axis_u).normalized()

        def angle(idx: int) -> float:
            rel = vertices[idx] - face_center
            return math.atan2(rel.dot(axis_v), rel.dot(axis_u))

        ordered = sorted(indices, key=angle)
        if len(ordered) >= 3:
            tri_normal = (vertices[ordered[1]] - vertices[ordered[0]]).cross(
                vertices[ordered[2]] - vertices[ordered[0]]
            )
            if tri_normal.dot(normal) < 0.0:
                ordered.reverse()
        return ordered

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

    def _shortest_unfolded_path(
        self,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
        start_id: int,
        end_id: int,
    ) -> Optional[_UnfoldedStrip]:
        adjacency = self._face_adjacency(faces)
        start_faces = [i for i, face in enumerate(faces) if start_id in face]
        end_faces = {i for i, face in enumerate(faces) if end_id in face}
        if not start_faces or not end_faces:
            return None

        best: Optional[_UnfoldedStrip] = None
        for start_face in start_faces:
            stack: list[tuple[int, list[int], set[int]]] = [(start_face, [start_face], {start_face})]
            while stack:
                current, sequence, visited = stack.pop()
                if current in end_faces:
                    candidate = self._unfold_sequence(vertices, faces, sequence, start_id, end_id)
                    if candidate is not None and (
                        best is None or candidate.distance < best.distance - 1e-8
                    ):
                        best = candidate

                if len(sequence) >= len(faces):
                    continue
                for nxt in adjacency[current]:
                    if nxt not in visited:
                        stack.append((nxt, [*sequence, nxt], visited | {nxt}))
        return best

    @staticmethod
    def _face_adjacency(faces: Sequence[Face]) -> Dict[int, List[int]]:
        edge_to_faces: Dict[Edge, List[int]] = {}
        for face_id, face in enumerate(faces):
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_to_faces.setdefault(tuple(sorted(edge)), []).append(face_id)

        adjacency = {i: [] for i in range(len(faces))}
        for incident in edge_to_faces.values():
            if len(incident) == 2:
                a, b = incident
                adjacency[a].append(b)
                adjacency[b].append(a)
        return adjacency

    def _unfold_sequence(
        self,
        vertices: Sequence[Vector3],
        faces: Sequence[Face],
        sequence: Sequence[int],
        start_id: int,
        end_id: int,
    ) -> Optional[_UnfoldedStrip]:
        first = faces[sequence[0]]
        coords_by_face: list[dict[int, Point2]] = [self._initial_face_coords(vertices, first)]

        for prev_face_id, face_id in zip(sequence, sequence[1:]):
            prev_face = faces[prev_face_id]
            face = faces[face_id]
            shared = [idx for idx in face if idx in prev_face]
            if len(shared) != 2:
                return None
            new_vertex = next(idx for idx in face if idx not in shared)
            prev_vertex = next(idx for idx in prev_face if idx not in shared)
            prev_coords = coords_by_face[-1]
            new_coords = self._place_adjacent_vertex(
                vertices,
                prev_coords,
                shared[0],
                shared[1],
                prev_vertex,
                new_vertex,
            )
            if new_coords is None:
                return None
            coords_by_face.append(new_coords)

        start_2d = coords_by_face[0].get(start_id)
        end_2d = coords_by_face[-1].get(end_id)
        if start_2d is None or end_2d is None:
            return None

        points = [vertices[start_id]]
        last_line_t = 0.0
        for step, (prev_face_id, face_id) in enumerate(zip(sequence, sequence[1:])):
            shared = [idx for idx in faces[face_id] if idx in faces[prev_face_id]]
            a2 = coords_by_face[step][shared[0]]
            b2 = coords_by_face[step][shared[1]]
            hit = self._segment_intersection_2d(start_2d, end_2d, a2, b2)
            if hit is None:
                return None
            line_t, edge_t = hit
            if line_t < last_line_t - 1e-8:
                return None
            last_line_t = line_t
            a3 = vertices[shared[0]]
            b3 = vertices[shared[1]]
            crossing = a3.lerp(b3, edge_t)
            if not crossing.approx_equal(points[-1], eps=1e-7):
                points.append(crossing)

        if not vertices[end_id].approx_equal(points[-1], eps=1e-7):
            points.append(vertices[end_id])
        return _UnfoldedStrip(points=points, distance=self._distance_2d(start_2d, end_2d))

    @staticmethod
    def _initial_face_coords(vertices: Sequence[Vector3], face: Face) -> dict[int, Point2]:
        a, b, c = face
        ab = vertices[a].distance_to(vertices[b])
        ac = vertices[a].distance_to(vertices[c])
        bc = vertices[b].distance_to(vertices[c])
        x = (ac * ac + ab * ab - bc * bc) / (2.0 * ab)
        y = max(0.0, ac * ac - x * x)
        return {a: (0.0, 0.0), b: (ab, 0.0), c: (x, math.sqrt(y))}

    def _place_adjacent_vertex(
        self,
        vertices: Sequence[Vector3],
        prev_coords: dict[int, Point2],
        edge_a: int,
        edge_b: int,
        prev_vertex: int,
        new_vertex: int,
    ) -> Optional[dict[int, Point2]]:
        a2 = prev_coords[edge_a]
        b2 = prev_coords[edge_b]
        prev2 = prev_coords[prev_vertex]
        da = vertices[new_vertex].distance_to(vertices[edge_a])
        db = vertices[new_vertex].distance_to(vertices[edge_b])
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
        return {edge_a: a2, edge_b: b2, new_vertex: chosen}

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
    def _segment_intersection_2d(
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
        line_t = (ax * sy - ay * sx) / denom
        edge_t = (ax * ry - ay * rx) / denom
        if -1e-8 <= line_t <= 1.0 + 1e-8 and -1e-8 <= edge_t <= 1.0 + 1e-8:
            return (max(0.0, min(1.0, line_t)), max(0.0, min(1.0, edge_t)))
        return None

    @staticmethod
    def _orient_2d(a: Point2, b: Point2, c: Point2) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def _distance_2d(a: Point2, b: Point2) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
