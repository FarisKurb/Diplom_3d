"""Renders the shortest-path line strip, start/end markers, and optional
debug overlays (raw un-smoothed path, graph sample nodes).

Uses the OpenGL fixed-function pipeline (immediate mode).
"""

from __future__ import annotations

import math
from typing import List

from OpenGL.GL import (  # type: ignore
    glBegin,
    glEnd,
    glVertex3f,
    glColor3f,
    glLineWidth,
    glPointSize,
    glDisable,
    glEnable,
    GL_LINE_STRIP,
    GL_POINTS,
    GL_LIGHTING,
    GL_TRIANGLE_FAN,
)

from core.vector3 import Vector3
from config import (
    PATH_COLOR, POINT_COLOR, POINT_RADIUS,
    START_POINT_COLOR, END_POINT_COLOR,
    RAW_PATH_COLOR, RAW_PATH_WIDTH,
    SAMPLE_NODE_COLOR, SAMPLE_NODE_SIZE,
)


def draw_sphere_marker(center: Vector3, radius: float, color: tuple[float, float, float],
                       slices: int = 12, stacks: int = 8) -> None:
    """Draw a small solid sphere at *center* using triangle fans.

    This is a lightweight replacement for ``glutSolidSphere`` that avoids
    a GLUT dependency.
    """
    glDisable(GL_LIGHTING)
    glColor3f(*color)

    for i in range(stacks):
        lat0 = math.pi * (-0.5 + i / stacks)
        lat1 = math.pi * (-0.5 + (i + 1) / stacks)
        z0 = math.sin(lat0) * radius
        z1 = math.sin(lat1) * radius
        r0 = math.cos(lat0) * radius
        r1 = math.cos(lat1) * radius

        glBegin(GL_TRIANGLE_FAN)
        for j in range(slices + 1):
            lng = 2.0 * math.pi * j / slices
            x_cos = math.cos(lng)
            y_sin = math.sin(lng)
            glVertex3f(center.x + x_cos * r1, center.y + y_sin * r1, center.z + z1)
            glVertex3f(center.x + x_cos * r0, center.y + y_sin * r0, center.z + z0)
        glEnd()


class PathRenderer:
    """Draws a 3-D polyline path and optional start/end point markers.

    Supports per-algorithm path colours, distinct start/end marker colours,
    and a debug overlay that shows graph sample nodes and the raw
    (un-smoothed) path.

    Attributes:
        path:            Ordered list of 3-D points forming the path.
        path_color:      RGB tuple for the path line strip.
        start_color:     RGB colour for the start-point marker.
        end_color:       RGB colour for the end-point marker.
        point_radius:    Radius of the marker spheres.
        line_width:      Width in pixels for the path line.
        start_point:     Optional start point to draw as a marker.
        end_point:       Optional end point to draw as a marker.
        raw_path:        Un-smoothed path (drawn as thin overlay in debug mode).
        sample_nodes:    Graph sample positions (drawn as dots in debug mode).
        debug:           Whether the debug overlay is visible.
    """

    def __init__(
        self,
        path: List[Vector3] | None = None,
        path_color: tuple[float, float, float] = PATH_COLOR,
        point_color: tuple[float, float, float] = POINT_COLOR,
        point_radius: float = POINT_RADIUS,
        line_width: float = 3.0,
    ) -> None:
        self.path: List[Vector3] = path if path is not None else []
        self.path_color = path_color
        self.start_color: tuple[float, float, float] = START_POINT_COLOR
        self.end_color: tuple[float, float, float] = END_POINT_COLOR
        self.point_radius = point_radius
        self.line_width = line_width
        self.start_point: Vector3 | None = None
        self.end_point: Vector3 | None = None

        # Debug overlay data.
        self.raw_path: List[Vector3] = []
        self.sample_nodes: List[Vector3] = []
        self.debug: bool = False

    def draw(self) -> None:
        """Render the path line strip, point markers, and debug overlay."""
        self._draw_markers()
        if self.debug:
            self._draw_sample_nodes()
            self._draw_raw_path()
        self._draw_path()

    def set_path(self, points: List[Vector3]) -> None:
        """Replace the current path with a new list of points."""
        self.path = list(points)

    def clear(self) -> None:
        """Remove the path, both markers, and debug data."""
        self.path.clear()
        self.raw_path.clear()
        self.sample_nodes.clear()
        self.start_point = None
        self.end_point = None

    # ── private helpers ─────────────────────────────────────

    def _draw_path(self) -> None:
        """Draw the path as a GL_LINE_STRIP."""
        if len(self.path) < 2:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(self.line_width)
        r, g, b = self.path_color
        glColor3f(r, g, b)

        glBegin(GL_LINE_STRIP)
        for p in self.path:
            glVertex3f(p.x, p.y, p.z)
        glEnd()

    def _draw_markers(self) -> None:
        """Draw small spheres at start and end points with distinct colours."""
        if self.start_point is not None:
            draw_sphere_marker(self.start_point, self.point_radius, self.start_color)
        if self.end_point is not None:
            draw_sphere_marker(self.end_point, self.point_radius, self.end_color)

    def _draw_raw_path(self) -> None:
        """Draw the un-smoothed path as a thin line (debug overlay)."""
        if len(self.raw_path) < 2:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(RAW_PATH_WIDTH)
        glColor3f(*RAW_PATH_COLOR)

        glBegin(GL_LINE_STRIP)
        for p in self.raw_path:
            glVertex3f(p.x, p.y, p.z)
        glEnd()

    def _draw_sample_nodes(self) -> None:
        """Draw graph sample nodes as small dots (debug overlay)."""
        if not self.sample_nodes:
            return

        glDisable(GL_LIGHTING)
        glPointSize(SAMPLE_NODE_SIZE)
        glColor3f(*SAMPLE_NODE_COLOR)

        glBegin(GL_POINTS)
        for p in self.sample_nodes:
            glVertex3f(p.x, p.y, p.z)
        glEnd()
