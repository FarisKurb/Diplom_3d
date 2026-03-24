"""Renders a :class:`Mesh` using OpenGL fixed-function pipeline.

Supports solid face rendering with per-face normals (flat shading) and
an optional wireframe overlay.
"""

from __future__ import annotations

from OpenGL.GL import (  # type: ignore
    glBegin,
    glEnd,
    glVertex3f,
    glNormal3f,
    glColor3f,
    glColor4f,
    glEnable,
    glDisable,
    glPolygonMode,
    glLineWidth,
    glPolygonOffset,
    glEnable as _glEnable,
    GL_TRIANGLES,
    GL_LINES,
    GL_FRONT_AND_BACK,
    GL_FILL,
    GL_LINE,
    GL_LIGHTING,
    GL_LIGHT0,
    GL_COLOR_MATERIAL,
    GL_POLYGON_OFFSET_FILL,
)

from mesh.mesh import Mesh
from config import MESH_COLOR, WIREFRAME_COLOR


class MeshRenderer:
    """Draws a :class:`Mesh` as solid faces with an optional wireframe overlay.

    Attributes:
        mesh:            The mesh to draw (can be replaced at runtime).
        face_color:      RGB tuple for mesh faces.
        wireframe_color: RGB tuple for wireframe lines.
        show_faces:      Whether to draw solid faces.
        show_wireframe:  Whether to draw wireframe overlay.
    """

    def __init__(
        self,
        mesh: Mesh | None = None,
        face_color: tuple[float, float, float] = MESH_COLOR,
        wireframe_color: tuple[float, float, float] = WIREFRAME_COLOR,
        show_faces: bool = True,
        show_wireframe: bool = True,
    ) -> None:
        self.mesh = mesh
        self.face_color = face_color
        self.wireframe_color = wireframe_color
        self.show_faces = show_faces
        self.show_wireframe = show_wireframe

    def draw(self) -> None:
        """Render the mesh.  Does nothing if *mesh* is ``None``."""
        if self.mesh is None:
            return

        if self.show_faces:
            self._draw_faces()

        if self.show_wireframe:
            self._draw_wireframe()

    # ── private helpers ─────────────────────────────────────

    def _draw_faces(self) -> None:
        """Draw filled triangles with flat shading."""
        assert self.mesh is not None
        # Use polygon offset so the wireframe renders on top.
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        r, g, b = self.face_color
        glColor3f(r, g, b)

        glBegin(GL_TRIANGLES)
        for tri in self.mesh.triangles:
            try:
                n = tri.normal()
            except ValueError:
                continue  # skip degenerate triangles
            glNormal3f(n.x, n.y, n.z)
            glVertex3f(tri.v0.x, tri.v0.y, tri.v0.z)
            glVertex3f(tri.v1.x, tri.v1.y, tri.v1.z)
            glVertex3f(tri.v2.x, tri.v2.y, tri.v2.z)
        glEnd()

        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_POLYGON_OFFSET_FILL)

    def _draw_wireframe(self) -> None:
        """Draw triangle edges as lines."""
        assert self.mesh is not None
        glDisable(GL_LIGHTING)
        glLineWidth(1.2)

        r, g, b = self.wireframe_color
        glColor3f(r, g, b)

        glBegin(GL_LINES)
        for tri in self.mesh.triangles:
            for a, b_v in [
                (tri.v0, tri.v1),
                (tri.v1, tri.v2),
                (tri.v2, tri.v0),
            ]:
                glVertex3f(a.x, a.y, a.z)
                glVertex3f(b_v.x, b_v.y, b_v.z)
        glEnd()
