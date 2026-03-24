"""Heads-up display — draws 2-D text overlays on the OpenGL viewport.

Uses the OpenGL fixed-function pipeline with an orthographic projection
to render bitmap-font text (via GLUT) at arbitrary screen positions.
"""

from __future__ import annotations

from typing import List, Tuple

from OpenGL.GL import (  # type: ignore
    glColor3f,
    glDisable,
    glEnable,
    glMatrixMode,
    glLoadIdentity,
    glPushMatrix,
    glPopMatrix,
    glRasterPos2f,
    glWindowPos2f,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_PROJECTION,
    GL_MODELVIEW,
)
from OpenGL.GLU import gluOrtho2D  # type: ignore
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12, GLUT_BITMAP_HELVETICA_18  # type: ignore

from config import HUD_COLOR, HUD_TITLE_COLOR


# One-time GLUT init flag.
_glut_initialised: bool = False


def _ensure_glut() -> None:
    """Initialise GLUT for bitmap fonts (once)."""
    global _glut_initialised
    if not _glut_initialised:
        glutInit()
        _glut_initialised = True


def draw_text(
    x: float,
    y: float,
    text: str,
    color: Tuple[float, float, float] = HUD_COLOR,
    *,
    font: int = GLUT_BITMAP_HELVETICA_12,
) -> None:
    """Render a string of text at window position (*x*, *y*).

    (*x*, *y*) are in pixels from the bottom-left corner of the window.

    Args:
        x:     Horizontal pixel position.
        y:     Vertical pixel position.
        text:  The string to render.
        color: RGB colour tuple.
        font:  GLUT bitmap font constant.
    """
    _ensure_glut()
    glColor3f(*color)
    glWindowPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(font, ord(ch))


class HudRenderer:
    """Draws on-screen information text (placement state, path stats, help).

    The HUD renders in a 2-D orthographic overlay on top of the 3-D scene.
    Call :meth:`draw` at the end of the render callback, after all 3-D
    drawing is complete.

    Attributes:
        lines:       List of (text, color) tuples to render in the top-left.
        show_help:   Whether to show the controls help block.
        viewport_size_fn: Callable returning ``(width, height)`` in pixels.
    """

    def __init__(
        self,
        viewport_size_fn=None,
    ) -> None:
        self.lines: List[Tuple[str, Tuple[float, float, float]]] = []
        self.show_help: bool = True
        self._viewport_size_fn = viewport_size_fn

    def set_viewport_size_fn(self, fn) -> None:
        self._viewport_size_fn = fn

    def set_lines(self, lines: List[Tuple[str, Tuple[float, float, float]]]) -> None:
        """Replace the status lines to display."""
        self.lines = list(lines)

    def draw(self) -> None:
        """Render the HUD overlay."""
        if self._viewport_size_fn is None:
            return

        w, h = self._viewport_size_fn()
        if w == 0 or h == 0:
            return

        _ensure_glut()

        # Switch to 2-D orthographic projection.
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, w, 0, h)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # ── Status lines (top-left) ────────────────────────
        y_pos = h - 24
        for text, color in self.lines:
            draw_text(12, y_pos, text, color)
            y_pos -= 18

        # ── Help block (bottom-left) ───────────────────────
        if self.show_help:
            help_lines = [
                "Left-click: place points",
                "Left-drag: orbit | Mid/Right-drag: pan",
                "Scroll: zoom | R: reset | S: smooth",
                "A: cycle algorithm | M: cycle mesh",
                "D: debug overlay | H: help | ESC/Q: quit",
            ]
            y_pos = 12 + 18 * (len(help_lines) - 1)
            for line in help_lines:
                draw_text(12, y_pos, line, HUD_COLOR)
                y_pos -= 18

        # Restore 3-D projection.
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
