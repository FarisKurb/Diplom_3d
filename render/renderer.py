"""OpenGL window manager built on GLFW.

Creates a window, sets up the OpenGL context, and drives the main
render loop.  Rendering and input callbacks are plugged in externally
so that this module only handles windowing concerns.
"""

from __future__ import annotations

import ctypes
from typing import Callable, Optional

import glfw  # type: ignore
from OpenGL.GL import (  # type: ignore
    glClear,
    glClearColor,
    glEnable,
    glViewport,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    glBlendFunc,
    glMatrixMode,
    glLoadIdentity,
    glLoadMatrixf,
    GL_PROJECTION,
    GL_MODELVIEW,
)
from OpenGL.GLU import gluPerspective  # type: ignore

from config import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE
from render.camera import Camera


# Callback type aliases.
RenderCallback = Callable[[], None]
MouseButtonCallback = Callable[[int, int, int, int], None]  # window, button, action, mods — we pass ints
ScrollCallback = Callable[[float], None]
ResizeCallback = Callable[[int, int], None]
KeyCallback = Callable[[int, int, int, int], None]


class Renderer:
    """Manages a GLFW window, OpenGL context, and the main render loop.

    Attributes:
        camera: The :class:`Camera` instance used for the scene.
    """

    def __init__(
        self,
        width: int = WINDOW_WIDTH,
        height: int = WINDOW_HEIGHT,
        title: str = WINDOW_TITLE,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.camera = Camera()
        self._window: Optional[object] = None

        # External callbacks.
        self._render_fn: Optional[RenderCallback] = None
        self._mouse_button_fn: Optional[MouseButtonCallback] = None
        self._key_fn: Optional[KeyCallback] = None

        # Mouse state for orbit / pan.
        self._last_mx: float = 0.0
        self._last_my: float = 0.0
        self._left_pressed: bool = False
        self._middle_pressed: bool = False
        self._right_pressed: bool = False

        # Click vs drag detection.
        self._press_mx: float = 0.0
        self._press_my: float = 0.0
        self._left_dragged: bool = False
        self._CLICK_THRESHOLD: float = 5.0  # pixels

    # ── public API ──────────────────────────────────────────

    def set_render_callback(self, fn: RenderCallback) -> None:
        self._render_fn = fn

    def set_mouse_button_callback(self, fn: MouseButtonCallback) -> None:
        self._mouse_button_fn = fn

    def set_key_callback(self, fn: KeyCallback) -> None:
        self._key_fn = fn

    def init(self) -> None:
        """Initialise GLFW, create window, and configure OpenGL state."""
        if not glfw.init():
            raise RuntimeError("Failed to initialise GLFW")

        glfw.window_hint(glfw.SAMPLES, 4)
        self._window = glfw.create_window(
            self.width, self.height, self.title, None, None
        )
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)

        # Register GLFW callbacks.
        glfw.set_framebuffer_size_callback(self._window, self._on_resize)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_scroll_callback(self._window, self._on_scroll)
        glfw.set_key_callback(self._window, self._on_key)

        # OpenGL defaults.
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.15, 0.15, 0.18, 1.0)

        self._apply_projection()

    def run(self) -> None:
        """Enter the main render loop.  Blocks until the window is closed."""
        assert self._window is not None, "Call init() before run()"
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self._apply_view()

            if self._render_fn is not None:
                self._render_fn()

            glfw.swap_buffers(self._window)

        glfw.terminate()

    def get_window(self) -> object:
        """Return the underlying GLFW window handle."""
        assert self._window is not None
        return self._window

    def get_cursor_pos(self) -> tuple[float, float]:
        """Return current cursor position in screen pixels."""
        assert self._window is not None
        return glfw.get_cursor_pos(self._window)

    def get_framebuffer_size(self) -> tuple[int, int]:
        assert self._window is not None
        return glfw.get_framebuffer_size(self._window)

    # ── projection / view helpers ───────────────────────────

    def _apply_projection(self) -> None:
        w, h = self.width, self.height
        if self._window is not None:
            w, h = glfw.get_framebuffer_size(self._window)
        if h == 0:
            h = 1
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        fov, aspect, near, far = self.camera.get_projection_params(w / h)
        gluPerspective(fov, aspect, near, far)
        glMatrixMode(GL_MODELVIEW)

    def _apply_view(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        mat = self.camera.get_view_matrix()
        glLoadMatrixf((ctypes.c_float * 16)(*mat))

    # ── GLFW callbacks ──────────────────────────────────────

    def _on_resize(self, _win: object, width: int, height: int) -> None:
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        self._apply_projection()

    def _on_cursor(self, _win: object, xpos: float, ypos: float) -> None:
        dx = xpos - self._last_mx
        dy = ypos - self._last_my
        self._last_mx = xpos
        self._last_my = ypos

        if self._left_pressed:
            # Check if we've moved enough to count as a drag.
            total_dx = xpos - self._press_mx
            total_dy = ypos - self._press_my
            if (total_dx * total_dx + total_dy * total_dy) >= self._CLICK_THRESHOLD ** 2:
                self._left_dragged = True
            if self._left_dragged:
                self.camera.orbit(dx, -dy)
        if self._middle_pressed or self._right_pressed:
            self.camera.pan(dx, dy)

    def _on_mouse_button(
        self, _win: object, button: int, action: int, mods: int
    ) -> None:
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._left_pressed = True
                self._left_dragged = False
                if self._window is not None:
                    self._press_mx, self._press_my = glfw.get_cursor_pos(self._window)
            else:
                self._left_pressed = False
                # Forward click (not drag) to external callback.
                if not self._left_dragged and self._mouse_button_fn is not None:
                    self._mouse_button_fn(button, glfw.PRESS, mods, 0)
                return
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._middle_pressed = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_pressed = action == glfw.PRESS

        # Update last position to avoid jump on first drag.
        if action == glfw.PRESS and self._window is not None:
            self._last_mx, self._last_my = glfw.get_cursor_pos(self._window)

    def _on_scroll(self, _win: object, _xoffset: float, yoffset: float) -> None:
        self.camera.zoom(yoffset)

    def _on_key(
        self, _win: object, key: int, scancode: int, action: int, mods: int
    ) -> None:
        if self._key_fn is not None:
            self._key_fn(key, scancode, action, mods)
