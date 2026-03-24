"""Interactive point placement controller.

Manages the workflow of clicking on a mesh to set a start point, then an
end point, with automatic state transitions.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Optional

from core.vector3 import Vector3
from mesh.mesh import Mesh
from render.camera import Camera
from interaction.mouse_picking import pick_mesh


class PlacementState(Enum):
    """State machine for point placement."""
    PLACE_START = auto()
    PLACE_END = auto()
    DONE = auto()


# Type alias for external notification callbacks.
PointPlacedCallback = Callable[[Vector3, Vector3], None]


class PointPlacer:
    """Handles left-click → mesh intersection → start/end point placement.

    Usage:
        1. Create a ``PointPlacer`` with a mesh, camera, and viewport size getter.
        2. Register ``on_mouse_button`` as the renderer's mouse-button callback.
        3. Optionally set ``on_both_placed`` to be notified when both points are set.

    Attributes:
        start_point: The placed start point (or *None*).
        end_point:   The placed end point (or *None*).
        state:       Current placement state.
    """

    def __init__(
        self,
        mesh: Mesh,
        camera: Camera,
        get_viewport_size: Callable[[], tuple[int, int]],
        get_cursor_pos: Callable[[], tuple[float, float]],
    ) -> None:
        self.mesh = mesh
        self.camera = camera
        self._get_viewport_size = get_viewport_size
        self._get_cursor_pos = get_cursor_pos

        self.start_point: Optional[Vector3] = None
        self.end_point: Optional[Vector3] = None
        self.state: PlacementState = PlacementState.PLACE_START

        # External callback fired when both points have been placed.
        self.on_both_placed: Optional[PointPlacedCallback] = None

    def reset(self) -> None:
        """Clear both points and restart placement."""
        self.start_point = None
        self.end_point = None
        self.state = PlacementState.PLACE_START

    def on_click(self, button: int, action: int, mods: int) -> None:
        """Handle a mouse-button event.

        Call this from the renderer's mouse-button callback.  Only reacts
        to left-button press events (``button == 0``, ``action == 1``).

        Args:
            button: GLFW button code (0 = left).
            action: GLFW action (1 = press).
            mods:   Modifier key bitmask (unused).
        """
        GLFW_MOUSE_BUTTON_LEFT = 0
        GLFW_PRESS = 1

        if button != GLFW_MOUSE_BUTTON_LEFT or action != GLFW_PRESS:
            return

        if self.state == PlacementState.DONE:
            return

        sx, sy = self._get_cursor_pos()
        vw, vh = self._get_viewport_size()

        result = pick_mesh(sx, sy, vw, vh, self.camera, self.mesh)
        if result is not None:
            hit_point, _tri_idx = result
        else:
            # Place point in free space: intersect ray with a plane
            # through the mesh centre, perpendicular to the camera view.
            ray = self.camera.screen_to_ray(sx, sy, vw, vh)
            plane_point = self.mesh.center()
            plane_normal = self.camera.forward
            denom = plane_normal.dot(ray.direction)
            if abs(denom) < 1e-9:
                return  # ray parallel to plane
            t = (plane_point - ray.origin).dot(plane_normal) / denom
            if t < 0:
                return  # behind camera
            hit_point = ray.point_at(t)

        if self.state == PlacementState.PLACE_START:
            self.start_point = hit_point
            self.state = PlacementState.PLACE_END
        elif self.state == PlacementState.PLACE_END:
            self.end_point = hit_point
            self.state = PlacementState.DONE
            if self.on_both_placed is not None and self.start_point is not None:
                self.on_both_placed(self.start_point, self.end_point)
