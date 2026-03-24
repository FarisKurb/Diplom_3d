"""Orbit camera for the 3-D viewer.

Provides rotation (orbit), zoom, and pan via mouse input.  The camera
looks at a *target* point and orbits around it at a given *distance*.
"""

from __future__ import annotations

import math
from typing import Tuple

from core.vector3 import Vector3
from core.ray import Ray
from config import CAMERA_FOV, CAMERA_NEAR, CAMERA_FAR, CAMERA_DISTANCE


class Camera:
    """Orbit camera defined by spherical angles around a target point.

    Attributes:
        target:   The point the camera looks at.
        distance: Distance from the target.
        yaw:      Horizontal rotation angle in degrees.
        pitch:    Vertical rotation angle in degrees (clamped to ±89°).
        fov:      Vertical field-of-view in degrees.
        near:     Near clipping plane.
        far:      Far clipping plane.
    """

    def __init__(
        self,
        target: Vector3 | None = None,
        distance: float = CAMERA_DISTANCE,
        yaw: float = 45.0,
        pitch: float = 30.0,
        fov: float = CAMERA_FOV,
        near: float = CAMERA_NEAR,
        far: float = CAMERA_FAR,
    ) -> None:
        self.target = target if target is not None else Vector3.zero()
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.fov = fov
        self.near = near
        self.far = far

    # ── derived position ────────────────────────────────────

    @property
    def position(self) -> Vector3:
        """Camera world-space position computed from spherical coordinates."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cos_p = math.cos(pitch_rad)
        x = self.target.x + self.distance * cos_p * math.sin(yaw_rad)
        y = self.target.y + self.distance * math.sin(pitch_rad)
        z = self.target.z + self.distance * cos_p * math.cos(yaw_rad)
        return Vector3(x, y, z)

    @property
    def forward(self) -> Vector3:
        """Unit vector from camera toward the target."""
        return (self.target - self.position).normalized()

    @property
    def right(self) -> Vector3:
        """Unit right vector (perpendicular to forward and world up)."""
        up = Vector3(0, 1, 0)
        return self.forward.cross(up).normalized()

    @property
    def up(self) -> Vector3:
        """Unit up vector for the camera."""
        return self.right.cross(self.forward).normalized()

    # ── interaction ─────────────────────────────────────────

    def orbit(self, dx: float, dy: float, sensitivity: float = 0.3) -> None:
        """Rotate the camera around the target by screen-space deltas.

        Args:
            dx: Horizontal mouse delta (pixels).
            dy: Vertical mouse delta (pixels).
            sensitivity: Degrees per pixel.
        """
        self.yaw += dx * sensitivity
        self.pitch += dy * sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def zoom(self, delta: float, sensitivity: float = 0.1) -> None:
        """Zoom in/out by changing the distance.

        Args:
            delta: Scroll delta (positive = zoom in).
            sensitivity: Distance change per scroll unit.
        """
        self.distance -= delta * sensitivity * self.distance
        self.distance = max(0.1, min(200.0, self.distance))

    def pan(self, dx: float, dy: float, sensitivity: float = 0.005) -> None:
        """Pan the target in the camera's local XY plane.

        Args:
            dx: Horizontal mouse delta.
            dy: Vertical mouse delta.
            sensitivity: World-units per pixel.
        """
        r = self.right
        u = self.up
        offset = r * (-dx * sensitivity * self.distance) + u * (-dy * sensitivity * self.distance)
        self.target = self.target + offset

    # ── matrices (for OpenGL fixed pipeline) ────────────────

    def get_view_matrix(self) -> list[float]:
        """Build and return a 4×4 column-major view matrix (list of 16 floats)
        compatible with ``glLoadMatrixf`` / ``glMultMatrixf``.
        """
        pos = self.position
        f = self.forward
        r = self.right
        u = self.up

        # Column-major layout.
        return [
            r.x,  u.x, -f.x, 0.0,
            r.y,  u.y, -f.y, 0.0,
            r.z,  u.z, -f.z, 0.0,
            -r.dot(pos), -u.dot(pos), f.dot(pos), 1.0,
        ]

    def get_projection_params(self, aspect: float) -> Tuple[float, float, float, float]:
        """Return ``(fov_y_degrees, aspect, near, far)`` for ``gluPerspective``."""
        return (self.fov, aspect, self.near, self.far)

    # ── ray unprojection ────────────────────────────────────

    def screen_to_ray(self, screen_x: float, screen_y: float,
                      viewport_width: int, viewport_height: int) -> Ray:
        """Convert a screen-space pixel coordinate to a world-space ray.

        Args:
            screen_x:        X in pixels (0 = left edge).
            screen_y:        Y in pixels (0 = top edge, GLFW convention).
            viewport_width:  Framebuffer width in pixels.
            viewport_height: Framebuffer height in pixels.

        Returns:
            A :class:`Ray` originating at the camera position, pointing
            into the scene through the given pixel.
        """
        # Normalised device coords (NDC), range [-1, 1].
        ndc_x = (2.0 * screen_x / viewport_width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / viewport_height)  # flip Y

        # View-space direction on the near plane.
        aspect = viewport_width / max(viewport_height, 1)
        half_h = math.tan(math.radians(self.fov / 2.0))
        half_w = half_h * aspect

        # Direction in camera-local space (forward = -Z).
        local_dir = Vector3(ndc_x * half_w, ndc_y * half_h, -1.0).normalized()

        # Transform to world space using camera basis vectors.
        r = self.right
        u = self.up
        f = self.forward
        world_dir = (r * local_dir.x + u * local_dir.y + f * (-local_dir.z)).normalized()

        return Ray(self.position, world_dir)

    def __repr__(self) -> str:
        return (
            f"Camera(target={self.target}, distance={self.distance:.2f}, "
            f"yaw={self.yaw:.1f}, pitch={self.pitch:.1f})"
        )
