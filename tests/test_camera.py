"""Unit tests for render.camera.Camera.

These tests exercise the pure-math parts of the camera (position,
orbit, zoom, pan, view matrix) without requiring an OpenGL context.
"""

import math
from core.vector3 import Vector3
from render.camera import Camera


class TestCameraPosition:
    def test_default_position_nonzero(self) -> None:
        cam = Camera()
        pos = cam.position
        assert pos.length() > 0

    def test_looking_along_z(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        pos = cam.position
        # yaw=0, pitch=0 → camera is at (0, 0, 5)
        assert pos.approx_equal(Vector3(0, 0, 5), eps=1e-6)

    def test_pitch_raises_camera(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=45.0, distance=5.0)
        assert cam.position.y > 0

    def test_yaw_rotates_horizontally(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=90.0, pitch=0.0, distance=5.0)
        pos = cam.position
        # yaw=90° → camera at (5, 0, 0)
        assert math.isclose(pos.x, 5.0, abs_tol=1e-6)
        assert math.isclose(pos.z, 0.0, abs_tol=1e-6)


class TestCameraDirections:
    def test_forward_toward_target(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        fwd = cam.forward
        # Should point from (0,0,5) toward origin → negative z
        assert fwd.z < 0

    def test_right_perpendicular(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        assert math.isclose(cam.forward.dot(cam.right), 0.0, abs_tol=1e-6)

    def test_up_perpendicular(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        assert math.isclose(cam.forward.dot(cam.up), 0.0, abs_tol=1e-6)


class TestCameraOrbit:
    def test_orbit_changes_yaw(self) -> None:
        cam = Camera(yaw=0.0, pitch=0.0)
        cam.orbit(10.0, 0.0)
        assert cam.yaw != 0.0

    def test_orbit_changes_pitch(self) -> None:
        cam = Camera(yaw=0.0, pitch=0.0)
        cam.orbit(0.0, 10.0)
        assert cam.pitch != 0.0

    def test_pitch_clamped_high(self) -> None:
        cam = Camera(pitch=80.0)
        cam.orbit(0.0, 1000.0)
        assert cam.pitch <= 89.0

    def test_pitch_clamped_low(self) -> None:
        cam = Camera(pitch=-80.0)
        cam.orbit(0.0, -1000.0)
        assert cam.pitch >= -89.0


class TestCameraZoom:
    def test_zoom_in(self) -> None:
        cam = Camera(distance=5.0)
        cam.zoom(1.0)
        assert cam.distance < 5.0

    def test_zoom_out(self) -> None:
        cam = Camera(distance=5.0)
        cam.zoom(-1.0)
        assert cam.distance > 5.0

    def test_zoom_min_clamp(self) -> None:
        cam = Camera(distance=0.2)
        cam.zoom(10000.0)
        assert cam.distance >= 0.1

    def test_zoom_max_clamp(self) -> None:
        cam = Camera(distance=190.0)
        cam.zoom(-10000.0)
        assert cam.distance <= 200.0


class TestCameraPan:
    def test_pan_moves_target(self) -> None:
        cam = Camera(target=Vector3.zero())
        old_target = cam.target
        cam.pan(50.0, 0.0)
        assert not cam.target.approx_equal(old_target)

    def test_pan_vertical(self) -> None:
        cam = Camera(target=Vector3.zero(), yaw=0.0, pitch=0.0, distance=5.0)
        cam.pan(0.0, -50.0)
        # Panning up should raise the target's y.
        assert cam.target.y > 0


class TestCameraViewMatrix:
    def test_matrix_length(self) -> None:
        cam = Camera()
        mat = cam.get_view_matrix()
        assert len(mat) == 16

    def test_projection_params(self) -> None:
        cam = Camera(fov=60.0)
        fov, aspect, near, far = cam.get_projection_params(16 / 9)
        assert fov == 60.0
        assert math.isclose(aspect, 16 / 9)


class TestCameraMisc:
    def test_repr(self) -> None:
        assert "Camera" in repr(Camera())
