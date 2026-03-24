"""Wavefront OBJ file loader.

Supports:
    - ``v`` (vertex positions)
    - ``f`` (triangular and polygonal faces, with optional texture/normal indices)

Polygonal faces with more than 3 vertices are automatically triangulated
using a simple fan triangulation (works for convex polygons).
"""

from __future__ import annotations

import os
from typing import List, Tuple

from core.vector3 import Vector3
from mesh.mesh import Mesh


def load_obj(filepath: str) -> Mesh:
    """Load a Wavefront .obj file and return a :class:`Mesh`.

    Args:
        filepath: Path to the ``.obj`` file.

    Returns:
        A :class:`Mesh` built from the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file contains no geometry.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    vertices: List[Vector3] = []
    faces: List[Tuple[int, int, int]] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            keyword = parts[0]

            if keyword == "v" and len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append(Vector3(x, y, z))

            elif keyword == "f" and len(parts) >= 4:
                # Parse face indices — handles formats:
                #   f 1 2 3
                #   f 1/2 3/4 5/6
                #   f 1/2/3 4/5/6 7/8/9
                #   f 1//3 4//6 7//9
                indices: List[int] = []
                for token in parts[1:]:
                    idx_str = token.split("/")[0]
                    idx = int(idx_str)
                    # OBJ indices are 1-based; convert to 0-based.
                    # Negative indices reference from the end.
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    indices.append(idx)

                # Fan-triangulate polygons with > 3 vertices.
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))

    if not vertices:
        raise ValueError(f"OBJ file contains no vertices: {filepath}")

    return Mesh(vertices, faces)


def _generate_cube_obj_content() -> str:
    """Return the text content of a minimal unit cube OBJ file.

    The cube is axis-aligned, centred at the origin, spanning [-0.5, 0.5]^3.
    """
    lines = [
        "# Unit cube centred at origin",
        "v -0.5 -0.5  0.5",
        "v  0.5 -0.5  0.5",
        "v  0.5  0.5  0.5",
        "v -0.5  0.5  0.5",
        "v -0.5 -0.5 -0.5",
        "v  0.5 -0.5 -0.5",
        "v  0.5  0.5 -0.5",
        "v -0.5  0.5 -0.5",
        "# front",
        "f 1 2 3 4",
        "# back",
        "f 5 8 7 6",
        "# left",
        "f 1 4 8 5",
        "# right",
        "f 2 6 7 3",
        "# top",
        "f 4 3 7 8",
        "# bottom",
        "f 1 5 6 2",
    ]
    return "\n".join(lines) + "\n"


def ensure_default_cube(filepath: str) -> str:
    """Create the default cube OBJ file if it does not already exist.

    Args:
        filepath: Desired path for the cube OBJ.

    Returns:
        The absolute path to the file.
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.isfile(abs_path):
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(_generate_cube_obj_content())
    return abs_path
