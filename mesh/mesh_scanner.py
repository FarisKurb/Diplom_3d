"""Scan a directory for loadable mesh files (.obj).

Provides a simple catalogue of available mesh assets so the application
can offer a runtime mesh-selection menu.
"""

from __future__ import annotations

import os
from typing import List


def scan_mesh_directory(directory: str) -> List[str]:
    """Return sorted list of ``.obj`` file paths found in *directory*.

    Only files with the ``.obj`` extension (case-insensitive) are included.
    Paths are returned as absolute paths.

    Args:
        directory: Path to the folder to scan.

    Returns:
        A sorted list of absolute ``.obj`` file paths.  Empty if the
        directory does not exist or contains no ``.obj`` files.
    """
    if not os.path.isdir(directory):
        return []

    result: List[str] = []
    for entry in os.listdir(directory):
        if entry.lower().endswith(".obj"):
            result.append(os.path.abspath(os.path.join(directory, entry)))
    result.sort()
    return result


def mesh_display_name(filepath: str) -> str:
    """Return a human-readable display name from a mesh file path.

    Strips the directory and extension, e.g.::

        ``"assets/Melon.obj"`` → ``"Melon"``
    """
    return os.path.splitext(os.path.basename(filepath))[0]
