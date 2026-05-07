"""Global configuration constants for the application."""

# ── Window ──────────────────────────────────────────────────
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
WINDOW_TITLE: str = "3D Shortest Path – Bachelor Thesis"

# ── Default mesh ────────────────────────────────────────────
DEFAULT_MESH_PATH: str = "assets/cube.obj"

# ── Camera ──────────────────────────────────────────────────
CAMERA_FOV: float = 45.0
CAMERA_NEAR: float = 0.1
CAMERA_FAR: float = 100.0
CAMERA_DISTANCE: float = 5.0

# ── Face sampling ───────────────────────────────────────────
FACE_SAMPLE_BARY_STEPS: int = 3  # barycentric grid resolution per triangle

# ── Pathfinding ─────────────────────────────────────────────
MAX_GRAPH_NODES: int = 50000

# ── Rendering ───────────────────────────────────────────────
PATH_COLOR: tuple = (1.0, 0.2, 0.1)
POINT_COLOR: tuple = (0.1, 1.0, 0.2)
MESH_COLOR: tuple = (0.6, 0.7, 0.8)
WIREFRAME_COLOR: tuple = (0.2, 0.2, 0.2)
POINT_RADIUS: float = 0.05

# Per-algorithm path colours (keyed by strategy.name)
ALGORITHM_COLORS: dict[str, tuple] = {
    "Chen-Han Exact": (0.9, 0.2, 0.9),  # magenta
    "Dijkstra": (1.0, 0.3, 0.1),        # red-orange
    "A*": (0.2, 0.9, 0.3),              # green
    "Visibility Graph": (0.3, 0.5, 1.0), # blue
    "Geodesic Approx": (1.0, 0.7, 0.1), # amber
}

# Distinct start / end marker colours
START_POINT_COLOR: tuple = (0.1, 1.0, 0.2)   # green
END_POINT_COLOR: tuple = (1.0, 0.2, 0.2)     # red

# Debug overlay
RAW_PATH_COLOR: tuple = (0.7, 0.7, 0.7)      # light grey
RAW_PATH_WIDTH: float = 1.5
SAMPLE_NODE_COLOR: tuple = (1.0, 1.0, 0.3)   # yellow
SAMPLE_NODE_SIZE: float = 4.0                 # GL point size in pixels

# ── HUD ─────────────────────────────────────────────────────
HUD_COLOR: tuple = (0.9, 0.9, 0.9)
HUD_TITLE_COLOR: tuple = (1.0, 0.8, 0.2)

# ── Numeric ─────────────────────────────────────────────────
EPSILON: float = 1e-9
