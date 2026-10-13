#!/usr/bin/env python3
"""
OCC (Occupancy) Prediction Bin File 3D Visualizer

Loads .bin files saved by SaveOccPredBin and visualizes the 3D occupancy grid
with class colors from colors_map.

Usage:
    python occ_vis.py

Dependencies:
    pip install numpy PyQt5 vispy
"""

import sys
import os
import csv
import numpy as np

# ---------------------------------------------------------------------------
#  Try importing GUI & 3D libraries — with helpful error messages
# ---------------------------------------------------------------------------
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox,
        QCheckBox, QGroupBox, QScrollArea, QStatusBar,
        QSplitter, QFrame, QGridLayout, QFormLayout,
        QSpinBox, QDoubleSpinBox,
        QSlider, QComboBox, QApplication
    )
except ImportError:
    print("PyQt5 is required. Install with: pip install PyQt5")
    sys.exit(1)

try:
    import vispy.app
    from vispy import scene
    from vispy.scene import visuals
    from vispy.visuals.transforms import STTransform
except ImportError:
    print("vispy is required. Install with: pip install vispy")
    sys.exit(1)

try:
    import OpenGL.GL as gl
except ImportError:
    gl = None  # will be caught later if needed

try:
    import imageio
except ImportError:
    imageio = None

# ---------------------------------------------------------------------------
#  Class definitions & colors  (matching colors_map in the C++ code)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "others",               # 0
    "barrier",              # 1
    "bicycle",              # 2
    "bus",                  # 3
    "car",                  # 4
    "construction_vehicle", # 5
    "motorcycle",           # 6
    "pedestrian",           # 7
    "traffic_cone",         # 8
    "trailer",              # 9
    "truck",                # 10
    "driveable_surface",    # 11
    "other_flat",           # 12
    "sidewalk",             # 13
    "terrain",              # 14
    "manmade",              # 15
    "vegetation",           # 16
    "free",                 # 17
]

# BGR(A) colors as in the C++ code — convert to RGB normalized for vispy
COLORS_BGR = [
    (0, 0, 0),              # 0  others         -> black
    (255, 120, 50),         # 1  barrier        -> orange
    (255, 192, 203),        # 2  bicycle        -> pink
    (255, 255, 0),          # 3  bus            -> yellow
    (0, 150, 245),          # 4  car            -> light blue
    (0, 255, 255),          # 5  construction   -> cyan
    (200, 180, 0),          # 6  motorcycle     -> dark yellow
    (255, 0, 0),            # 7  pedestrian     -> red
    (255, 240, 150),        # 8  traffic_cone   -> light orange
    (135, 60, 0),           # 9  trailer        -> brown
    (160, 32, 240),         # 10 truck          -> purple
    (255, 0, 255),          # 11 driveable      -> magenta
    (175, 0, 75),           # 12 other_flat     -> dark red
    (75, 0, 75),            # 13 sidewalk       -> dark purple
    (150, 240, 80),         # 14 terrain        -> light green
    (230, 230, 250),        # 15 manmade        -> lavender
    (0, 175, 0),            # 16 vegetation     -> green
    (255, 255, 255),        # 17 free           -> white
]

# Convert BGR → RGB and normalize to [0,1]
COLORS_RGB = [(b / 255.0, g / 255.0, r / 255.0) for (b, g, r) in COLORS_BGR]
NUM_CLASSES = len(CLASS_NAMES)

# Default hidden classes (background / free space)
DEFAULT_HIDDEN = {17}  # hide "free"

# Default shape — adjust as needed; the bin file only stores raw data
DEFAULT_H = 200
DEFAULT_W = 200
DEFAULT_Z = 16

# Default voxel size (metres per voxel) and grid range in ego frame
# OCC grid: 200×200×16, covering [-40,40]×[-40,40]×[-1.0,5.4] in ego
DEFAULT_VOXEL_SIZE = 0.4
DEFAULT_GRID_X_MIN = -40.0
DEFAULT_GRID_X_MAX = 40.0
DEFAULT_GRID_Y_MIN = -40.0
DEFAULT_GRID_Y_MAX = 40.0
DEFAULT_GRID_Z_MIN = -1.0
DEFAULT_GRID_Z_MAX = 5.4

# Default paths (relative to the script's directory)
DEFAULT_FRAMES_LST = "frames.lst"
DEFAULT_EGO_POSE_CSV = "data/val_ego_pose.csv"
DEFAULT_BBOXES_JSON = "data/val_instance_bboxes.json"

# Map nuScenes category_name → CLASS_NAMES index
CATEGORY_TO_CLASS = {
    "vehicle.car": 4,
    "vehicle.truck": 10,
    "vehicle.bus": 3,
    "vehicle.trailer": 9,
    "vehicle.construction": 5,
    "vehicle.motorcycle": 6,
    "vehicle.bicycle": 2,
    "human.pedestrian": 7,
    "movable_object.trafficcone": 8,
    "movable_object.barrier": 1,
    "flat.driveable_surface": 11,
    "flat.other": 12,
    "flat.sidewalk": 13,
    "flat.terrain": 14,
    "static.manmade": 15,
    "static.vegetation": 16,
}


# ---------------------------------------------------------------------------
#  Data loading utilities
# ---------------------------------------------------------------------------
def load_bin(filepath: str, H: int, W: int, Z: int) -> np.ndarray:
    """
    Load a .bin file produced by SaveOccPredBin.
    Returns int16 numpy array reshaped to (H, W, Z).
    """
    expected_size = H * W * Z
    data = np.fromfile(filepath, dtype=np.int16)
    if data.size != expected_size:
        raise ValueError(
            f"File size mismatch: file has {data.size} elements, "
            f"but H×W×Z = {H}×{W}×{Z} = {expected_size}. "
            f"Please check the shape parameters."
        )
    return data.reshape((H, W, Z)).astype(np.int32)


def load_npz(filepath: str):
    """
    Load an Occ3D ground-truth .npz file.

    Expected keys:
        semantics   : (H, W, Z) uint8  — per-voxel semantic class id (0..17)
        mask_lidar  : (H, W, Z) uint8  — 1 where the voxel is observed by LiDAR
        mask_camera : (H, W, Z) uint8  — 1 where the voxel is observed by camera

    Returns
    -------
    semantics : np.ndarray  shape (H, W, Z), dtype int32
    masks : dict  {name: bool ndarray (H, W, Z)} for each present mask key
    """
    data = np.load(filepath)
    if "semantics" not in data.files:
        raise ValueError(
            f"No 'semantics' array in {filepath}. "
            f"Available keys: {list(data.files)}"
        )
    semantics = data["semantics"].astype(np.int32)
    if semantics.ndim != 3:
        raise ValueError(
            f"Expected 3D 'semantics', got shape {semantics.shape}"
        )
    masks = {}
    for key in ("mask_lidar", "mask_camera"):
        if key in data.files:
            masks[key] = data[key].astype(bool)
    return semantics, masks


def build_voxel_mesh(
    volume: np.ndarray, hidden_classes: set, stride: int = 1,
    filter_mask: np.ndarray = None,
):
    """
    Build per-voxel cube mesh data for solid 3D rendering.

    Parameters
    ----------
    volume : np.ndarray  shape (H, W, Z), dtype int32
    hidden_classes : set of int
    stride : int  stride for downsampling (1 = full resolution)

    Returns
    -------
    vertices : np.ndarray  shape (N*36, 3), float32
        Non-indexed triangle vertices (12 triangles × 3 verts per cube).
    colors : np.ndarray  shape (N*36, 3), float32
        Per-vertex RGB colors in [0, 1].
    """
    H, W, Z = volume.shape
    mask = np.isin(volume, list(hidden_classes), invert=True)
    if filter_mask is not None:
        mask = mask & filter_mask

    if stride > 1:
        mask = mask[::stride, ::stride, ::stride]
        vol_sub = volume[::stride, ::stride, ::stride]
    else:
        vol_sub = volume

    idxs = np.argwhere(mask)  # (N, 3) rows of (h, w, z)
    if idxs.size == 0:
        empty_f32 = np.zeros((0, 3), dtype=np.float32)
        return empty_f32, np.zeros((0, 3), dtype=np.uint32), empty_f32

    # Centers in display coords: (h, w, z) → (X=forward, Y=left, Z=up)
    centers = idxs.astype(np.float32)  # already (N,3) in (h, w, z) order
    N = len(centers)

    # ---- 36 vertex offsets for a unit cube (12 triangles × 3 verts) ----
    _cube_verts = np.array([
        # back  (z = -0.5)
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        # front (z = +0.5)
        [-0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
        [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
        # bottom (y = -0.5)
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5],
        # top (y = +0.5)
        [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        [ 0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
        # left (x = -0.5)
        [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5, -0.5,  0.5],
        # right (x = +0.5)
        [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
    ], dtype=np.float32)

    # Vectorised: centers[:, None] + offsets[None, :] → (N, 36, 3)
    vertices = (centers[:, np.newaxis, :] + _cube_verts[np.newaxis, :, :])
    vertices = vertices.reshape(-1, 3).astype(np.float32)

    # Explicit faces: every 3 consecutive vertices = 1 triangle
    faces = np.arange(N * 36, dtype=np.uint32).reshape(-1, 3)

    # Per-vertex colour repeated 36 times → flat (N*36, 4) RGBA
    class_ids = vol_sub[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
    col_rgb = np.array([COLORS_RGB[c] for c in class_ids], dtype=np.float32)
    col_rgb = np.repeat(col_rgb, 36, axis=0).astype(np.float32)
    alpha = np.ones((len(col_rgb), 1), dtype=np.float32)
    colors = np.concatenate([col_rgb, alpha], axis=-1)

    return vertices, faces, colors


def build_cube_edges(centers: np.ndarray, scale: float = 1.0):
    """
    Build wireframe edge lines for cube outlines.

    Parameters
    ----------
    centers : np.ndarray  shape (N, 3), float32 — cube centres (x, y, z)
    scale : float  scale factor applied to the unit cube

    Returns
    -------
    edge_verts : np.ndarray  shape (N*24, 3), float32
        Line segment vertices (12 edges × 2 verts per cube).
    edge_colors : np.ndarray  shape (N*24, 4), float32 RGBA
    """
    if centers.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )

    h = 0.5 * scale
    c = np.array([
        [-h, -h, -h], [ h, -h, -h],
        [ h, -h, -h], [ h,  h, -h],
        [ h,  h, -h], [-h,  h, -h],
        [-h,  h, -h], [-h, -h, -h],
        # front face
        [-h, -h,  h], [ h, -h,  h],
        [ h, -h,  h], [ h,  h,  h],
        [ h,  h,  h], [-h,  h,  h],
        [-h,  h,  h], [-h, -h,  h],
        # connecting edges
        [-h, -h, -h], [-h, -h,  h],
        [ h, -h, -h], [ h, -h,  h],
        [ h,  h, -h], [ h,  h,  h],
        [-h,  h, -h], [-h,  h,  h],
    ], dtype=np.float32)

    N = len(centers)
    edge_verts = (centers[:, np.newaxis, :] + c[np.newaxis, :, :])
    edge_verts = edge_verts.reshape(-1, 3).astype(np.float32)

    dark = np.array([0.15, 0.15, 0.15, 0.50], dtype=np.float32)
    edge_colors = np.tile(dark, (N * 24, 1))

    return edge_verts, edge_colors


# ---------------------------------------------------------------------------
#  Ego-pose & multi-frame utilities
# ---------------------------------------------------------------------------
def load_frames_list(filepath: str) -> list:
    """
    Parse frames.lst.
    Each line:  sample_000123_abcd1234
    Returns list of (frame_index: int, sample_token_prefix: str).
    """
    frames = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_")
            # Expected format: sample_<NNNNNN>_<token_prefix8>
            if len(parts) >= 3:
                try:
                    frame_idx = int(parts[1])
                except ValueError:
                    continue
                token_prefix = parts[2]
                frames.append((frame_idx, token_prefix))
    return frames


def load_ego_poses_csv(csv_path: str) -> dict:
    """
    Parse val_ego_pose.csv.
    Returns dict:  sample_token (full 32-char) → {
        'tx','ty','tz','qw','qx','qy','qz',
        'scene_name','timestamp'
    }
    Also builds a prefix→token index for fast lookup.
    """
    poses = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row["sample_token"]
            poses[token] = {
                "tx": float(row["tx"]),
                "ty": float(row["ty"]),
                "tz": float(row["tz"]),
                "qw": float(row["qw"]),
                "qx": float(row["qx"]),
                "qy": float(row["qy"]),
                "qz": float(row["qz"]),
                "scene_name": row.get("scene_name", ""),
                "timestamp": int(row.get("timestamp", 0)),
            }
    return poses


def _build_token_prefix_index(poses: dict) -> dict:
    """Build a mapping from first-8-char prefix → list of full tokens."""
    idx = {}
    for token in poses:
        prefix = token[:8]
        idx.setdefault(prefix, []).append(token)
    return idx


def quat_to_rot_mat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3×3 rotation matrix."""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    return R


def voxel_indices_to_ego(
    idxs: np.ndarray,
    voxel_size: float,
    grid_x_range: tuple,
    grid_y_range: tuple,
    grid_z_range: tuple,
) -> np.ndarray:
    """
    Convert voxel indices (h, w, z) to ego-frame metric coordinates (x, y, z).

    Mapping convention (nuScenes ego frame):
      - index h (dim 0, 200 bins) → ego X (forward)   [-40, 40]
      - index w (dim 1, 200 bins) → ego Y (left)      [-40, 40]
      - index z (dim 2,  16 bins) → ego Z (up)        [-1.0, 5.4]

    Parameters
    ----------
    idxs : (N, 3) int array, columns = (h, w, z)
    voxel_size : float, metres per voxel
    grid_x_range : (x_min, x_max) in ego metres (forward)
    grid_y_range : (y_min, y_max) in ego metres (left)
    grid_z_range : (z_min, z_max) in ego metres (up)

    Returns
    -------
    ego_pts : (N, 3) float32 array, columns = (ego_x, ego_y, ego_z)
    """
    h_idx = idxs[:, 0].astype(np.float32)
    w_idx = idxs[:, 1].astype(np.float32)
    z_idx = idxs[:, 2].astype(np.float32)

    ego_x = h_idx * voxel_size + grid_x_range[0] + voxel_size * 0.5
    ego_y = w_idx * voxel_size + grid_y_range[0] + voxel_size * 0.5
    ego_z = z_idx * voxel_size + grid_z_range[0] + voxel_size * 0.5

    return np.column_stack([ego_x, ego_y, ego_z]).astype(np.float32)


def transform_vertices_ego_to_global(
    vertices: np.ndarray,
    translation: tuple,
    rotation_mat: np.ndarray,
) -> np.ndarray:
    """
    Transform vertices from ego frame to global frame.

    vertices: (N, 3) in ego (x_forward, y_left, z_up)
    translation: (tx, ty, tz) ego→global translation
    rotation_mat: 3×3 ego→global rotation matrix

    Returns: (N, 3) in global coordinates.
    """
    T = np.array(translation, dtype=np.float32)
    return (vertices @ rotation_mat.T) + T


def build_ego_marker(translation: tuple, marker_size: float = 2.0):
    """
    Build a small coloured coordinate-frame marker at an ego position.
    Returns (vertices, faces, colors) for a tri-axis tripod.
    """
    cx, cy, cz = translation
    s = marker_size
    verts = np.array([
        [cx, cy, cz], [cx + s, cy, cz],  # +X red
        [cx, cy, cz], [cx, cy + s, cz],  # +Y green
        [cx, cy, cz], [cx, cy, cz + s],  # +Z blue
    ], dtype=np.float32)
    colors = np.array([
        [1, 0.2, 0.2, 1], [1, 0.2, 0.2, 1],
        [0.2, 1, 0.2, 1], [0.2, 1, 0.2, 1],
        [0.3, 0.5, 1, 1], [0.3, 0.5, 1, 1],
    ], dtype=np.float32)
    return verts, colors


def match_frames_to_poses(
    frames: list, poses: dict, prefix_index: dict
) -> list:
    """
    Match (frame_idx, token_prefix) entries to full ego poses.
    Returns list of dicts: {
        'frame_idx', 'sample_token', 'pose', 'bin_name'
    }
    """
    matched = []
    for frame_idx, prefix in frames:
        candidates = prefix_index.get(prefix, [])
        if not candidates:
            continue
        # Use the first match (prefix is unique enough for nuScenes val)
        token = candidates[0]
        matched.append({
            "frame_idx": frame_idx,
            "sample_token": token,
            "pose": poses[token],
            "bin_name": f"occ_rank0_{frame_idx:06d}.bin",
        })
    return matched


def load_bboxes_json(json_path: str) -> dict:
    """
    Load val_instance_bboxes.json.
    Returns dict:  sample_token → list of annotation dicts.
    """
    import json as _json
    with open(json_path, "r") as f:
        return _json.load(f)


def build_bbox_wireframe(
    translation: tuple,
    size: tuple,
    rotation_quat: tuple,
    color_rgba: tuple,
) -> tuple:
    """
    Build a 3D bounding-box wireframe as line segments.

    Parameters
    ----------
    translation : (cx, cy, cz)  global centre
    size : (w, l, h)  width (y-axis), length (x-axis), height (z-axis)
    rotation_quat : (qw, qx, qy, qz)  global rotation
    color_rgba : (r, g, b, a) in [0,1]

    Returns
    -------
    verts : (24, 3) float32  — 12 edges × 2 verts
    colors : (24, 4) float32
    """
    w, l, h = size
    # Corners in local frame (x=forward/length, y=left/width, z=up/height)
    hl, hw, hh = l * 0.5, w * 0.5, h * 0.5
    corners = np.array([
        [-hl, -hw, -hh], [ hl, -hw, -hh], [ hl,  hw, -hh], [-hl,  hw, -hh],
        [-hl, -hw,  hh], [ hl, -hw,  hh], [ hl,  hw,  hh], [-hl,  hw,  hh],
    ], dtype=np.float32)

    # Rotate
    qw, qx, qy, qz = rotation_quat
    R = quat_to_rot_mat(qw, qx, qy, qz).astype(np.float32)
    corners = corners @ R.T

    # Translate
    cx, cy, cz = translation
    corners += np.array([cx, cy, cz], dtype=np.float32)

    # 12 edges
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],   # bottom
        [4,5],[5,6],[6,7],[7,4],   # top
        [0,4],[1,5],[2,6],[3,7],   # vertical
    ], dtype=np.int32)

    verts = np.empty((24, 3), dtype=np.float32)
    for i, (a, b) in enumerate(edges):
        verts[i*2] = corners[a]
        verts[i*2+1] = corners[b]

    cols = np.tile(np.array(color_rgba, dtype=np.float32), (24, 1))
    return verts, cols


# ---------------------------------------------------------------------------
#  Custom VisPy canvas embedded in Qt
# ---------------------------------------------------------------------------
class OccCanvas(scene.SceneCanvas):
    """3D canvas that displays the OCC voxel grid as solid cubes."""

    def __init__(self, parent=None):
        scene.SceneCanvas.__init__(self, keys="interactive", show=False,
                                   bgcolor="#1a1a2e", parent=parent)
        self.unfreeze()

        # Camera — TurntableCamera, up="+z" (Z=up) to match H→X/W→Y/Z→Z
        self.view = self.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=45, elevation=25, azimuth=45, distance=300,
            up="+z"
        )

        # Set GL state at the view level so it persists across focus changes
        self.view.set_gl_state(depth_test=True, cull_face=False,
                               blend=False, depth_mask=True)

        # Store the volume dimensions for axis sizing (set on first render)
        self._vol_H = DEFAULT_H
        self._vol_W = DEFAULT_W
        self._vol_Z = DEFAULT_Z

        # Mesh — solid per-voxel cubes, flat colour, no lighting
        self._mesh = visuals.Mesh(shading=None)
        self._mesh.set_gl_state(depth_test=True, cull_face=False,
                                blend=False, depth_mask=True)
        self.view.add(self._mesh)

        # Wireframe overlay — dark thin edges between cubes
        self._wire = visuals.Line(
            connect='segments', method='gl',
            color=(0.15, 0.15, 0.15, 0.50),
        )
        self._wire.set_gl_state(depth_test=True, depth_mask=True)
        self.view.add(self._wire)

        # Cached for resizing
        self._vertices = None
        self._colors = None
        self._ego_marker_lines = []  # track ego-marker Line visuals
        self._bbox_lines = []  # track bbox wireframe Line visuals
        self._bbox_labels = []  # track bbox ID Text visuals

        # Grid
        grid = visuals.GridLines(color=(0.3, 0.3, 0.3, 0.5))
        self.view.add(grid)

        # Axis — custom lines with arrow cones; lengths set at first render
        self._axes_built = False

        # Ensure GL state is reapplied on every draw (fixes focus-loss blur)
        self.events.draw.connect(self._on_draw_gl_state)

        self.freeze()

    def _on_draw_gl_state(self, event):
        """Re-apply depth test & depth mask on every frame.
        This is needed because vispy may lose per-visual GL state
        when the canvas loses and regains focus."""
        if gl is None:
            return
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthFunc(gl.GL_LESS)

    def _build_axes(self):
        """Draw X(H)/Y(W)/Z(Z) axes with arrow cones at positive ends."""
        # Use the shortest dimension for all three axes so they are
        # visually comparable and stay inside the visible bounds.
        L = float(min(self._vol_H, self._vol_W, self._vol_Z))
        Lx = Ly = Lz = L

        # --- axis lines ---
        axis_verts = np.array([
            [0, 0, 0], [Lx,   0,   0],   # +X red   (H forwards)
            [0, 0, 0], [  0, Ly,   0],   # +Y green (W left)
            [0, 0, 0], [  0,  0,  Lz],   # +Z blue  (Z up)
        ], dtype=np.float32)
        axis_colors = np.array([
            [1, 0.2, 0.2, 1], [1, 0.2, 0.2, 1],
            [0.2, 1, 0.2, 1], [0.2, 1, 0.2, 1],
            [0.3, 0.5, 1, 1], [0.3, 0.5, 1, 1],
        ], dtype=np.float32)

        ax_line = visuals.Line(
            axis_verts, color=axis_colors,
            connect='segments', method='gl', width=5.0,
            parent=self.view.scene,
        )
        ax_line.set_gl_state(depth_test=False)
        ax_line.order = 100

        # --- arrow cones ---
        a = 1.0
        cone = np.array([
            [0, 0, a], [ a*0.4,  a*0.4, 0], [-a*0.4,  a*0.4, 0],
            [0, 0, a], [-a*0.4,  a*0.4, 0], [-a*0.4, -a*0.4, 0],
            [0, 0, a], [-a*0.4, -a*0.4, 0], [ a*0.4, -a*0.4, 0],
            [0, 0, a], [ a*0.4, -a*0.4, 0], [ a*0.4,  a*0.4, 0],
        ], dtype=np.float32)

        def make_arrow(axis_id):
            v = cone.copy()
            if axis_id == 'x':
                v = v[:, [2, 0, 1]]   # z→x
                tip = np.array([[Lx, 0, 0]], dtype=np.float32)
            elif axis_id == 'y':
                v = v[:, [0, 2, 1]]   # z→y
                tip = np.array([[0, Ly, 0]], dtype=np.float32)
            else:
                tip = np.array([[0, 0, Lz]], dtype=np.float32)
            v += tip
            return v

        col_red   = np.array([1, 0.2, 0.2, 1], dtype=np.float32)
        col_green = np.array([0.2, 1, 0.2, 1], dtype=np.float32)
        col_blue  = np.array([0.3, 0.5, 1, 1], dtype=np.float32)

        for pts, c in [(make_arrow('x'), col_red),
                       (make_arrow('y'), col_green),
                       (make_arrow('z'), col_blue)]:
            f = np.arange(len(pts), dtype=np.uint32).reshape(-1, 3)
            clr_v = np.tile(c, (len(pts), 1)).astype(np.float32)
            m = visuals.Mesh(pts, faces=f, vertex_colors=clr_v, shading=None,
                             parent=self.view.scene)
            m.set_gl_state(depth_test=False)
            m.order = 100

        # --- labels ---
        pad = 2.0
        labels = [
            ('X(H)', Lx+pad, 0, 0, col_red),
            ('Y(W)', 0, Ly+pad, 0, col_green),
            ('Z(Z)', 0, 0, Lz+pad, col_blue),
        ]
        for text, px, py, pz, c in labels:
            t = visuals.Text(text, pos=(px, py, pz), color=c[:3],
                             font_size=14, bold=True, anchor_x='center',
                             anchor_y='center', parent=self.view.scene)
            t.set_gl_state(depth_test=False)
            t.order = 100

        self._axes_built = True

    def set_data(self, vertices: np.ndarray, faces: np.ndarray,
                 colors: np.ndarray, edge_verts: np.ndarray,
                 edge_colors: np.ndarray, H: int, W: int, Z: int,
                 ego_markers_verts=None, ego_markers_colors=None):
        """Upload cube mesh + wireframe edges + optional ego markers."""
        if not self._axes_built:
            self._vol_H = H
            self._vol_W = W
            self._vol_Z = Z
            self._build_axes()
        self._vertices = vertices
        self._colors = colors

        # --- Remove old ego markers ---
        for line in self._ego_marker_lines:
            line.parent = None
        self._ego_marker_lines.clear()

        if vertices.size == 0:
            self._mesh.set_data(np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.uint32))
            self._wire.set_data(np.zeros((0, 3)))
            self._draw_ego_markers(ego_markers_verts, ego_markers_colors)
            return
        self._mesh.set_data(vertices, faces=faces, vertex_colors=colors)
        self._wire.set_data(edge_verts, color=edge_colors)
        self.view.camera.set_range(
            (vertices[:, 0].min(), vertices[:, 0].max()),
            (vertices[:, 1].min(), vertices[:, 1].max()),
            (vertices[:, 2].min(), vertices[:, 2].max()),
        )
        self._draw_ego_markers(ego_markers_verts, ego_markers_colors)

    def _draw_ego_markers(self, markers_verts, markers_colors):
        """Draw ego-position markers as small axis tripods."""
        if markers_verts is None or markers_verts.size == 0:
            return
        marker_line = visuals.Line(
            markers_verts, color=markers_colors,
            connect='segments', method='gl', width=3.0,
            parent=self.view.scene,
        )
        marker_line.set_gl_state(depth_test=False)
        marker_line.order = 99
        self._ego_marker_lines.append(marker_line)

    def set_bbox_data(self, bbox_verts, bbox_colors,
                       label_positions=None, label_texts=None, label_colors=None):
        """Draw 3D bounding-box wireframes + optional ID labels."""
        # --- Remove old bbox lines ---
        for line in self._bbox_lines:
            line.parent = None
        self._bbox_lines.clear()

        for lbl in self._bbox_labels:
            lbl.parent = None
        self._bbox_labels.clear()

        if bbox_verts is not None and bbox_verts.size > 0:
            bbox_line = visuals.Line(
                bbox_verts, color=bbox_colors,
                connect='segments', method='gl', width=2.0,
                parent=self.view.scene,
            )
            bbox_line.set_gl_state(depth_test=False)
            bbox_line.order = 98
            self._bbox_lines.append(bbox_line)

        # --- ID labels ---
        if label_positions is not None and len(label_positions) > 0:
            for pos, txt, col in zip(label_positions, label_texts, label_colors):
                t = visuals.Text(
                    txt, pos=tuple(pos), color=col[:3],
                    font_size=640, bold=True,
                    anchor_x='center', anchor_y='center',
                    parent=self.view.scene,
                )
                t.set_gl_state(depth_test=False)
                t.order = 97
                self._bbox_labels.append(t)

    def clear_data(self):
        self._mesh.set_data(np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.uint32))
        self._wire.set_data(np.zeros((0, 3)))
        for line in self._ego_marker_lines:
            line.parent = None
        self._ego_marker_lines.clear()
        for line in self._bbox_lines:
            line.parent = None
        self._bbox_lines.clear()
        for lbl in self._bbox_labels:
            lbl.parent = None
        self._bbox_labels.clear()


# ---------------------------------------------------------------------------
#  Color legend widget
# ---------------------------------------------------------------------------
class LegendWidget(QScrollArea):
    """Scrollable legend showing class colors and names with toggle checkboxes."""

    toggled = QtCore.pyqtSignal(int, bool)  # class_id, visible

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._checkboxes: dict[int, QCheckBox] = {}

        container = QWidget()
        self._layout = QGridLayout(container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(2)
        self._layout.setAlignment(QtCore.Qt.AlignTop)

        for cid, name in enumerate(CLASS_NAMES):
            r, g, b = COLORS_RGB[cid]
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #555;"
            )

            # Checkbox
            cb = QCheckBox(name)
            cb.setChecked(cid not in DEFAULT_HIDDEN)
            cb.setStyleSheet("color: #ccc; font-size: 11px;")
            cb.toggled.connect(lambda checked, cid=cid: self.toggled.emit(cid, checked))

            self._checkboxes[cid] = cb
            row = cid
            self._layout.addWidget(swatch, row, 0)
            self._layout.addWidget(cb, row, 1)

        self.setWidget(container)
        self.setMaximumWidth(220)

    def get_hidden_classes(self) -> set:
        """Return the set of class IDs that are currently unchecked."""
        return {cid for cid, cb in self._checkboxes.items() if not cb.isChecked()}


# ---------------------------------------------------------------------------
#  Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCC Bin 3D Visualizer")
        self.resize(1400, 900)
        self._volume: np.ndarray | None = None
        self._filepath: str = ""
        self._mask: np.ndarray | None = None   # GT mask filter (NPZ)
        self._npz_masks: dict = {}             # name → bool ndarray
        # Multi-frame state
        self._multiframe_data: list = []  # list of matched frame dicts
        self._ego_poses: dict = {}
        self._multiframe_mode: bool = False
        self._bboxes_data: dict = {}  # sample_token → list of annotations

        # ---------- central widget ----------
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ---------- 3D canvas ----------
        self._canvas = OccCanvas()
        self._canvas.create_native()
        self._canvas.native.setParent(self)

        # ---------- side panel ----------
        side = QFrame()
        side.setMaximumWidth(280)
        side.setStyleSheet("QFrame { background: #252540; }")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(6)

        # -- File section --
        file_grp = QGroupBox("File")
        file_grp.setStyleSheet(self._group_style())
        file_grp_layout = QVBoxLayout(file_grp)

        btn_open = QPushButton("Open .bin File")
        btn_open.setStyleSheet(self._btn_style("#44aa99"))
        btn_open.clicked.connect(self._on_open)
        file_grp_layout.addWidget(btn_open)

        btn_open_npz = QPushButton("Open .npz GT File")
        btn_open_npz.setStyleSheet(self._btn_style("#7755cc"))
        btn_open_npz.clicked.connect(self._on_open_npz)
        file_grp_layout.addWidget(btn_open_npz)

        self._lbl_file = QLabel("No file loaded")
        self._lbl_file.setStyleSheet("color: #888; font-size: 10px;")
        self._lbl_file.setWordWrap(True)
        file_grp_layout.addWidget(self._lbl_file)

        side_layout.addWidget(file_grp)

        # -- GT Mask selection (NPZ) --
        mask_grp = QGroupBox("GT Mask (NPZ)")
        mask_grp.setStyleSheet(self._group_style())
        mask_layout = QVBoxLayout(mask_grp)
        self._cmb_mask = QComboBox()
        self._cmb_mask.addItems([
            "None",
            "mask_lidar",
            "mask_camera",
            "Both (union)",
            "Both (intersection)",
        ])
        self._cmb_mask.setStyleSheet(
            "QComboBox { background: #2a2a3e; color: #ddd; }"
        )
        self._cmb_mask.currentTextChanged.connect(self._on_mask_changed)
        self._cmb_mask.setEnabled(False)
        mask_layout.addWidget(self._cmb_mask)
        side_layout.addWidget(mask_grp)

        # -- Multi-Frame section --
        mf_grp = QGroupBox("Multi-Frame (Ego-Pose Stitch)")
        mf_grp.setStyleSheet(self._group_style())
        mf_layout = QFormLayout(mf_grp)

        # Data directory
        dir_row = QHBoxLayout()
        self._cmb_data_dir = QComboBox()
        self._cmb_data_dir.setEditable(True)
        self._cmb_data_dir.setStyleSheet(
            "QComboBox { background: #2a2a3e; color: #ddd; }"
        )
        self._populate_data_dirs()
        btn_browse_dir = QPushButton("...")
        btn_browse_dir.setFixedWidth(30)
        btn_browse_dir.setStyleSheet(self._btn_style("#555", "10px"))
        btn_browse_dir.clicked.connect(self._on_browse_data_dir)
        dir_row.addWidget(self._cmb_data_dir)
        dir_row.addWidget(btn_browse_dir)
        mf_layout.addRow("Data Dir:", dir_row)

        # Frames list file
        frames_row = QHBoxLayout()
        self._txt_frames_lst = QLineEdit(DEFAULT_FRAMES_LST)
        self._txt_frames_lst.setStyleSheet(
            "QLineEdit { background: #2a2a3e; color: #ddd; }"
        )
        btn_browse_frames = QPushButton("...")
        btn_browse_frames.setFixedWidth(30)
        btn_browse_frames.setStyleSheet(self._btn_style("#555", "10px"))
        btn_browse_frames.clicked.connect(self._on_browse_frames_lst)
        frames_row.addWidget(self._txt_frames_lst)
        frames_row.addWidget(btn_browse_frames)
        mf_layout.addRow("Frames.lst:", frames_row)

        # Ego pose CSV
        pose_row = QHBoxLayout()
        self._txt_pose_csv = QLineEdit(DEFAULT_EGO_POSE_CSV)
        self._txt_pose_csv.setStyleSheet(
            "QLineEdit { background: #2a2a3e; color: #ddd; }"
        )
        btn_browse_pose = QPushButton("...")
        btn_browse_pose.setFixedWidth(30)
        btn_browse_pose.setStyleSheet(self._btn_style("#555", "10px"))
        btn_browse_pose.clicked.connect(self._on_browse_pose_csv)
        pose_row.addWidget(self._txt_pose_csv)
        pose_row.addWidget(btn_browse_pose)
        mf_layout.addRow("Ego Pose CSV:", pose_row)

        # Start frame & num frames
        range_row = QHBoxLayout()
        self._spin_start_frame = QSpinBox()
        self._spin_start_frame.setRange(0, 999999)
        self._spin_start_frame.setValue(0)
        self._spin_start_frame.setToolTip("Starting frame index in frames.lst")
        self._spin_num_frames = QSpinBox()
        self._spin_num_frames.setRange(1, 500)
        self._spin_num_frames.setValue(1)
        self._spin_num_frames.setToolTip("Number of consecutive frames to load")
        range_row.addWidget(QLabel("Start:"))
        range_row.addWidget(self._spin_start_frame)
        range_row.addWidget(QLabel("N:"))
        range_row.addWidget(self._spin_num_frames)
        mf_layout.addRow("Frames:", range_row)

        # Voxel size
        self._spin_voxel_size = QDoubleSpinBox()
        self._spin_voxel_size.setRange(0.01, 10.0)
        self._spin_voxel_size.setValue(DEFAULT_VOXEL_SIZE)
        self._spin_voxel_size.setSingleStep(0.1)
        self._spin_voxel_size.setDecimals(3)
        self._spin_voxel_size.setToolTip("Metres per voxel")
        mf_layout.addRow("Voxel Size (m):", self._spin_voxel_size)

        # Grid range (ego frame)
        grid_row = QHBoxLayout()
        self._spin_grid_xmin = QDoubleSpinBox()
        self._spin_grid_xmin.setRange(-500, 500)
        self._spin_grid_xmin.setValue(DEFAULT_GRID_X_MIN)
        self._spin_grid_xmin.setToolTip("Ego X min (forward, m) e.g. -40")
        self._spin_grid_xmax = QDoubleSpinBox()
        self._spin_grid_xmax.setRange(-500, 500)
        self._spin_grid_xmax.setValue(DEFAULT_GRID_X_MAX)
        self._spin_grid_xmax.setToolTip("Ego X max (forward, m) e.g. 40")
        grid_row.addWidget(QLabel("X:"))
        grid_row.addWidget(self._spin_grid_xmin)
        grid_row.addWidget(self._spin_grid_xmax)
        mf_layout.addRow("Ego X range:", grid_row)

        grid_row2 = QHBoxLayout()
        self._spin_grid_ymin = QDoubleSpinBox()
        self._spin_grid_ymin.setRange(-500, 500)
        self._spin_grid_ymin.setValue(DEFAULT_GRID_Y_MIN)
        self._spin_grid_ymin.setToolTip("Ego Y min (left, m) e.g. -40")
        self._spin_grid_ymax = QDoubleSpinBox()
        self._spin_grid_ymax.setRange(-500, 500)
        self._spin_grid_ymax.setValue(DEFAULT_GRID_Y_MAX)
        self._spin_grid_ymax.setToolTip("Ego Y max (left, m) e.g. 40")
        grid_row2.addWidget(QLabel("Y:"))
        grid_row2.addWidget(self._spin_grid_ymin)
        grid_row2.addWidget(self._spin_grid_ymax)
        mf_layout.addRow("Ego Y range:", grid_row2)

        grid_row3 = QHBoxLayout()
        self._spin_grid_zmin = QDoubleSpinBox()
        self._spin_grid_zmin.setRange(-500, 500)
        self._spin_grid_zmin.setValue(DEFAULT_GRID_Z_MIN)
        self._spin_grid_zmin.setToolTip("Ego Z min (up, m) e.g. -1.0")
        self._spin_grid_zmax = QDoubleSpinBox()
        self._spin_grid_zmax.setRange(-500, 500)
        self._spin_grid_zmax.setValue(DEFAULT_GRID_Z_MAX)
        self._spin_grid_zmax.setToolTip("Ego Z max (up, m) e.g. 5.4")
        grid_row3.addWidget(QLabel("Z:"))
        grid_row3.addWidget(self._spin_grid_zmin)
        grid_row3.addWidget(self._spin_grid_zmax)
        mf_layout.addRow("Ego Z range:", grid_row3)

        # Show ego markers checkbox
        self._chk_show_markers = QCheckBox("Show ego markers")
        self._chk_show_markers.setChecked(True)
        self._chk_show_markers.setStyleSheet("color: #bbb;")
        mf_layout.addRow("", self._chk_show_markers)

        # BBox JSON
        bbox_row = QHBoxLayout()
        self._txt_bbox_json = QLineEdit(DEFAULT_BBOXES_JSON)
        self._txt_bbox_json.setStyleSheet(
            "QLineEdit { background: #2a2a3e; color: #ddd; }"
        )
        btn_browse_bbox = QPushButton("...")
        btn_browse_bbox.setFixedWidth(30)
        btn_browse_bbox.setStyleSheet(self._btn_style("#555", "10px"))
        btn_browse_bbox.clicked.connect(self._on_browse_bbox_json)
        bbox_row.addWidget(self._txt_bbox_json)
        bbox_row.addWidget(btn_browse_bbox)
        mf_layout.addRow("BBox JSON:", bbox_row)

        # Show bboxes checkbox
        self._chk_show_bboxes = QCheckBox("Show 3D BBoxes + IDs")
        self._chk_show_bboxes.setChecked(False)
        self._chk_show_bboxes.setStyleSheet("color: #bbb;")
        self._chk_show_bboxes.toggled.connect(self._on_bbox_toggled)
        mf_layout.addRow("", self._chk_show_bboxes)

        # Load Multi-Frame button
        btn_load_mf = QPushButton("Load Multi-Frame")
        btn_load_mf.setStyleSheet(self._btn_style("#ee8844"))
        btn_load_mf.clicked.connect(self._on_load_multiframe)
        mf_layout.addRow(btn_load_mf)

        self._lbl_mf_status = QLabel("")
        self._lbl_mf_status.setStyleSheet("color: #888; font-size: 10px;")
        self._lbl_mf_status.setWordWrap(True)
        mf_layout.addRow(self._lbl_mf_status)

        side_layout.addWidget(mf_grp)

        # -- Shape section --
        shape_grp = QGroupBox("Shape (H × W × Z)")
        shape_grp.setStyleSheet(self._group_style())
        shape_layout = QFormLayout(shape_grp)

        self._spin_H = QSpinBox()
        self._spin_H.setRange(1, 512)
        self._spin_H.setValue(DEFAULT_H)
        self._spin_H.setToolTip("Height bins (vertical)")
        shape_layout.addRow("H:", self._spin_H)

        self._spin_W = QSpinBox()
        self._spin_W.setRange(1, 1024)
        self._spin_W.setValue(DEFAULT_W)
        self._spin_W.setToolTip("Width (lateral)")
        shape_layout.addRow("W:", self._spin_W)

        self._spin_Z = QSpinBox()
        self._spin_Z.setRange(1, 1024)
        self._spin_Z.setValue(DEFAULT_Z)
        self._spin_Z.setToolTip("Depth (longitudinal)")
        shape_layout.addRow("Z:", self._spin_Z)

        side_layout.addWidget(shape_grp)

        # -- Cube size --
        cube_grp = QGroupBox("Cube Size")
        cube_grp.setStyleSheet(self._group_style())
        cube_layout = QHBoxLayout(cube_grp)
        self._spin_cube = QDoubleSpinBox()
        self._spin_cube.setRange(0.3, 5.0)
        self._spin_cube.setValue(1.0)
        self._spin_cube.setSingleStep(0.1)
        self._spin_cube.setToolTip("Scale factor applied to each unit cube")
        self._spin_cube.valueChanged.connect(self._on_cube_scale)
        cube_layout.addWidget(QLabel("Scale:"))
        cube_layout.addWidget(self._spin_cube)
        side_layout.addWidget(cube_grp)

        # -- Stride (downsample) --
        stride_grp = QGroupBox("Stride (Downsample)")
        stride_grp.setStyleSheet(self._group_style())
        stride_layout = QHBoxLayout(stride_grp)
        self._spin_stride = QSpinBox()
        self._spin_stride.setRange(1, 50)
        self._spin_stride.setValue(1)
        self._spin_stride.setToolTip("1 = full res; larger = sparser, faster")
        self._spin_stride.valueChanged.connect(self._on_stride_changed)
        stride_layout.addWidget(QLabel("Step:"))
        stride_layout.addWidget(self._spin_stride)
        side_layout.addWidget(stride_grp)

        # -- Render button --
        btn_render = QPushButton("Render")
        btn_render.setStyleSheet(self._btn_style("#5599ff"))
        btn_render.clicked.connect(self._on_render)
        side_layout.addWidget(btn_render)

        # -- Save Screenshot button --
        btn_save = QPushButton("Save Screenshot")
        btn_save.setStyleSheet(self._btn_style("#cc8844"))
        btn_save.clicked.connect(self._on_save_screenshot)
        side_layout.addWidget(btn_save)

        # -- View presets --
        view_grp = QGroupBox("View Presets")
        view_grp.setStyleSheet(self._group_style())
        view_grid = QGridLayout(view_grp)
        view_grid.setSpacing(3)

        preset_views = [
            ("Front (+X)",   90,   0),   # looking down +Y axis at X-Z face (H×Z)
            ("Side (+Y)",     0,   0),   # looking down +X axis at Y-Z face (W×Z)
            ("Top (-Z)",      0,  90),   # looking down from above
            ("Front 45\u00b0",  90,  45),   # front, slanted down
            ("Back 45\u00b0",   270, 45),   # back, slanted down
            ("Side 45\u00b0",    0,  45),   # side, slanted down
        ]
        for i, (label, az, el) in enumerate(preset_views):
            btn = QPushButton(label)
            btn.setStyleSheet(self._btn_style("#666", font_size="10px"))
            btn.clicked.connect(lambda checked, a=az, e=el: self._set_view(a, e))
            view_grid.addWidget(btn, i // 3, i % 3)

        side_layout.addWidget(view_grp)

        # -- Legend --
        legend_grp = QGroupBox("Classes")
        legend_grp.setStyleSheet(self._group_style())
        legend_layout = QVBoxLayout(legend_grp)
        self._legend = LegendWidget()
        self._legend.toggled.connect(self._on_class_toggled)
        legend_layout.addWidget(self._legend)

        # Select all / deselect all
        sel_layout = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("None")
        btn_none.clicked.connect(self._select_none)
        sel_layout.addWidget(btn_all)
        sel_layout.addWidget(btn_none)
        legend_layout.addLayout(sel_layout)

        side_layout.addWidget(legend_grp)

        # -- Stats --
        stats_grp = QGroupBox("Stats")
        stats_grp.setStyleSheet(self._group_style())
        stats_layout = QVBoxLayout(stats_grp)
        self._lbl_stats = QLabel("—")
        self._lbl_stats.setStyleSheet("color: #aaa; font-size: 11px;")
        stats_layout.addWidget(self._lbl_stats)
        side_layout.addWidget(stats_grp)

        side_layout.addStretch()

        # ---------- assemble ----------
        # 画布在左侧（可伸缩），工具栏/控制面板固定在最右侧
        splitter = QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._canvas.native)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(1, False)   # 工具栏不允许被折叠消失
        splitter.setSizes([1100, 280])      # 默认宽度：画布占大头，工具栏靠右
        main_layout.addWidget(splitter)

        # ---------- status bar ----------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a .bin file to begin")

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _group_style() -> str:
        return """
            QGroupBox {
                color: #ccc;
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """

    @staticmethod
    def _btn_style(color: str, font_size: str = "12px") -> str:
        return f"""
            QPushButton {{
                background: {color};
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 4px 6px;
                font-weight: bold;
                font-size: {font_size};
            }}
            QPushButton:hover {{
                background: {color};
                opacity: 0.85;
            }}
            QPushButton:pressed {{
                background: {color};
            }}
        """

    # ---- slots ----------------------------------------------------------
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open OCC Bin File", "",
            "Bin Files (*.bin);;All Files (*)"
        )
        if not path:
            return
        self._load_file(path)

    def _on_open_npz(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open OCC GT NPZ File", "",
            "NPZ Files (*.npz);;All Files (*)"
        )
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path: str):
        """Load a single .bin or .npz file and (for npz) apply GT masks."""
        self._filepath = path
        self._lbl_file.setText(os.path.basename(path))
        self._volume = None
        self._npz_masks = {}
        self._multiframe_mode = False
        self._multiframe_data = []

        if path.lower().endswith(".npz"):
            try:
                semantics, masks = load_npz(path)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))
                self._cmb_mask.setEnabled(False)
                return
            self._volume = semantics
            self._npz_masks = masks
            H, W, Z = semantics.shape
            self._spin_H.setValue(H)
            self._spin_W.setValue(W)
            self._spin_Z.setValue(Z)
            self._cmb_mask.blockSignals(True)
            self._cmb_mask.setCurrentText("None")
            self._cmb_mask.blockSignals(False)
            self._cmb_mask.setEnabled(True)
            self._mask = None
            self._refresh_view()
            self._update_stats()
            self._status.showMessage(
                f"Loaded NPZ: {os.path.basename(path)} "
                f"({H}×{W}×{Z}), masks={list(masks.keys())}"
            )
        else:
            self._cmb_mask.blockSignals(True)
            self._cmb_mask.setCurrentText("None")
            self._cmb_mask.blockSignals(False)
            self._cmb_mask.setEnabled(False)
            self._mask = None
            self._status.showMessage(
                f"Loaded file: {path} — set shape & press Render"
            )

    def _compute_mask(self):
        """Return the currently selected GT mask array (or None)."""
        sel = self._cmb_mask.currentText()
        l = self._npz_masks.get("mask_lidar")
        c = self._npz_masks.get("mask_camera")
        if sel == "mask_lidar":
            return l
        if sel == "mask_camera":
            return c
        if sel == "Both (union)":
            if l is not None and c is not None:
                return l | c
            return l if l is not None else c
        if sel == "Both (intersection)":
            if l is not None and c is not None:
                return l & c
            return None
        return None

    def _on_mask_changed(self, text: str):
        self._mask = self._compute_mask()
        if self._volume is not None and not self._multiframe_mode:
            self._refresh_view()
            self._update_stats()

    def _on_render(self):
        if not self._filepath:
            QMessageBox.warning(self, "No File", "Please open a .bin file first.")
            return

        self._multiframe_mode = False
        H = self._spin_H.value()
        W = self._spin_W.value()
        Z = self._spin_Z.value()

        try:
            self._volume = load_bin(self._filepath, H, W, Z)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._refresh_view()
        self._update_stats()

    def _on_save_screenshot(self):
        """Save the current 3D view as a PNG image."""
        if self._volume is None and not (self._multiframe_mode and self._multiframe_data):
            QMessageBox.warning(self, "Nothing to save", "Please render data first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if not path:
            return

        try:
            # Use Qt grab() to capture the on-screen canvas pixels.
            # vispy's render() goes through an FBO that may lack a depth
            # buffer, losing correct occlusion.  grab() reads the actual
            # on-screen framebuffer which has proper depth.
            native = self._canvas.native
            pixmap = native.grab()
            qimg = pixmap.toImage()

            # Convert QImage → numpy array (RGBA, uint8)
            qimg = qimg.convertToFormat(QtGui.QImage.Format_RGBA8888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.constBits()
            ptr.setsize(h * w * 4)
            img = np.ndarray((h, w, 4), dtype=np.uint8, buffer=ptr)
            img = img.copy()  # own the data before QImage is freed
        except Exception as e:
            QMessageBox.critical(self, "Render error",
                                 f"Failed to capture view:\n{e}")
            return

        if imageio is not None:
            try:
                imageio.imwrite(path, img)
                self._status.showMessage(f"Screenshot saved: {path}")
                return
            except Exception as e:
                QMessageBox.critical(self, "Save error",
                                     f"Failed to save image:\n{e}")
                return
        else:
            QMessageBox.warning(
                self, "Missing dependency",
                "imageio is required to save screenshots.\n"
                "Install with: pip install imageio"
            )
            return

    def _refresh_view(self):
        if self._multiframe_mode and self._multiframe_data:
            self._refresh_multiframe_view()
            return
        if self._volume is None:
            self._canvas.clear_data()
            return
        hidden = self._legend.get_hidden_classes()
        stride = self._spin_stride.value()
        verts, faces, cols = build_voxel_mesh(
            self._volume, hidden, stride, self._mask
        )
        # Apply cube scale
        scale = self._spin_cube.value()
        if scale != 1.0 and verts.size > 0:
            verts = verts * scale
        # Wireframe edges — use mesh centres for clean outlines
        N = faces.shape[0]  # number of triangles
        n_verts = (N // 12) if N > 0 else 0
        if n_verts > 0:
            # recover original centres from vertices (every 36 verts = 1 cube)
            ecenters = verts.reshape(-1, 36, 3)[:, 0, :]
        else:
            ecenters = np.zeros((0, 3), dtype=np.float32)
        epts, ecols = build_cube_edges(ecenters, scale)
        self._canvas.set_data(verts, faces, cols, epts, ecols,
                              self._volume.shape[0],
                              self._volume.shape[1],
                              self._volume.shape[2])

    def _refresh_multiframe_view(self):
        """Render all frames positioned by their ego poses in global frame."""
        hidden = self._legend.get_hidden_classes()
        stride = self._spin_stride.value()
        scale = self._spin_cube.value()
        voxel_size = self._spin_voxel_size.value()
        grid_x_range = (self._spin_grid_xmin.value(), self._spin_grid_xmax.value())
        grid_y_range = (self._spin_grid_ymin.value(), self._spin_grid_ymax.value())
        grid_z_range = (self._spin_grid_zmin.value(), self._spin_grid_zmax.value())
        show_markers = self._chk_show_markers.isChecked()
        show_bboxes = self._chk_show_bboxes.isChecked()

        all_verts = []
        all_faces = []
        all_colors = []
        all_epts = []
        all_ecols = []
        all_marker_verts = []
        all_marker_colors = []
        all_bbox_verts = []
        all_bbox_colors = []
        all_label_pos = []
        all_label_text = []
        all_label_col = []
        face_offset = 0
        n_loaded = 0

        for fdata in self._multiframe_data:
            bin_path = os.path.join(
                self._cmb_data_dir.currentText(), fdata["bin_name"]
            )
            if not os.path.isfile(bin_path):
                continue

            H = self._spin_H.value()
            W = self._spin_W.value()
            Z = self._spin_Z.value()
            try:
                vol = load_bin(bin_path, H, W, Z)
            except Exception:
                continue

            pose = fdata["pose"]
            tx, ty, tz = pose["tx"], pose["ty"], pose["tz"]
            rot = quat_to_rot_mat(pose["qw"], pose["qx"], pose["qy"], pose["qz"])

            # Build mesh in voxel-index space
            verts, faces, cols = build_voxel_mesh(vol, hidden, stride)
            if verts.size == 0:
                continue

            # Recover voxel centres (every 36 verts = 1 cube)
            ecenters_idx = verts.reshape(-1, 36, 3)[:, 0, :]  # (N, 3) in (h,w,z)

            # Convert centres from voxel indices → ego metric
            ecenters_ego = voxel_indices_to_ego(
                ecenters_idx, voxel_size,
                grid_x_range, grid_y_range, grid_z_range,
            )
            # Transform centres: ego → global
            ecenters_global = transform_vertices_ego_to_global(
                ecenters_ego, (tx, ty, tz), rot,
            )

            # Rebuild vertices: centres_global + cube offsets (scaled)
            _cube_verts = np.array([
                [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5],
                [-0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
                [-0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
                [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
                [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
                [-0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
                [ 0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
                [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5],
                [-0.5, -0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5, -0.5,  0.5],
                [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
                [ 0.5, -0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
            ], dtype=np.float32) * voxel_size * scale

            global_verts = (ecenters_global[:, np.newaxis, :]
                            + _cube_verts[np.newaxis, :, :])
            global_verts = global_verts.reshape(-1, 3).astype(np.float32)

            faces_offset = faces + face_offset

            all_verts.append(global_verts)
            all_faces.append(faces_offset)
            all_colors.append(cols)

            # Edges
            epts, ecols = build_cube_edges(ecenters_global, voxel_size * scale)
            all_epts.append(epts)
            all_ecols.append(ecols)

            # Ego marker
            if show_markers:
                mv, mc = build_ego_marker((tx, ty, tz), marker_size=2.0)
                all_marker_verts.append(mv)
                all_marker_colors.append(mc)

            # Bboxes for this frame
            if show_bboxes and self._bboxes_data:
                sample_token = fdata["sample_token"]
                anns = self._bboxes_data.get(sample_token, [])
                for ann in anns:
                    cat = ann.get("category_name", "others")
                    cls_id = CATEGORY_TO_CLASS.get(cat, 0)
                    col_rgba = COLORS_RGB[cls_id] + (1.0,)
                    trans = tuple(ann["translation"])
                    bsize = tuple(ann["size"])
                    brot = (
                        ann["rotation"][0], ann["rotation"][1],
                        ann["rotation"][2], ann["rotation"][3],
                    )
                    bv, bc = build_bbox_wireframe(trans, bsize, brot, col_rgba)
                    all_bbox_verts.append(bv)
                    all_bbox_colors.append(bc)

                    # ID label — place above bbox top
                    inst_id = ann.get("instance_id")
                    if inst_id is None:
                        inst_id = ann.get("instance_token", "")
                    # Short ID: first 4 hex chars
                    short_id = str(inst_id)[:4] if inst_id else ""
                    if short_id:
                        cx, cy, cz = trans
                        _, _, h = bsize
                        label_z = cz + h * 0.5 + 0.2  # above top + 2m offset
                        all_label_pos.append(np.array([cx, cy, label_z], dtype=np.float32))
                        all_label_text.append(short_id)
                        all_label_col.append(col_rgba)

            face_offset += len(global_verts)
            n_loaded += 1

        if n_loaded == 0:
            self._canvas.clear_data()
            self._status.showMessage("No frames loaded — check data dir & file names")
            return

        merged_verts = np.concatenate(all_verts) if all_verts else np.zeros((0, 3), dtype=np.float32)
        merged_faces = np.concatenate(all_faces) if all_faces else np.zeros((0, 3), dtype=np.uint32)
        merged_colors = np.concatenate(all_colors) if all_colors else np.zeros((0, 4), dtype=np.float32)
        merged_epts = np.concatenate(all_epts) if all_epts else np.zeros((0, 3), dtype=np.float32)
        merged_ecols = np.concatenate(all_ecols) if all_ecols else np.zeros((0, 4), dtype=np.float32)

        if all_marker_verts:
            merged_mv = np.concatenate(all_marker_verts)
            merged_mc = np.concatenate(all_marker_colors)
        else:
            merged_mv = None
            merged_mc = None

        if all_bbox_verts:
            merged_bv = np.concatenate(all_bbox_verts)
            merged_bc = np.concatenate(all_bbox_colors)
        else:
            merged_bv = None
            merged_bc = None

        self._canvas.set_data(
            merged_verts, merged_faces, merged_colors,
            merged_epts, merged_ecols,
            self._spin_H.value(), self._spin_W.value(), self._spin_Z.value(),
            ego_markers_verts=merged_mv,
            ego_markers_colors=merged_mc,
        )
        self._canvas.set_bbox_data(
            merged_bv, merged_bc,
            label_positions=all_label_pos if all_label_pos else None,
            label_texts=all_label_text if all_label_text else None,
            label_colors=all_label_col if all_label_col else None,
        )

        bbox_info = ""
        if show_bboxes:
            n_labels = len(all_label_pos)
            bbox_info = f", {sum(1 for _ in all_bbox_verts)} bboxes"
            if n_labels:
                bbox_info += f", {n_labels} labels"
        self._status.showMessage(
            f"Multi-frame: {n_loaded} frames loaded, "
            f"{merged_verts.shape[0]} vertices{bbox_info}"
        )

    def _on_load_multiframe(self):
        """Load frames.lst, match ego poses, find bin files, and render."""
        frames_lst_path = self._txt_frames_lst.text()
        pose_csv_path = self._txt_pose_csv.text()
        data_dir = self._cmb_data_dir.currentText()

        if not os.path.isfile(frames_lst_path):
            QMessageBox.critical(self, "Error",
                                 f"frames.lst not found:\n{frames_lst_path}")
            return
        if not os.path.isfile(pose_csv_path):
            QMessageBox.critical(self, "Error",
                                 f"Ego pose CSV not found:\n{pose_csv_path}")
            return
        if not os.path.isdir(data_dir):
            QMessageBox.critical(self, "Error",
                                 f"Data directory not found:\n{data_dir}")
            return

        try:
            all_frames = load_frames_list(frames_lst_path)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to parse frames.lst:\n{e}")
            return

        try:
            self._ego_poses = load_ego_poses_csv(pose_csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to parse ego pose CSV:\n{e}")
            return

        prefix_index = _build_token_prefix_index(self._ego_poses)

        start = self._spin_start_frame.value()
        n_frames = self._spin_num_frames.value()
        selected = all_frames[start:start + n_frames]

        matched = match_frames_to_poses(selected, self._ego_poses, prefix_index)

        if not matched:
            QMessageBox.warning(self, "No Matches",
                                "No frames could be matched to ego poses.")
            return

        # Check which bin files exist
        existing = []
        missing = []
        for fdata in matched:
            bin_path = os.path.join(data_dir, fdata["bin_name"])
            if os.path.isfile(bin_path):
                existing.append(fdata)
            else:
                missing.append(fdata["bin_name"])

        self._multiframe_data = existing
        self._multiframe_mode = True
        self._volume = None

        # Auto-load bbox JSON if checkbox is checked
        if self._chk_show_bboxes.isChecked() and not self._bboxes_data:
            self._load_bboxes_if_needed()

        msg = f"Matched {len(matched)} frames, {len(existing)} bin files found"
        if missing:
            msg += f", {len(missing)} missing"
            if len(missing) <= 5:
                msg += ": " + ", ".join(missing)
        self._lbl_mf_status.setText(msg)
        self._status.showMessage(msg)

        self._refresh_view()
        self._update_stats()

    def _populate_data_dirs(self):
        """Scan data/ for subdirectories that contain .bin files."""
        base = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(base, "data")
        dirs = []
        if os.path.isdir(data_root):
            for entry in sorted(os.listdir(data_root)):
                full = os.path.join(data_root, entry)
                if os.path.isdir(full):
                    try:
                        has_bin = any(
                            f.endswith(".bin") for f in os.listdir(full)
                        )
                    except Exception:
                        has_bin = False
                    if has_bin:
                        dirs.append(full)
        self._cmb_data_dir.addItems(dirs)
        if dirs:
            self._cmb_data_dir.setCurrentIndex(0)

    def _on_browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if d:
            idx = self._cmb_data_dir.findText(d)
            if idx < 0:
                self._cmb_data_dir.addItem(d)
                idx = self._cmb_data_dir.count() - 1
            self._cmb_data_dir.setCurrentIndex(idx)

    def _on_browse_frames_lst(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select frames.lst", "", "List Files (*.lst);;All Files (*)"
        )
        if path:
            self._txt_frames_lst.setText(path)

    def _on_browse_pose_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ego pose CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._txt_pose_csv.setText(path)

    def _on_browse_bbox_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select bbox JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._txt_bbox_json.setText(path)

    def _on_bbox_toggled(self, checked: bool):
        """Load bbox JSON on first toggle, then refresh."""
        if checked and not self._bboxes_data:
            if not self._load_bboxes_if_needed():
                self._chk_show_bboxes.blockSignals(True)
                self._chk_show_bboxes.setChecked(False)
                self._chk_show_bboxes.blockSignals(False)
                return
        if self._multiframe_mode and self._multiframe_data:
            self._refresh_view()
            self._update_stats()

    def _load_bboxes_if_needed(self) -> bool:
        """Load bbox JSON. Returns True on success."""
        json_path = self._txt_bbox_json.text()
        if not os.path.isfile(json_path):
            QMessageBox.critical(self, "Error",
                                 f"BBox JSON not found:\n{json_path}")
            return False
        try:
            self._bboxes_data = load_bboxes_json(json_path)
            self._status.showMessage(
                f"Loaded bboxes: {len(self._bboxes_data)} samples"
            )
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load bbox JSON:\n{e}")
            return False

    def _on_class_toggled(self, class_id: int, visible: bool):
        if self._volume is not None or (self._multiframe_mode and self._multiframe_data):
            self._refresh_view()
            self._update_stats()

    def _set_view(self, azimuth: float, elevation: float):
        """Snap the camera to a preset (azimuth, elevation)."""
        cam = self._canvas.view.camera
        cam.azimuth = azimuth
        cam.elevation = elevation

    def _on_cube_scale(self, val: float):
        if self._volume is not None or (self._multiframe_mode and self._multiframe_data):
            self._refresh_view()

    def _on_stride_changed(self, val: int):
        if self._volume is not None or (self._multiframe_mode and self._multiframe_data):
            self._refresh_view()
            self._update_stats()

    def _select_all(self):
        for cid in range(NUM_CLASSES):
            self._legend._checkboxes[cid].setChecked(True)

    def _select_none(self):
        for cid in range(NUM_CLASSES):
            self._legend._checkboxes[cid].setChecked(False)

    def _update_stats(self):
        if self._multiframe_mode and self._multiframe_data:
            n_frames = len(self._multiframe_data)
            self._lbl_stats.setText(f"Multi-frame mode\n{len(self._multiframe_data)} frames loaded")
            return
        if self._volume is None:
            self._lbl_stats.setText("—")
            return
        hidden = self._legend.get_hidden_classes()
        stride = self._spin_stride.value()
        if stride > 1:
            vol = self._volume[::stride, ::stride, ::stride]
        else:
            vol = self._volume
        visible_mask = np.isin(vol, list(hidden), invert=True)
        if self._mask is not None:
            m = self._mask
            if stride > 1:
                m = m[::stride, ::stride, ::stride]
            visible_mask = visible_mask & m
        mask = visible_mask.sum()
        total = vol.size
        pc = mask / total * 100 if total > 0 else 0
        extra = (f"\nstride={stride}" if stride > 1 else "")
        if self._mask is not None:
            extra += f"\nmask={self._cmb_mask.currentText()}"
        self._lbl_stats.setText(
            f"Visible cubes: {mask:,} / {total:,}\n"
            f"({pc:.1f}% of grid){extra}"
        )


# ===================================================================
def main():
    # Ensure vispy uses PyQt5 backend
    vispy.app.use_app("pyqt5")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 50))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(40, 40, 60))
    dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(50, 50, 70))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(70, 130, 200))
    app.setPalette(dark_palette)

    # Global stylesheet
    app.setStyleSheet("""
        QSpinBox, QDoubleSpinBox {
            background: #2a2a3e;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 4px;
        }
        QLabel {
            color: #bbb;
        }
        QScrollArea {
            border: none;
            background: transparent;
        }
    """)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
