import os
import sys
import pdb
import time
import yaml
import torch
import chamfer
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation

import open3d
import open3d as o3d
from copy import deepcopy


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)

def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances




def lidar_to_world_to_lidar(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc


def lidar_to_world_to_ego(pc, lidar_calibrated_sensor, lidar_ego_pose, target_ego_pose):
    """Transform lidar-local points into the target ego frame (via world).

    NuScenes convention: p_ego = R @ p_sensor + t.
    """
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))
    pc.translate(-np.array(target_ego_pose['translation']))
    pc.rotate(Quaternion(target_ego_pose['rotation']).rotation_matrix.T)
    return pc


def ego_to_world_to_ego(pc, src_ego_pose, dst_ego_pose):
    """Transform ego-frame points (N, D) into another ego frame (via world).

    Only the first three (xyz) columns are transformed; any extra columns
    (e.g. semantic labels) are preserved.  Returns a plain numpy array.
    """
    pc = np.asarray(pc, dtype=np.float64).copy()
    R_src = Quaternion(src_ego_pose['rotation']).rotation_matrix   # src ego -> world
    t_src = np.array(src_ego_pose['translation'], dtype=np.float64)
    R_dst = Quaternion(dst_ego_pose['rotation']).rotation_matrix   # dst ego -> world
    t_dst = np.array(dst_ego_pose['translation'], dtype=np.float64)

    xyz = pc[:, :3] @ R_src.T + t_src             # -> world
    xyz = (R_dst.T @ (xyz - t_dst).T).T           # -> dst ego
    pc[:, :3] = xyz
    return pc


def bbox_lidar_to_ego(gt_bbox_3d, lidar_calibrated_sensor):
    """Convert boxes [x, y, z, w, l, h, yaw] from the LiDAR sensor frame to ego.

    `nusc.get_sample_data` returns boxes in the current sensor (LIDAR_TOP)
    frame; our occupancy grid lives in the ego frame, so apply the
    sensor->ego calibration transform (p_ego = R @ p_sensor + t) to the center
    and add the sensor->ego yaw offset to the heading.
    """
    R_l2e = Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix
    t_l2e = np.array(lidar_calibrated_sensor['translation'], dtype=np.float64)
    c = gt_bbox_3d[:, :3].astype(np.float64)
    c_ego = (R_l2e @ c.T).T + t_l2e                                    # p_ego = R @ p + t
    out = gt_bbox_3d.copy()
    out[:, :3] = c_ego
    yaw_l2e = Quaternion(lidar_calibrated_sensor['rotation']).yaw_pitch_roll[0]
    out[:, 6] = out[:, 6] + yaw_l2e
    return out


def bbox_ego_to_ego(gt_bbox_3d, src_ego_pose, dst_ego_pose):
    """Transform boxes from one ego frame to another (for cross-frame removal).

    The center is transformed via world; the heading is offset by the yaw
    difference between the two ego frames (both are z-axis rotations).
    """
    out = gt_bbox_3d.copy()
    out[:, :3] = ego_to_world_to_ego(gt_bbox_3d[:, :3], src_ego_pose, dst_ego_pose)
    h_src = Quaternion(src_ego_pose['rotation']).yaw_pitch_roll[0]
    h_dst = Quaternion(dst_ego_pose['rotation']).yaw_pitch_roll[0]
    out[:, 6] = out[:, 6] + (h_src - h_dst)
    return out


def bev_surface_normalize(points_sem, class_ids, bin_size, z_tol,
                          quantile=0.1, smooth_kernel=3, snap=False):
    """Slope-aware BEV surface normalization for ground-like classes.

    The old per-column *median* filter anchored each column to its middle z.
    On a slope that middle is far from the actual ground surface, so it kept a
    thick band of points (bumps) instead of a clean layer.  This version
    estimates a continuous, slope-following surface and keeps only points
    within `z_tol` of it:

      1. Bin points into BEV cells of edge length `bin_size`.
      2. Estimate each cell's surface height as the `quantile`-th percentile of
         z (a low percentile anchors to the floor / actual surface).
      3. Median-smooth the cell-height grid to suppress cell outliers.
      4. Bilinearly interpolate the grid -> continuous surface z(x, y).
      5. Keep points with |z - surface(x, y)| <= z_tol (optionally snap z).

    points_sem    : (N, 4) [x, y, z, label]
    class_ids     : semantic ids to flatten (learning_map ids)
    bin_size      : BEV bin edge length (meters)
    z_tol         : max vertical distance to the surface (meters)
    quantile      : per-cell percentile used as surface anchor (0..1, 0 = min)
    smooth_kernel : median-filter size on the height grid (odd; 1 = disabled)
    snap          : if True, set kept points' z to the interpolated surface

    Returns a filtered copy (M, 4).
    """
    import warnings

    points_sem = np.asarray(points_sem)
    if points_sem.shape[0] == 0:
        return points_sem

    labels = points_sem[:, 3].astype(np.int64)
    keep = np.ones(points_sem.shape[0], dtype=bool)
    BIG = 1_000_000

    for cls in class_ids:
        cls = int(cls)
        sel = labels == cls
        if not sel.any():
            continue
        idx = np.where(sel)[0]
        x = points_sem[idx, 0].astype(np.float64)
        y = points_sem[idx, 1].astype(np.float64)
        z = points_sem[idx, 2].astype(np.float64)

        cx = np.floor(x / bin_size).astype(np.int64)
        cy = np.floor(y / bin_size).astype(np.int64)
        cell_key = cx * BIG + cy

        # sort by (cell, z) and compute the per-cell quantile height
        order = np.lexsort((z, cell_key))
        cell_sorted = cell_key[order]
        z_sorted = z[order]
        uniq_key, starts = np.unique(cell_sorted, return_index=True)
        ends = np.concatenate([starts[1:], [cell_sorted.shape[0]]])
        counts = ends - starts
        qoff = np.floor((counts - 1) * quantile).astype(np.int64)
        cell_z = z_sorted[starts + qoff]

        # the (cx, cy) of each unique cell (sampled from its first point)
        uniq_cx = cx[order[starts]]
        uniq_cy = cy[order[starts]]
        # map every point back to its unique cell index
        inverse = np.searchsorted(uniq_key, cell_key)
        cell_z_pt = cell_z[inverse]

        # build the height grid
        cx_min = uniq_cx.min()
        cy_min = uniq_cy.min()
        H = np.full((int(uniq_cx.max()) - int(cx_min) + 1,
                     int(uniq_cy.max()) - int(cy_min) + 1),
                    np.nan, dtype=np.float64)
        H[uniq_cx - cx_min, uniq_cy - cy_min] = cell_z

        # median-smooth the grid (empty cells stay NaN and are ignored)
        if smooth_kernel > 1:
            k = int(smooth_kernel)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            Hp = np.pad(H, pad, mode='constant', constant_values=np.nan)
            windows = np.stack(
                [Hp[dy:dy + H.shape[0], dx:dx + H.shape[1]]
                 for dy in range(k) for dx in range(k)], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                H = np.nanmedian(windows, axis=0)

        # bilinearly interpolate the surface at each point's exact (x, y)
        nh, nw = H.shape
        gx = cx - cx_min
        gy = cy - cy_min
        fx = x / bin_size - cx       # fractional position within the cell [0, 1)
        fy = y / bin_size - cy
        gx0 = np.clip(gx, 0, nh - 1)
        gy0 = np.clip(gy, 0, nw - 1)
        gx1 = np.clip(gx + 1, 0, nh - 1)
        gy1 = np.clip(gy + 1, 0, nw - 1)

        vals = np.stack([H[gx0, gy0], H[gx0, gy1],
                         H[gx1, gy0], H[gx1, gy1]], axis=1)      # (N, 4)
        wx1 = fx
        wy1 = fy
        w = np.stack([(1.0 - wx1) * (1.0 - wy1), (1.0 - wx1) * wy1,
                      wx1 * (1.0 - wy1), wx1 * wy1], axis=1)     # (N, 4)
        valid = ~np.isnan(vals)
        vals = np.where(valid, vals, 0.0)
        w = np.where(valid, w, 0.0)
        wsum = w.sum(axis=1)
        surface = np.where(wsum > 0,
                           (vals * w).sum(axis=1) / np.maximum(wsum, 1e-12),
                           cell_z_pt)

        local_keep = np.abs(z - surface) <= z_tol
        keep[idx[~local_keep]] = False
        if snap:
            points_sem[idx[local_keep], 2] = surface[local_keep]

    return points_sem[keep]


def remove_isolated_voxels(semantics, free_val, neighborhood=6):
    """Remove isolated occupied voxels (single-cell speckle noise).

    A voxel that is occupied but has NO occupied neighbor in its
    `neighborhood` (6 or 26 connectivity) is treated as noise and reset to
    `free_val`.  Vectorized via slice-shift accumulation, so it is O(V).
    """
    occ = semantics != free_val
    if not occ.any():
        return semantics
    n0, n1, n2 = occ.shape
    padded = np.zeros((n0 + 2, n1 + 2, n2 + 2), dtype=np.uint8)
    padded[1:-1, 1:-1, 1:-1] = occ

    if neighborhood == 6:
        offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                   (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    elif neighborhood == 5:
        # up (+Z) + the 4 horizontal neighbors (+-X, +-Y), excluding down (-Z)
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    else:
        offsets = [(i, j, k)
                   for i in (-1, 0, 1)
                   for j in (-1, 0, 1)
                   for k in (-1, 0, 1)
                   if not (i == j == k == 0)]

    neigh = np.zeros_like(occ, dtype=np.int16)
    for dz, dy, dx in offsets:
        neigh += padded[1 + dz:1 + dz + n0,
                       1 + dy:1 + dy + n1,
                       1 + dx:1 + dx + n2]

    isolated = occ & (neigh == 0)
    out = semantics.copy()
    out[isolated] = free_val
    return out


def _fill_floor_holes(floor, valid):
    """Fill unknown floor cells (valid == False) from neighbouring cells.

    Every BEV column with no ground voxel (a "hole" in the ground layer, e.g.
    where a parked car, a pole or a pedestrian occupies the whole column) still
    needs a floor height, otherwise the part of that object that dips below the
    road surface would survive the below-ground removal.  Interior cells are
    filled iteratively from their 8 raster neighbours using the median height,
    which converges for any connected hole (at most max(H, W) iterations).
    """
    import warnings

    H, W = floor.shape
    filled = floor.astype(np.float64).copy()
    filled[~valid] = np.nan

    for _ in range(max(H, W)):
        nan_mask = np.isnan(filled)
        if not nan_mask.any():
            break
        padded = np.pad(filled, 1, mode='constant', constant_values=np.nan)
        neighbours = [
            padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
        ]
        stack = np.stack(neighbours, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
        updatable = nan_mask & ~np.isnan(med)
        if not updatable.any():
            break
        filled[updatable] = med[updatable]

    return np.where(np.isnan(filled), -1, np.rint(filled).astype(np.int64))


def remove_below_ground(semantics, free_val, ground_class_ids, tolerance=0,
                        fill_holes=True):
    """Remove occupied voxels that lie below the ground floor.

    For every BEV column (x, y) the bottom-most ground-class voxel marks the
    ground floor; matter cannot physically exist under the ground, so every
    occupied voxel strictly below that floor is reset to `free_val`.

    A key failure mode is a column with no ground voxel at all: a parked car,
    a pole or a pedestrian occupies the whole column and "pierces" through the
    ground layer, leaving no ground label there (a hole).  The parts of that
    object below the road surface would then survive.  With `fill_holes=True`
    such columns inherit the floor height from their neighbours, so the
    under-ground part of the piercing object is removed as well.

    semantics        : (X, Y, Z) uint8 volume; Z is vertical, index 0 = bottom.
    ground_class_ids : semantic ids treated as the ground / floor.
    tolerance        : extra z-voxels to preserve below the floor (0 = remove all).
    fill_holes       : interpolate the floor into columns without ground.
    """
    semantics = np.asarray(semantics)
    occ = semantics != free_val
    if not occ.any():
        return semantics

    ground = np.isin(semantics, np.asarray(ground_class_ids, dtype=semantics.dtype))
    has_ground = ground.any(axis=2)                    # (X, Y)
    if not has_ground.any():
        return semantics

    nz = semantics.shape[2]
    # Bottom-most ground z: argmax on a bool volume returns the FIRST True along
    # the (bottom -> top) Z axis, i.e. the lowest ground voxel.
    floor = np.where(has_ground, ground.argmax(axis=2), -1)   # (X, Y)

    if fill_holes and (~has_ground).any():
        floor = _fill_floor_holes(floor, has_ground)

    z = np.arange(nz, dtype=np.int64)                          # (Z,)
    below = z[None, None, :] < (floor[:, :, None] - tolerance)  # (X, Y, Z)
    below &= (floor[:, :, None] >= 0)                           # skip unfillable holes
    remove = below & occ

    if not remove.any():
        return semantics
    out = semantics.copy()
    out[remove] = free_val
    return out


# --- Occ3D ground-truth helpers -------------------------------------------
# Occ3D nuScenes uses a 17-class semantic taxonomy: 0 = "others" (noise /
# rare / unlabeled), 1..16 for the 16 standard classes, and the special value
# 17 = FREE / EMPTY voxels in the dense semantics grid.
OCC3D_FREE = 17


def _grid_indices(occ_size):
    """Return all voxel indices (K, 3) with axis order [X, Y, Z]."""
    x = np.arange(occ_size[0])
    y = np.arange(occ_size[1])
    z = np.arange(occ_size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.int64)


def _ray_voxels(origin, point, occ_size, pc_range, voxel_size):
    """Amanatides & Woo 3D DDA over the voxel grid.

    origin / point: continuous coordinates in the same frame as pc_range.
    Returns an (K, 3) int64 array of voxel indices traversed by the segment.
    """
    gmin = np.array(pc_range[:3], dtype=np.float64)
    o = (np.asarray(origin, dtype=np.float64) - gmin) / voxel_size
    p = (np.asarray(point, dtype=np.float64) - gmin) / voxel_size

    dx = p[0] - o[0]
    dy = p[1] - o[1]
    dz = p[2] - o[2]

    x = int(np.floor(o[0]))
    y = int(np.floor(o[1]))
    z = int(np.floor(o[2]))

    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        if 0 <= x < occ_size[0] and 0 <= y < occ_size[1] and 0 <= z < occ_size[2]:
            return np.array([[x, y, z]], dtype=np.int64)
        return np.zeros((0, 3), dtype=np.int64)

    end_x = int(np.floor(p[0]))
    end_y = int(np.floor(p[1]))
    end_z = int(np.floor(p[2]))

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    t_max_x = ((x + (1 if dx > 0 else 0)) - o[0]) / dx if dx != 0 else np.inf
    t_max_y = ((y + (1 if dy > 0 else 0)) - o[1]) / dy if dy != 0 else np.inf
    t_max_z = ((z + (1 if dz > 0 else 0)) - o[2]) / dz if dz != 0 else np.inf

    t_delta_x = abs(1.0 / dx) if dx != 0 else np.inf
    t_delta_y = abs(1.0 / dy) if dy != 0 else np.inf
    t_delta_z = abs(1.0 / dz) if dz != 0 else np.inf

    visited = []
    max_steps = occ_size[0] + occ_size[1] + occ_size[2] + 3
    for _ in range(max_steps):
        if 0 <= x < occ_size[0] and 0 <= y < occ_size[1] and 0 <= z < occ_size[2]:
            visited.append((x, y, z))
        if x == end_x and y == end_y and z == end_z:
            break
        if t_max_x < t_max_y and t_max_x < t_max_z:
            x += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_z:
            y += step_y
            t_max_y += t_delta_y
        else:
            z += step_z
            t_max_z += t_delta_z

    if visited:
        return np.array(visited, dtype=np.int64)
    return np.zeros((0, 3), dtype=np.int64)


def build_lidar_mask(lidar_points, occ_size, pc_range, voxel_size, origin=None):
    """GPU ray-cast from `origin` through each point.

    Vectorized Amanatides & Woo DDA that marches every ray simultaneously on
    the GPU, replacing the original per-point Python loop.  Marks every voxel
    traversed by a ray (free space up to the hit) as visible.

    origin: (3,) ray origin in the same frame as lidar_points (ego frame).
    """
    if lidar_points is None or lidar_points.shape[0] == 0:
        return np.zeros(occ_size, dtype=np.uint8)

    device = torch.device('cuda')
    sx, sy, sz = int(occ_size[0]), int(occ_size[1]), int(occ_size[2])
    gmin = torch.tensor(pc_range[:3], dtype=torch.float32, device=device)

    pts = torch.from_numpy(np.asarray(lidar_points, dtype=np.float32)).to(device)  # (N,3)
    N = pts.shape[0]

    # Origin and endpoint in grid coordinates.
    if origin is None:
        origin = torch.zeros(3, device=device)
    else:
        origin = torch.tensor(np.asarray(origin, dtype=np.float32), device=device)
    o_grid = (origin - gmin) / voxel_size                                 # (3,)
    p_grid = (pts - gmin) / voxel_size                                    # (N,3)

    cur = torch.floor(o_grid).long().unsqueeze(0).expand(N, 3).contiguous().clone()  # (N,3)
    end = torch.floor(p_grid).long()                                        # (N,3)

    direction = p_grid - o_grid.unsqueeze(0)                                # (N,3)
    step = torch.sign(direction).long()                                     # -1/0/+1

    tmax = torch.full((N, 3), float('inf'), dtype=torch.float32, device=device)
    tdelta = torch.full((N, 3), float('inf'), dtype=torch.float32, device=device)
    for ax in range(3):
        nz = direction[:, ax].abs() > 1e-6
        boundary = (cur[:, ax].float() + torch.where(step[:, ax] > 0, 1.0, 0.0)) - o_grid[ax]
        tmax[:, ax] = torch.where(nz, boundary / direction[:, ax], tmax[:, ax])
        tdelta[:, ax] = torch.where(nz, (1.0 / direction[:, ax]).abs(), tdelta[:, ax])

    mask = torch.zeros((sx, sy, sz), dtype=torch.uint8, device=device)

    def inside(c):
        return ((c[:, 0] >= 0) & (c[:, 0] < sx) &
                (c[:, 1] >= 0) & (c[:, 1] < sy) &
                (c[:, 2] >= 0) & (c[:, 2] < sz))

    ib = inside(cur)
    if ib.any():
        c = cur[ib]
        mask[c[:, 0], c[:, 1], c[:, 2]] = 1

    active = (cur != end).any(dim=1)                                        # (N,)
    for _ in range(sx + sy + sz + 3):
        if not active.any():
            break
        smallest = tmax.argmin(dim=1)                                       # (N,) min axis
        for ax in range(3):
            sel = active & (smallest == ax)
            if sel.any():
                cur[sel, ax] += step[sel, ax]
                tmax[sel, ax] += tdelta[sel, ax]
        ib = inside(cur)
        mark = active & ib
        if mark.any():
            c = cur[mark]
            mask[c[:, 0], c[:, 1], c[:, 2]] = 1
        active = active & (cur != end).any(dim=1)

    return mask.cpu().numpy().astype(np.uint8)


def get_camera_projections(nusc, lidar_token):
    """Return per-camera projection info as dict( (intrinsic, ego2cam, w, h) ).

    The Occ3D grid lives in the ego frame, so this returns the ego-frame ->
    camera-frame transform (inverse of the camera -> ego calibration).
    """
    lidar_sd = nusc.get('sample_data', lidar_token)
    sample = nusc.get('sample', lidar_sd['sample_token'])

    cams = {}
    for name, sd_token in sample['data'].items():
        if not name.startswith('CAM'):
            continue
        sd = nusc.get('sample_data', sd_token)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        R_c2e = Quaternion(cs['rotation']).rotation_matrix       # camera -> ego
        t_c2e = np.array(cs['translation'], dtype=np.float64)

        # ego -> camera = inverse of (camera -> ego)
        ego2cam = np.eye(4, dtype=np.float64)
        ego2cam[:3, :3] = R_c2e.T
        ego2cam[:3, 3] = -R_c2e.T @ t_c2e

        intrinsic = np.array(cs['camera_intrinsic'], dtype=np.float64)
        cams[name] = (intrinsic, ego2cam, int(sd['width']), int(sd['height']))
    return cams


def build_camera_mask(centers, cams):
    """Project 3D points (N, 3) into every camera frustum; mark visible voxels.

    centers: continuous coordinates in the lidar frame.
    Returns a uint8 flat array of shape (N,).  Runs on GPU.
    """
    if centers.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8)

    device = torch.device('cuda')
    C = torch.from_numpy(np.asarray(centers, dtype=np.float32)).to(device)  # (N,3)
    N = C.shape[0]
    vis = torch.zeros(N, dtype=torch.bool, device=device)

    for _, (K, l2c, w, h) in cams.items():
        K_t = torch.from_numpy(np.asarray(K, dtype=np.float32)).to(device)
        l2c_t = torch.from_numpy(np.asarray(l2c, dtype=np.float32)).to(device)
        cam = C @ l2c_t[:3, :3].T + l2c_t[:3, 3]   # (N,3) camera frame
        depth = cam[:, 2]
        uv = cam @ K_t.T                            # (N,3) pixel-homogeneous
        z = uv[:, 2]
        u = uv[:, 0] / z
        v = uv[:, 1] / z
        inside = (z > 1e-3) & (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        vis |= inside

    return vis.cpu().numpy().astype(np.uint8)


def main(nusc, val_list, indice, nuscenesyaml, args, config):

    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml['learning_map']
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']
    accumulate_frames = int(config.get('accumulate_frames', 0))  # 0 = all keyframes
    dynamic_box_scale = float(config.get('dynamic_box_scale', 1.3))
    do_second_pass = bool(config.get('dynamic_remove_second_pass', True))
    dynamic_remove_min_z = float(config.get('dynamic_remove_min_z', 0.3))
    bev_cfg = config.get('bev_median_normalize', None) or {}
    bev_enabled = bool(bev_cfg.get('enabled', False))
    bev_bin = float(bev_cfg.get('bin_size', 1.0))
    bev_z_tol = float(bev_cfg.get('z_tol', 0.25))
    bev_quantile = float(bev_cfg.get('quantile', 0.1))
    bev_smooth_kernel = int(bev_cfg.get('smooth_kernel', 3))
    bev_snap = bool(bev_cfg.get('snap', False))
    bev_class_ids = [int(c) for c in bev_cfg.get('class_ids', [11, 12, 13, 14])]
    remove_isolated = bool(config.get('remove_isolated', True))
    isolated_neighborhood = int(config.get('isolated_neighborhood', 6))
    below_ground_cfg = config.get('remove_below_ground', None) or {}
    do_remove_below_ground = bool(below_ground_cfg.get('enabled', False))
    below_ground_class_ids = [int(c) for c in below_ground_cfg.get('class_ids', [11, 12, 13, 14])]
    below_ground_tolerance = int(below_ground_cfg.get('tolerance', 0))
    below_ground_fill_holes = bool(below_ground_cfg.get('fill_holes', True))

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'

    if args.split == 'train':
        if my_scene['token'] in val_list:
            print(f"[skip] scene_idx={indice} name={my_scene['name']} is in val split; not a train scene.")
            return
    elif args.split == 'val':
        if my_scene['token'] not in val_list:
            print(f"[skip] scene_idx={indice} name={my_scene['name']} is NOT in val split.")
            return
    elif args.split == 'all':
        pass
    else:
        raise NotImplementedError


    # load the first sample to start
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    # collect LiDAR sequence (keyframes only)
    dict_list = []
    keyframe_count = 0

    while True:
        # Option A: skip sweep frames. Sweeps have no 3D boxes, so dynamic
        # objects cannot be removed from them -> they become ghost trails.
        if not lidar_data['is_key_frame']:
            next_token = lidar_data['next']
            if next_token != '':
                lidar_data = nusc.get('sample_data', next_token)
                continue
            break
        ############################# get boxes ##########################
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
        boxes_token = [box.token for box in boxes]
        object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
        object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]

        ############################# get object categories ##########################
        converted_object_category = []
        for category in object_category:
            for (j, label) in enumerate(nuscenesyaml['labels']):
                if category == nuscenesyaml['labels'][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1 # Slightly expand the bbox to wrap all object points
        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename'] # load LiDAR names
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4]
        if lidar_data['is_key_frame']: # only key frame has semantic annotations
            lidar_sd_token = lidar_data['token']
            lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

        ############################# ego pose & lidar calibration (lidar -> ego) ##########################
        lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        R_l2e = Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix
        t_l2e = np.array(lidar_calibrated_sensor['translation'])

        ############################# cut out movable object points (in the LIDAR frame) ##########################
        # `get_sample_data` returns box annotations and pc0 both in the LIDAR_TOP
        # sensor frame.  Run the in-box test in that native frame to avoid any box
        # ego-transform error; the surviving points are moved to ego afterwards.
        # The ego-frame boxes are still stored separately for aggregation/placement.
        gt_bbox_3d_lidar = gt_bbox_3d                                  # lidar frame (in-box test)
        gt_bbox_3d = bbox_lidar_to_ego(gt_bbox_3d, lidar_calibrated_sensor)  # ego frame (stored)
        # Inflate the box used for dynamic-point removal so fuzzy object edges
        # are removed too (otherwise they leak into the static cloud -> trails).
        remove_box = gt_bbox_3d_lidar.astype(np.float32).copy()
        if dynamic_box_scale != 1.0:
            remove_box[:, 3:6] = remove_box[:, 3:6] * dynamic_box_scale
        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(remove_box[np.newaxis, :]))
        object_points_list = []
        j = 0
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:,j].bool()
            object_points = pc0[:, :3][object_points_mask]              # lidar frame
            object_points = (object_points @ R_l2e.T + t_l2e).astype(np.float32)  # -> ego
            object_points_list.append(object_points)
            j = j + 1

        moving_mask = torch.ones_like(points_in_boxes)
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes[0])

        ############################# get point mask of the vehicle itself ##########################
        range = config['self_range']
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > range[0]) |
                                        (np.abs(pc0[:, 1]) > range[1]) |
                                        (np.abs(pc0[:, 2]) > range[2]))

        ############################# get static scene segment (lidar frame -> ego) ##########################
        points_mask = points_mask & oneself_mask
        pc_lidar = pc0[:, :3][points_mask]                              # (N, 3) static points, lidar frame
        pc_ego = (pc_lidar @ R_l2e.T + t_l2e).astype(np.float32)        # -> ego

        ################## align static points to the first ego frame  ##################
        lidar_pc = ego_to_world_to_ego(pc_ego, lidar_ego_pose, lidar_ego_pose0)
        ################## record Non-key frame information into a dict  ########################
        dict = {"object_tokens": object_tokens,
                "object_points_list": object_points_list,
                "lidar_pc": lidar_pc,
                "lidar_ego_pose": lidar_ego_pose,
                "lidar_calibrated_sensor": lidar_calibrated_sensor,
                "lidar_token": lidar_data['token'],
                "is_key_frame": lidar_data['is_key_frame'],
                "gt_bbox_3d": gt_bbox_3d,
                "converted_object_category": converted_object_category,
                "pc_file_name": pc_file_name.split('/')[-1]}
        ################## record semantic information into the dict if it's a key frame  ########################
        if lidar_data['is_key_frame']:
            pc_with_semantic = pc_with_semantic[points_mask]              # (N, 4) lidar frame
            pc_with_semantic[:, :3] = (pc_with_semantic[:, :3].astype(np.float64) @ R_l2e.T + t_l2e)  # -> ego
            lidar_pc_with_semantic = ego_to_world_to_ego(pc_with_semantic.copy(),
                                                         lidar_ego_pose,
                                                         lidar_ego_pose0)
            dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic

        dict_list.append(dict)
        keyframe_count += 1
        if accumulate_frames > 0 and keyframe_count >= accumulate_frames:
            break
        ################## go to next frame of the sequence  ########################
        next_token = lidar_data['next']
        if next_token != '':
            lidar_data = nusc.get('sample_data', next_token)
        else:
            break

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
    lidar_pc = np.concatenate(lidar_pc_list, axis=0)

    ################## second pass: cross-frame dynamic-point removal  ########################
    # A box only covers a moving object at its own frame; points that leak past
    # the box stay in the static cloud and, once aligned to the anchor ego
    # frame across frames, leave a trail.  Re-apply EVERY frame's boxes in the
    # anchor ego frame to remove these leaked points.  Only points above
    # `dynamic_remove_min_z` are removed so we don't punch holes in the road at
    # historical object positions.
    if do_second_pass and lidar_pc.shape[0] > 0:
        all_boxes = []
        for d in dict_list:
            b = d['gt_bbox_3d']
            if b.shape[0] > 0:
                all_boxes.append(bbox_ego_to_ego(b.astype(np.float64).copy(),
                                                 d['lidar_ego_pose'], lidar_ego_pose0))
        if all_boxes:
            all_boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
            z_sel = lidar_pc[:, 2] > dynamic_remove_min_z
            if z_sel.any():
                pib = points_in_boxes_cpu(
                    torch.from_numpy(lidar_pc[z_sel, :3][np.newaxis, :, :].astype(np.float32)),
                    torch.from_numpy(all_boxes[np.newaxis, :, :]))
                in_any = pib[0].any(dim=1).numpy()
                z_idx = np.where(z_sel)[0]
                drop = z_idx[in_any]
                keep = np.ones(lidar_pc.shape[0], dtype=bool)
                keep[drop] = False
                lidar_pc = lidar_pc[keep]

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = []
    for dict in dict_list:
        if dict['is_key_frame']:
            lidar_pc_with_semantic_list.append(dict['lidar_pc_with_semantic'])
    lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=0)

    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []
    object_semantic = []
    for dict in dict_list:
        for i,object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo:
                if (dict['object_points_list'][i].shape[0] > 0):
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

    object_points_dict = {}

    for query_object_token in object_token_zoo:
        object_points_dict[query_object_token] = []
        for dict in dict_list:
            for i, object_token in enumerate(dict['object_tokens']):
                if query_object_token == object_token:
                    object_points = dict['object_points_list'][i]
                    if object_points.shape[0] > 0:
                        object_points = object_points[:,:3] - dict['gt_bbox_3d'][i][:3]
                        rots = dict['gt_bbox_3d'][i][6]
                        Rot = Rotation.from_euler('z', -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points)
                        object_points_dict[query_object_token].append(rotated_object_points)
                else:
                    continue
        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                axis=0)


    object_points_vertice = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_vertice.append(point_cloud[:,:3])
    # print('object finish')


    i = 0
    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
        if i >= len(dict_list):
            print('finish scene!')
            return
        dict = dict_list[i]
        is_key_frame = dict['is_key_frame']
        if not is_key_frame: # only use key frame as GT
            i = i + 1
            continue

        ################## convert the static scene to the target ego coordinate system ##############
        lidar_calibrated_sensor = dict['lidar_calibrated_sensor']
        lidar_ego_pose = dict['lidar_ego_pose']
        lidar_pc_i = ego_to_world_to_ego(lidar_pc.copy(), lidar_ego_pose0, lidar_ego_pose)
        lidar_pc_i_semantic = ego_to_world_to_ego(lidar_pc_with_semantic.copy(), lidar_ego_pose0, lidar_ego_pose)
        point_cloud = lidar_pc_i[:, :3]
        point_cloud_with_semantic = lidar_pc_i_semantic

        ################## load bbox of target frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(dict['lidar_token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1
        # get_sample_data returns boxes in the LIDAR sensor frame; move to ego frame
        gt_bbox_3d = bbox_lidar_to_ego(gt_bbox_3d, lidar_calibrated_sensor)
        rots = gt_bbox_3d[:,6:7]
        locs = gt_bbox_3d[:,0:3]

        ################## bbox placement ##############
        object_points_list = []
        object_semantic_list = []
        for j, object_token in enumerate(dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token==object_token_in_zoo:
                    points = object_points_vertice[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    if points.shape[0] >= 5:
                        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                              torch.from_numpy(gt_bbox_3d[j:j+1][np.newaxis, :]))
                        points = points[points_in_boxes[0,:,0].bool()]

                    object_points_list.append(points)
                    semantics = np.ones_like(points[:,0:1]) * object_semantic[k]
                    object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

        try: # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)
            scene_points = np.concatenate([point_cloud, temp])
        except:
            scene_points = point_cloud
        try:
            temp = np.concatenate(object_semantic_list)
            scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
        except:
            scene_semantic_points = point_cloud_with_semantic

        ################## BEV surface normalization (per-class z flattening) ##############
        if bev_enabled and scene_semantic_points.shape[0] > 0:
            scene_semantic_points = bev_surface_normalize(
                scene_semantic_points, bev_class_ids, bev_bin, bev_z_tol,
                bev_quantile, bev_smooth_kernel, bev_snap)
            # geometry follows the flattened semantic points so the ground
            # (and any other configured class) becomes a single flat layer
            scene_points = scene_semantic_points[:, :3].copy()

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < pc_range[3]) & (np.abs(scene_points[:, 1]) < pc_range[4]) \
               & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
        scene_points = scene_points[mask]

        ################## get mesh via Possion Surface Reconstruction ##############
        if args.to_mesh:
            point_cloud_original = o3d.geometry.PointCloud()
            with_normal2 = o3d.geometry.PointCloud()
            point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
            with_normal = preprocess(point_cloud_original, config)
            with_normal2.points = with_normal.points
            with_normal2.normals = with_normal.normals
            mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                           config['min_density'], with_normal2)
            scene_points = np.asarray(mesh.vertices, dtype=float)

        ################## remain points with a spatial range ##############
        mask = (np.abs(scene_points[:, 0]) < pc_range[3]) & (np.abs(scene_points[:, 1]) < pc_range[4]) \
               & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
        scene_points = scene_points[mask]

        ################## convert points to voxels ##############
        pcd_np = scene_points
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
        pcd_np = np.floor(pcd_np).astype(np.int64)
        # Poisson reconstruction may emit vertices slightly outside the grid;
        # drop any voxel index that falls outside occ_size.
        in_grid = (pcd_np[:, 0] >= 0) & (pcd_np[:, 0] < occ_size[0]) & \
                  (pcd_np[:, 1] >= 0) & (pcd_np[:, 1] < occ_size[1]) & \
                  (pcd_np[:, 2] >= 0) & (pcd_np[:, 2] < occ_size[2])
        pcd_np = pcd_np[in_grid]
        voxel = np.zeros(occ_size)
        voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

        ################## convert voxel coordinates to LiDAR system  ##############
        gt_ = voxel
        x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
        y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
        z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        vv = np.stack([X, Y, Z], axis=-1)
        fov_voxels = vv[gt_ > 0]
        fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
        fov_voxels[:, 0] += pc_range[0]
        fov_voxels[:, 1] += pc_range[1]
        fov_voxels[:, 2] += pc_range[2]

        ################## get semantics of sparse points  ##############
        mask = (np.abs(scene_semantic_points[:, 0]) < pc_range[3]) & (np.abs(scene_semantic_points[:, 1]) < pc_range[4]) \
               & (scene_semantic_points[:, 2] > pc_range[2]) & (scene_semantic_points[:, 2] < pc_range[5])
        scene_semantic_points = scene_semantic_points[mask]

        ################## Nearest Neighbor to assign semantics ##############
        dense_voxels = fov_voxels
        sparse_voxels_semantic = scene_semantic_points

        x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
        y = torch.from_numpy(sparse_voxels_semantic[:,:3]).cuda().unsqueeze(0).float()
        d1, d2, idx1, idx2 = chamfer.forward(x,y)
        indices = idx1[0].cpu().numpy()


        dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]

        ################## build Occ3D-style dense ground truth ##############
        # occupied voxel indices, axis order [X, Y, Z] == occ_size
        vox_idx = np.floor((fov_voxels[:, :3] - np.array(pc_range[:3])) / voxel_size).astype(np.int64)

        # semantics: 17 = free, 0 = others, 1..16 = classes (Occ3D convention)
        # SurroundOcc's learning_map already produces 0..16 with 0 = noise/others
        # and 1..16 = barrier..vegetation, which maps 1:1 onto Occ3D classes.
        semantics = np.full(occ_size, OCC3D_FREE, dtype=np.uint8)
        semantics[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = dense_semantic.astype(np.uint8)

        ################## remove isolated single-voxel noise ##############
        if remove_isolated:
            semantics = remove_isolated_voxels(semantics, OCC3D_FREE, isolated_neighborhood)

        ################## remove voxels below the ground floor ##############
        if do_remove_below_ground:
            semantics = remove_below_ground(semantics, OCC3D_FREE,
                                            below_ground_class_ids,
                                            below_ground_tolerance,
                                            fill_holes=below_ground_fill_holes)

        ################## mask_lidar: free space via lidar ray casting (ego frame) ##############
        lidar_sd = nusc.get('sample_data', dict['lidar_token'])
        raw_pc = np.fromfile(os.path.join(data_root, lidar_sd['filename']),
                             dtype=np.float32, count=-1).reshape(-1, 5)[..., :4]
        # lidar frame -> ego frame (points and ray origin)
        R_l2e = Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix
        t_l2e = np.array(lidar_calibrated_sensor['translation'], dtype=np.float64)
        pts_ego = (raw_pc[:, :3].astype(np.float64) @ R_l2e.T + t_l2e)
        rng_min = np.array(pc_range[:3], dtype=np.float64)
        rng_max = np.array(pc_range[3:6], dtype=np.float64)
        in_range = np.all((pts_ego >= rng_min) & (pts_ego <= rng_max), axis=1)
        lidar_pts = pts_ego[in_range]
        self_r = np.array(config['self_range'], dtype=np.float64)
        outside_self = (np.abs(lidar_pts[:, 0]) > self_r[0]) | \
                       (np.abs(lidar_pts[:, 1]) > self_r[1]) | \
                       (np.abs(lidar_pts[:, 2]) > self_r[2])
        mask_lidar = build_lidar_mask(lidar_pts[outside_self], occ_size, pc_range, voxel_size, origin=t_l2e)

        ################## mask_camera: voxels within any camera frustum ##############
        cams = get_camera_projections(nusc, dict['lidar_token'])
        all_idx = _grid_indices(occ_size)
        centers = (all_idx.astype(np.float64) + 0.5) * voxel_size + np.array(pc_range[:3], dtype=np.float64)
        mask_camera = build_camera_mask(centers, cams).reshape(occ_size).astype(np.uint8)

        ################## save Occ3D npz ##############
        sample_token = lidar_sd['sample_token']
        scene_name = my_scene['name']
        out_dir = os.path.join(save_path, 'gts', scene_name, sample_token)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.savez_compressed(
            os.path.join(out_dir, 'labels.npz'),
            semantics=semantics,
            mask_lidar=mask_lidar,
            mask_camera=mask_camera,
        )

        i = i + 1
        continue


def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:,:3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


def _occ_worker(gpu_id, indices, args, config, nuscenesyaml, val_list):
    """Process the given scene indices on a single GPU.

    Each worker runs as an independent subprocess whose environment already has
    `CUDA_VISIBLE_DEVICES` set to this GPU, so torch/CUDA (imported at the top
    of this module) bind to the correct GPU.  Setting it again here is a no-op
    safety net for the direct single-GPU call path.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    assert args.dataset == 'nuscenes'
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=True)
    for i in indices:
        print(f'[GPU {gpu_id}] processing sequecne: {i}', flush=True)
        main(nusc, val_list, indice=i,
             nuscenesyaml=nuscenesyaml, args=args, config=config)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--dataset', type=str, default='nuscenes')
    parse.add_argument('--config_path', type=str, default='config.yaml')
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--save_path', type=str, default='./data/GT_occupancy/')
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=850)
    parse.add_argument('--dataroot', type=str, default='./data/nuScenes/')
    parse.add_argument('--nusc_val_list', type=str, default='./nuscenes_val_list.txt')
    parse.add_argument('--label_mapping', type=str, default='nuscenes.yaml')
    parse.add_argument('--to_mesh', action='store_true', default=False,
                       help='Apply Poisson surface reconstruction before voxelization. '
                            'When omitted, voxelize the raw LiDAR points directly (much faster).')
    parse.add_argument('--gpus', type=str, default='0',
                       help='Comma-separated GPU ids to distribute scenes across, e.g. "0,1,2,3".')
    parse.add_argument('--scene_indices', type=str, default=None,
                       help='Explicit comma-separated scene indices for a single GPU worker '
                            '(used internally by the multi-GPU launcher).')
    args=parse.parse_args()


    if args.dataset=='nuscenes':
        val_list = []
        with open(args.nusc_val_list, 'r') as file:
            for item in file:
                val_list.append(item[:-1])
        file.close()
    else:
        raise NotImplementedError

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)


    # Distribute scene indices across the requested GPUs (round-robin).
    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]
    if args.scene_indices is not None:
        indices = [int(s) for s in args.scene_indices.split(',') if s.strip()]
    else:
        indices = list(range(args.start, args.end))

    if len(gpus) == 1:
        _occ_worker(gpus[0], indices, args, config, nuscenesyaml, val_list)
    else:
        import subprocess
        per_gpu = {g: [] for g in gpus}
        for k, idx in enumerate(indices):
            per_gpu[gpus[k % len(gpus)]].append(idx)

        # Launch one independent Python process per GPU.  `CUDA_VISIBLE_DEVICES`
        # must be set in the child's environment BEFORE Python imports torch /
        # the chamfer CUDA extension / mmcv / open3d (all imported at the top of
        # this module), otherwise every process ends up bound to GPU 0.
        # subprocess gives us an explicit per-child environment, which the older
        # multiprocessing-spawn approach could not provide in time.
        script = os.path.abspath(__file__)
        procs = []
        for g in gpus:
            if not per_gpu[g]:
                continue
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(g)
            cmd = [sys.executable, script,
                   '--dataset', args.dataset,
                   '--config_path', args.config_path,
                   '--split', args.split,
                   '--save_path', args.save_path,
                   '--dataroot', args.dataroot,
                   '--nusc_val_list', args.nusc_val_list,
                   '--label_mapping', args.label_mapping,
                   '--gpus', str(g),
                   '--scene_indices', ','.join(str(i) for i in per_gpu[g])]
            if args.to_mesh:
                cmd.append('--to_mesh')
            procs.append(subprocess.Popen(cmd, env=env))
        for p in procs:
            p.wait()
