#!/usr/bin/env python3
"""Infer the full nuScenes validation set using a float (or QAT) model.

For every sample the script writes three artefacts:

  <output_dir>/
    occ_preds/
      sample_NNNNNN_<token16>.bin       int16 argmax OCC grid  (200x200x16)
    quant_inputs/
      sample_NNNNNN_<token16>/
        imgs.bin     int16  - quantised camera images
        uv.bin       int16  - quantised UV feature coords
        zcam.bin     int16  - quantised camera depth
        points.bin   float32 - LiDAR points in ego frame (N_PTS x 4)
    occ_to_gt.json            {rel_pred_path: abs_gt_labels_npz_path}

The JSON file can be fed to eval_occ_bins.py to compute mIoU without
re-running the model.

Usage
-----
  # float mode, single GPU
  python tools/infer_float_dataset.py \\
      projects/configs/flashocc/flashocc-r50-M0_bevfusionocc_horizon_2.py \\
      --checkpoint work_dirs/.../epoch_24.pth \\
      --output-dir ./float_dataset_out

  # float mode, 4 GPUs (results are bit-identical to single-GPU)
  python tools/infer_float_dataset.py config.py -c ckpt.pth -o ./out --num-gpus 4

  # QAT mode (fake-quant VALIDATION, requires horizon_plugin_pytorch)
  python tools/infer_float_dataset.py \\
      projects/configs/flashocc/flashocc-r50-M0_bevfusionocc_horizon_2.py \\
      --mode qat \\
      --checkpoint work_dirs/.../calibrated_checkpoint.pth \\
      --march NASH_E \\
      --output-dir ./qat_dataset_out

  # process a sub-range (manual sharding)
  python tools/infer_float_dataset.py config.py -c ckpt.pth \\
      --start 0 --end 1000 -o ./out_shard0

  # skip saving quantised inputs
  python tools/infer_float_dataset.py config.py -c ckpt.pth \\
      --no-quant-inputs -o ./out_fast
"""

import argparse
import json
import logging
import os
import os.path as osp
import pickle
import sys

import numpy as np
import cv2
import torch
import torch.multiprocessing as mp
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm


DEFAULT_NUSCENES_ROOT = '/data01/chenmu/data/nuscenes-full'
DEFAULT_PKL = osp.join(DEFAULT_NUSCENES_ROOT,
                       'bevdetv2-nuscenes_infos_val.pkl')
DEFAULT_OUTPUT_DIR = './float_dataset_out'

FH, FW       = 256, 704
SRC_H, SRC_W = 900, 1600
CAM_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',]

IMG_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32) 
IMG_STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)  
N_PTS    = 35000

POINT_CLOUD_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4] 
QUANT_IMGS_SCALE    = 0.0177
QUANT_UV_SCALE      = 0.064469
QUANT_ZCAM_SCALE    = 0.00138971
QUANT_PREV_BEV_SCALE = 0.218627


# ---------------------------------------------------------------------------
# Plugin / config helpers
# ---------------------------------------------------------------------------

def _register_plugin(cfg):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', 'projects/mmdet3d_plugin/')
    _dir  = osp.dirname(plugin_dir.rstrip('/'))
    _name = osp.basename(plugin_dir.rstrip('/'))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    import importlib
    importlib.import_module(_name)

def _compute_uv(bev_grid, sample_offsets, frustum_hw, grid_size, downsample,
                sensor2ego, cam2imgs, post_rots, post_trans, bda, ego2sensor):
    fH, fW = frustum_hw
    B  = sensor2ego.shape[0]
    N  = sensor2ego.shape[1]
    Dx = int(grid_size[0].item())
    Dy = int(grid_size[1].item())
    Dz = int(grid_size[2].item())
    S  = sample_offsets.shape[0]

    bda_inv   = torch.inverse(bda)
    grid_flat = (bev_grid.to(sensor2ego)
                 .view(1, Dx * Dy * Dz, 3)
                 .expand(B, -1, -1))
    centres   = (bda_inv @ grid_flat.transpose(-1, -2)).transpose(-1, -2)

    offsets   = sample_offsets.to(sensor2ego)
    all_pts   = centres.unsqueeze(2) + offsets.reshape(1, 1, S, 3)
    all_pts_f = all_pts.reshape(B, Dx * Dy * Dz * S, 3)
    P = Dx * Dy * Dz * S

    R = ego2sensor[:, :, :3, :3]
    t = ego2sensor[:, :, :3,  3]
    pts = all_pts_f.unsqueeze(1).expand(B, N, P, 3)
    p_cam = (
        R.reshape(B * N, 3, 3) @
        pts.reshape(B * N, P, 3).transpose(-1, -2)
    ).transpose(-1, -2).view(B, N, P, 3) + t.view(B, N, 1, 3)

    z_cam   = p_cam[..., 2]
    z_safe  = z_cam.clamp(min=1.0).unsqueeze(-1)
    p_img   = (
        cam2imgs.reshape(B * N, 3, 3) @
        (p_cam / z_safe).reshape(B * N, P, 3).transpose(-1, -2)
    ).transpose(-1, -2).view(B, N, P, 3)
    uv_raw  = p_img[..., :2]

    ones    = torch.ones(B, N, P, 1, device=uv_raw.device, dtype=uv_raw.dtype)
    uvh     = torch.cat([uv_raw, ones], dim=-1)
    PR      = post_rots.reshape(B * N, 3, 3)
    PT      = post_trans.view(B, N, 1, 3)
    uv_aug  = (
        PR @ uvh.reshape(B * N, P, 3).transpose(-1, -2)
    ).transpose(-1, -2).view(B, N, P, 3) + PT

    uv_feat = uv_aug[..., :2] * (1.0 / downsample)
    return uv_feat, z_cam


def _build_bev_grid(cfg):
    vt_cfg      = cfg.model.img_view_transformer
    grid_config = vt_cfg['grid_config']
    downsample  = vt_cfg.get('downsample', 16)
    num_samples = vt_cfg.get('num_samples', (2, 2, 2))
    data_cfg    = getattr(cfg, 'data_config', {'input_size': (FH, FW)})
    input_size  = vt_cfg.get('input_size', data_cfg['input_size'])

    x_c, y_c, z_c = grid_config['x'], grid_config['y'], grid_config['z']
    Dx = round((x_c[1] - x_c[0]) / x_c[2])
    Dy = round((y_c[1] - y_c[0]) / y_c[2])
    Dz = round((z_c[1] - z_c[0]) / z_c[2])

    gs = torch.tensor([Dx, Dy, Dz], dtype=torch.float32)
    xs = torch.tensor([x_c[0] + (i + 0.5) * x_c[2] for i in range(Dx)])
    ys = torch.tensor([y_c[0] + (i + 0.5) * y_c[2] for i in range(Dy)])
    zs = torch.tensor([z_c[0] + (i + 0.5) * z_c[2] for i in range(Dz)])
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
    bev_grid   = torch.stack([gx, gy, gz], dim=-1).float()

    Sx, Sy, Sz = num_samples
    ox = torch.tensor([(-0.5 + (i + 0.5) / Sx) * x_c[2] for i in range(Sx)])
    oy = torch.tensor([(-0.5 + (i + 0.5) / Sy) * y_c[2] for i in range(Sy)])
    oz = torch.tensor([(-0.5 + (i + 0.5) / Sz) * z_c[2] for i in range(Sz)])
    gox, goy, goz  = torch.meshgrid(ox, oy, oz, indexing='ij')
    sample_offsets = torch.stack([gox, goy, goz], dim=-1).view(-1, 3).float()

    fH_feat = input_size[0] // downsample
    fW_feat = input_size[1] // downsample

    return bev_grid, sample_offsets, gs, downsample, (fH_feat, fW_feat)


def _prepare_inputs(info, nuscenes_root,
                    bev_grid, sample_offsets, grid_size, downsample, frustum_hw):
    """Build model inputs for one sample.

    Returns
    -------
    img_inputs    : list[Tensor]  [imgs(1,6,3,H,W), uv(1,6,P,2), zcam(1,6,P)]
    pts_tensor    : Tensor  (N_PTS, 4) float32, CPU
    imgs_np       : ndarray (1,6,3,H,W) float32  -- for quant-input saving
    uv_np         : ndarray (1,6,P,2)   float32  -- for quant-input saving
    zcam_np       : ndarray (1,6,P)     float32  -- for quant-input saving
    pts_np        : ndarray (N_PTS,4)   float32  -- for quant-input saving
    """
    resize        = float(FW) / float(SRC_W)
    resize_w      = int(SRC_W * resize)
    resize_h      = int(SRC_H * resize)
    crop_h_offset = int((1 - 0.0) * resize_h) - FH
    crop_w_offset = int(max(0, resize_w - FW) / 2)
    crop          = (crop_w_offset, crop_h_offset,
                     crop_w_offset + FW, crop_h_offset + FH)

    imgs_list, s2e_list, e2g_list, intr_list = [], [], [], []
    raw_imgs_list = []

    for cam_name in CAM_NAMES:
        cam_data = info['cams'][cam_name]
        img_path = cam_data['data_path']
        if not osp.exists(img_path):
            rel = (img_path.split('nuscenes/')[-1]
                   if 'nuscenes/' in img_path else img_path)
            img_path = osp.join(nuscenes_root, rel)
        if not osp.exists(img_path):
            raise FileNotFoundError(f'Image not found: {img_path}')

        img = Image.open(img_path).convert('RGB')
        img = img.resize((resize_w, resize_h))
        img = img.crop(crop)

        raw_img_np = np.array(img, dtype=np.uint8)
        raw_imgs_list.append(raw_img_np)

        img_np = np.array(img, dtype=np.float32)
        img_np = img_np[:, :, ::-1].copy()
        img_np = (img_np - IMG_MEAN) / IMG_STD
        imgs_list.append(img_np.transpose(2, 0, 1))

        K     = np.array(cam_data['cam_intrinsic'], dtype=np.float32)
        K_adj = K.copy()
        K_adj[0, 0] *= resize;  K_adj[1, 1] *= resize
        K_adj[0, 2]  = K[0, 2] * resize - crop_w_offset
        K_adj[1, 2]  = K[1, 2] * resize - crop_h_offset
        intr_list.append(K_adj)

        w, x, y, z = cam_data['sensor2ego_rotation']
        rot  = Quaternion(w, x, y, z).rotation_matrix.astype(np.float32)
        tran = np.array(cam_data['sensor2ego_translation'], dtype=np.float32)
        s2e  = np.eye(4, dtype=np.float32)
        s2e[:3, :3] = rot;  s2e[:3, 3] = tran
        s2e_list.append(s2e)

        w, x, y, z = cam_data['ego2global_rotation']
        e2g_rot  = Quaternion(w, x, y, z).rotation_matrix.astype(np.float32)
        e2g_tran = np.array(cam_data['ego2global_translation'], dtype=np.float32)
        e2g  = np.eye(4, dtype=np.float32)
        e2g[:3, :3] = e2g_rot;  e2g[:3, 3] = e2g_tran
        e2g_list.append(e2g)

    imgs_f32    = np.stack(imgs_list, axis=0)[np.newaxis]         # (1,6,3,H,W)
    sensor2egos = np.stack(s2e_list,  axis=0)[np.newaxis]         # (1,6,4,4)
    ego2globals = np.stack(e2g_list,  axis=0)[np.newaxis]         # (1,6,4,4)
    intrins_f32 = np.stack(intr_list, axis=0)[np.newaxis]         # (1,6,3,3)
    bda_rot     = np.eye(3, dtype=np.float32)[np.newaxis]         # (1,3,3)

    _s2e_d  = sensor2egos.astype(np.float64)     # (1,6,4,4)
    _e2g_d  = ego2globals.astype(np.float64)     # (1,6,4,4)
    keyego2global   = _e2g_d[:, 0:1, ...]        # (1,1,4,4)  — camera-0's ego
    global2keyego   = np.linalg.inv(keyego2global)            # (1,1,4,4)
    sensor2keyegos  = (global2keyego @ _e2g_d @ _s2e_d).astype(np.float32)  # (1,6,4,4)
    keyegos2sensor  = np.linalg.inv(sensor2keyegos).astype(np.float32)      # (1,6,4,4)

    pr_dummy    = np.tile(np.eye(3, dtype=np.float32), (1, 6, 1, 1))
    pt_dummy    = np.zeros((1, 6, 3), dtype=np.float32)

    lidar_path = info['lidar_path']
    if not osp.exists(lidar_path):
        rel = (lidar_path.split('nuscenes/')[-1]
               if 'nuscenes/' in lidar_path else lidar_path)
        lidar_path = osp.join(nuscenes_root, rel)
    if not osp.exists(lidar_path):
        raise FileNotFoundError(f'LiDAR not found: {lidar_path}')

    pts_raw     = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    l2e_rot     = Quaternion(info['lidar2ego_rotation']).rotation_matrix.astype(
        np.float32)
    l2e_tran    = np.array(info['lidar2ego_translation'], dtype=np.float32)
    pts_xyz_ego = pts_raw[:, :3] @ l2e_rot.T + l2e_tran
    pts_ego_raw = np.concatenate(
        [pts_xyz_ego, pts_raw[:, 3:4]], axis=1).astype(np.float32)

    pcr  = POINT_CLOUD_RANGE         
    mask = (
        (pts_ego_raw[:, 0] >= pcr[0]) & (pts_ego_raw[:, 0] <= pcr[3]) &
        (pts_ego_raw[:, 1] >= pcr[1]) & (pts_ego_raw[:, 1] <= pcr[4]) &
        (pts_ego_raw[:, 2] >= pcr[2]) & (pts_ego_raw[:, 2] <= pcr[5])
    )
    pts_in = pts_ego_raw[mask]

    n_in = pts_in.shape[0]
    if n_in >= N_PTS:
        # pts_np = pts_in[:N_PTS]
        raise Exception("Truncation should not be triggered")
    else:
        pad       = np.zeros((N_PTS - n_in, 4), dtype=np.float32)
        pad[:, 0] = 9999.0           # x far outside pc_range -> discarded by voxelizer
        pts_np    = np.concatenate([pts_in, pad], axis=0)

    _s2e  = torch.from_numpy(sensor2keyegos)
    _intr = torch.from_numpy(intrins_f32)
    _pr   = torch.from_numpy(pr_dummy)
    _pt   = torch.from_numpy(pt_dummy)
    _bda  = torch.from_numpy(bda_rot)
    _e2s  = torch.from_numpy(keyegos2sensor)
    _imgs = torch.from_numpy(imgs_f32)

    with torch.no_grad():
        uv_feat, z_cam = _compute_uv(
            bev_grid=bev_grid, sample_offsets=sample_offsets,
            frustum_hw=frustum_hw, grid_size=grid_size, downsample=downsample,
            sensor2ego=_s2e, cam2imgs=_intr,
            post_rots=_pr, post_trans=_pt,
            bda=_bda, ego2sensor=_e2s,
        )

    img_inputs = [_imgs, uv_feat, z_cam]
    pts_tensor = torch.from_numpy(pts_np)
    raw_imgs = np.stack(raw_imgs_list, axis=0)  # (6, H, W, 3) uint8

    return img_inputs, pts_tensor, imgs_f32, uv_feat.numpy(), z_cam.numpy(), pts_np, raw_imgs


def _quantize_s16(arr, scale):
    q = np.round(arr.astype(np.float32) / scale)
    return np.clip(q, -32768, 32767).astype(np.int16)


def _quantize_s8(arr, scale):
    """Quantize float32 array to int8 using the given scale."""
    q = np.round(arr.astype(np.float32) / scale)
    return np.clip(q, -128, 127).astype(np.int8)

# OCC visualization
_OCC_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

_OCC_COLOR_MAP = np.array([
    [0,   0,   0,   255],
    [255, 120,  50, 255],
    [255, 192, 203, 255],
    [255, 255,   0, 255],
    [0,  150, 245, 255],
    [0,  255, 255, 255],
    [200, 180,   0, 255],
    [255,   0,   0, 255],
    [255, 240, 150, 255],
    [135,  60,   0, 255],
    [160,  32, 240, 255],
    [255,   0, 255, 255],
    [175,   0,  75, 255],
    [ 75,   0,  75, 255],
    [150, 240,  80, 255],
    [230, 230, 250, 255],
    [0,  175,   0, 255],
    [255, 255, 255, 255],
], dtype=np.uint8)


def _occ2img(semantics):
    """Project 3D semantic voxels (H, W, D) to a top-down 2D image."""
    # print(semantics.shape)
    semantics = semantics[0]
    H, W, D = semantics.shape
    free_id = len(_OCC_CLASS_NAMES) - 1
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id
    for i in range(D):
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]
    viz = _OCC_COLOR_MAP[semantics_2d][..., :3]
    viz = cv2.resize(viz, dsize=(800, 800), interpolation=cv2.INTER_NEAREST)
    viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
    return viz


def _is_4d_model(cfg):
    #     mtype = cfg.model.type
    #     return '4D' in mtype or '4d' in mtype
    pass


def _get_prev_info(all_infos, global_idx):
    if global_idx <= 0:
        return None
    prev_info = all_infos[global_idx - 1]
    curr_info = all_infos[global_idx]
    if prev_info.get('scene_token', '') != curr_info.get('scene_token', ''):
        return None
    return prev_info


def _resolve_gt_path(info, nuscenes_root):
    """Return absolute path to labels.npz for this sample.

    occ_path in the pkl looks like:
        ./data/nuscenes/gts/scene-XXXX/<token>
    Strip './data/nuscenes' and resolve against nuscenes_root.
    """
    occ_path = info.get('occ_path', '')
    parts    = occ_path.lstrip('./').split('/')
    # parts: ['data', 'nuscenes', 'gts', 'scene-XXXX', '<token>']
    if len(parts) >= 3 and parts[0] == 'data':
        rel = '/'.join(parts[2:])   # 'gts/scene-XXXX/<token>'
    else:
        rel = '/'.join(parts)
    return osp.join(nuscenes_root, rel, 'labels.npz')


def _build_float_model(cfg, checkpoint_path, device, logger):
    from mmdet3d.models import build_model
    logger.info('Building float model ...')
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    model.img_view_transformer.deploy = True

    state      = torch.load(checkpoint_path, map_location='cpu')
    state_dict = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f'Missing keys ({len(missing)}): {missing[:5]} ...')
    if unexpected:
        logger.warning(f'Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...')

    model.to(device).eval()
    logger.info(f'Float model ready on {device}')
    return model


def _build_qat_model(cfg, checkpoint_path, march_str, device, logger):
    import horizon_plugin_pytorch as horizon
    from horizon_plugin_pytorch.march import March, set_march
    from horizon_plugin_pytorch.quantization import (
        FakeQuantState, set_fake_quantize)

    _hat_root = osp.join(osp.dirname(__file__), '..', 'horizon_flashocc')
    if _hat_root not in sys.path:
        sys.path.insert(0, _hat_root)
    from hat.utils import qconfig_manager

    march = getattr(March, march_str.upper(), None)
    if march is None:
        raise ValueError(f'Unknown march {march_str!r}.')
    set_march(march)
    logger.info(f'BPU march: {march_str}')

    from mmdet3d.models import build_model as _build
    logger.info('Building QAT model ...')
    model = _build(cfg.model, train_cfg=None, test_cfg=None)
    model.img_view_transformer.deploy = True

    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.CALIBRATION)
    model.qconfig = qconfig_manager.get_default_qconfig()
    if hasattr(model, 'set_qconfig'):
        model.set_qconfig()
    model.eval()
    model.to(device)
    model = horizon.quantization.prepare_qat(model)

    state      = torch.load(checkpoint_path, map_location='cpu')
    state_dict = state.get('state_dict', state)
    # Filter out minmax_scale keys (calibration artifacts not needed for
    # inference) — identical to test.py's _build_calibration_model.
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.endswith('minmax_scale')}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f'Missing keys ({len(missing)}): {missing[:5]} ...')
    if unexpected:
        logger.warning(f'Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...')

    set_fake_quantize(model, FakeQuantState.VALIDATION)
    model.eval()
    logger.info(f'QAT model ready on {device}')
    return model

def _run_worker(rank, world_size, args, cfg_path,
                infos, all_infos, global_start,
                pred_dir, quant_dir,
                bev_grid, sample_offsets, grid_size, downsample, frustum_hw):

    # Re-initialise logging (needed in spawned subprocesses)
    logging.basicConfig(
        format=(f'[GPU{rank}/{world_size}] %(asctime)-15s '
                f'%(levelname)s %(message)s'),
        level=logging.DEBUG if args.debug else logging.INFO,
        force=True,
    )
    logger = logging.getLogger(__name__ + f'.rank{rank}')

    # Each spawned process needs its own sys.path / plugin registration
    proj_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from mmcv import Config
    cfg = Config.fromfile(cfg_path)
    _register_plugin(cfg)

    # ------------------------------------------------------------------
    # Compute this rank's contiguous slice of infos
    #   infos[start_r : end_r]  <->  global indices [g0, g0 + chunk)
    # ------------------------------------------------------------------
    n = len(infos)
    base, rem = divmod(n, world_size)
    start_r  = rank * base + min(rank, rem)
    end_r    = start_r + base + (1 if rank < rem else 0)
    my_infos = infos[start_r:end_r]
    my_g0    = global_start + start_r   # global index of first sample

    # ------------------------------------------------------------------
    # Build model on this rank's GPU
    # ------------------------------------------------------------------
    if world_size == 1 and not torch.cuda.is_available():
        # Allow CPU fallback for single-GPU float mode
        device = torch.device(args.device)
    else:
        device = torch.device(f'cuda:{rank}')

    if args.mode == 'float':
        model = _build_float_model(cfg, args.checkpoint, device, logger)
    else:
        model = _build_qat_model(cfg, args.checkpoint, args.march, device, logger)

    # is_4d = _is_4d_model(cfg)
    is_4d = True
    if is_4d:
        logger.info('4D model detected – using forward_dummy with prev_bev')

    occ_to_gt = {}
    n_ok = n_skip = 0

    for i, info in enumerate(tqdm(my_infos,
                                  desc=f'GPU{rank}',
                                  unit='sample',
                                  position=rank,
                                  leave=(rank == 0))):
        global_idx = my_g0 + i
        token      = info.get('token', f'unk{global_idx}')
        stem       = f'sample_{global_idx:06d}_{token[:16]}'

        # ---- Prepare inputs --------------------------------------------------
        try:
            img_inputs, pts_tensor, imgs_np, uv_np, zcam_np, pts_np, raw_imgs = \
                _prepare_inputs(
                    info, args.nuscenes_root,
                    bev_grid, sample_offsets, grid_size, downsample, frustum_hw,
                )
        except Exception as exc:
            logger.warning(f'[{global_idx}] Input prep error: {exc}')
            n_skip += 1
            continue

        # Capture numpy arrays before tensors move to device
        # (forward pass may modify the img list in-place via quant stubs)
        uv_np_saved   = uv_np   if args.save_quant_inputs else None
        zcam_np_saved = zcam_np if args.save_quant_inputs else None
        raw_imgs_saved = raw_imgs if args.save_vis else None

        # ---- Forward pass ----------------------------------------------------
        try:
            img_in = [t.to(device) for t in img_inputs]
            pts_in = pts_tensor.to(device)

            if is_4d:
                
                prev_bev = None
                prev_info = _get_prev_info(all_infos, global_idx)
                if prev_info is not None:
                    try:
                        prev_img_inputs, _, _, _, _, _, _ = \
                            _prepare_inputs(
                                prev_info, args.nuscenes_root,
                                bev_grid, sample_offsets, grid_size,
                                downsample, frustum_hw,
                            )
                        prev_img_in = [t.to(device) for t in prev_img_inputs]
                        with torch.no_grad():
                            _, prev_bev = model.forward_dummy(
                                points    = [pts_in],
                                img_metas = [{}],
                                img_inputs= prev_img_in,
                                prev_bev  = None,
                            )
                    except Exception as pexc:
                        logger.warning(
                            f'[{global_idx}] Prev-frame error: {pexc}')

                with torch.no_grad():
                    occ_out, _ = model.forward_dummy(
                        points    = [pts_in],
                        img_metas = [{}],
                        img_inputs= img_in,
                        prev_bev  = prev_bev,
                    )
                pred = occ_out.cpu().numpy().astype(np.int16)
            else:
                # 3D model: use simple_test (existing path)
                with torch.no_grad():
                    occ_list = model.simple_test(
                        points    = [pts_in],
                        img_metas = [{}],
                        img       = img_in,
                    )
                pred = occ_list[0]
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                pred = pred.astype(np.int16)
        except Exception as exc:
            logger.warning(f'[{global_idx}] Forward error: {exc}')
            n_skip += 1
            continue

        
        pred_bin_path = osp.join(pred_dir, f'{stem}.bin')
        pred.tofile(pred_bin_path)

        if args.save_quant_inputs:
            qin_dir = osp.join(quant_dir, stem)
            os.makedirs(qin_dir, exist_ok=True)
            _quantize_s16(imgs_np,        QUANT_IMGS_SCALE).tofile(
                osp.join(qin_dir, 'imgs.bin'))
            _quantize_s16(uv_np_saved,    QUANT_UV_SCALE).tofile(
                osp.join(qin_dir, 'uv.bin'))
            _quantize_s16(zcam_np_saved,  QUANT_ZCAM_SCALE).tofile(
                osp.join(qin_dir, 'zcam.bin'))
            pts_np.tofile(
                osp.join(qin_dir, 'points.bin'))

            # Save quantised prev_bev for 4D models
            # print('prev_bev is None: ', prev_bev is None)
            if is_4d and prev_bev is not None:
                _quantize_s8(
                    prev_bev.cpu().numpy(), QUANT_PREV_BEV_SCALE
                ).tofile(osp.join(qin_dir, 'prev_bev.bin'))

        if args.save_vis:
            vis_dir = osp.join(quant_dir, stem, 'vis')
            os.makedirs(vis_dir, exist_ok=True)
            for cam_idx, cam_name in enumerate(CAM_NAMES):
                img_bgr = raw_imgs_saved[cam_idx][..., ::-1]
                cv2.imwrite(osp.join(vis_dir, f'{cam_name}.png'), img_bgr)
            occ_img = _occ2img(pred.astype(np.int32))
            cv2.imwrite(osp.join(vis_dir, 'occ_topdown.png'), occ_img)

        gt_path  = _resolve_gt_path(info, args.nuscenes_root)
        rel_pred = osp.relpath(pred_bin_path, args.output_dir)
        occ_to_gt[rel_pred] = gt_path
        n_ok += 1

        if args.debug:
            logger.debug(
                f'[{global_idx}] {stem}  pred={pred.shape}  '
                f'gt_exists={osp.exists(gt_path)}')

    # Each rank writes a partial JSON; main() merges them after mp.spawn returns
    partial = osp.join(args.output_dir, f'_partial_json_rank{rank:04d}.json')
    with open(partial, 'w') as fh:
        json.dump(occ_to_gt, fh, indent=2)

    logger.info(f'Rank {rank}: {n_ok} processed, {n_skip} skipped.')


def parse_args():
    p = argparse.ArgumentParser(
        description='Infer the full nuScenes val set and save OCC predictions.')
    p.add_argument('config',
                   help='mmdet3d config file path')
    p.add_argument('--checkpoint', '-c', required=True,
                   help='Float or calibrated .pth checkpoint')
    p.add_argument('--mode', choices=['float', 'qat'], default='float',
                   help='float: plain model | qat: fake-quant VALIDATION')
    p.add_argument('--march', default='NASH_E',
                   help='[qat] BPU march string (e.g. NASH_E / NASH_M / NASH_P)')
    p.add_argument('--pkl', default=DEFAULT_PKL,
                   help='Path to bevdetv2-nuscenes_infos_val.pkl')
    p.add_argument('--nuscenes-root', default=DEFAULT_NUSCENES_ROOT,
                   dest='nuscenes_root',
                   help='NuScenes data root directory')
    p.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR,
                   dest='output_dir',
                   help='Root output directory (created if absent)')
    p.add_argument('--device', default='cuda:0',
                   help='[float, single-GPU] torch device string (e.g. cuda:0, cpu)')
    p.add_argument('--num-gpus', type=int, default=1, dest='num_gpus',
                   help='Number of GPUs to use for parallel inference (default: 1). '
                        'Each GPU processes an independent contiguous chunk; '
                        'results are bit-identical to single-GPU inference.')
    p.add_argument('--start', type=int, default=0,
                   help='First sample index to process (inclusive)')
    p.add_argument('--end', type=int, default=-1,
                   help='Last sample index (exclusive); -1 = process all')
    p.add_argument('--save-quant-inputs', dest='save_quant_inputs',
                   action='store_true', default=True,
                   help='Save quantised HBM-format inputs per sample (default on)')
    p.add_argument('--no-quant-inputs', dest='save_quant_inputs',
                    action='store_false',
                    help='Skip quantised inputs (faster, less disk usage)')
    p.add_argument('--debug', action='store_true',
                    help='Enable DEBUG logging')
    p.add_argument('--save-vis', dest='save_vis',
                    action='store_true', default=False,
                    help='Save original camera images and OCC top-down '
                         'visualization under quant_inputs/<sample>/vis/')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)-15s %(levelname)s %(message)s',
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger = logging.getLogger(__name__)

    proj_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from mmcv import Config
    cfg = Config.fromfile(args.config)
    _register_plugin(cfg)

    logger.info(f'Config     : {args.config}')
    logger.info(f'Checkpoint : {args.checkpoint}')
    logger.info(f'Mode       : {args.mode}')
    logger.info(f'Num GPUs   : {args.num_gpus}')
    logger.info(f'Output dir : {args.output_dir}')

    # Validate multi-GPU arguments
    if args.num_gpus > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('--num-gpus > 1 requires CUDA.')
        n_avail = torch.cuda.device_count()
        if args.num_gpus > n_avail:
            raise ValueError(
                f'--num-gpus {args.num_gpus} exceeds available GPUs ({n_avail}).')

    # Load validation pkl
    logger.info(f'Loading PKL: {args.pkl}')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    all_infos = data['infos'] if isinstance(data, dict) else data
    logger.info(f'Total val samples: {len(all_infos)}')

    end_idx = len(all_infos) if args.end < 0 else min(args.end, len(all_infos))
    infos   = all_infos[args.start:end_idx]
    logger.info(
        f'Processing [{args.start}, {end_idx})  ->  {len(infos)} samples')

    pred_dir  = osp.join(args.output_dir, 'occ_preds')
    quant_dir = osp.join(args.output_dir, 'quant_inputs')
    os.makedirs(pred_dir, exist_ok=True)
    if args.save_quant_inputs:
        os.makedirs(quant_dir, exist_ok=True)

    bev_grid, sample_offsets, grid_size, downsample, frustum_hw = \
        _build_bev_grid(cfg)

    worker_kwargs = dict(
        world_size    = args.num_gpus,
        args          = args,
        cfg_path      = args.config,
        infos         = infos,
        all_infos     = all_infos,
        global_start  = args.start,
        pred_dir      = pred_dir,
        quant_dir     = quant_dir,
        bev_grid      = bev_grid,
        sample_offsets= sample_offsets,
        grid_size     = grid_size,
        downsample    = downsample,
        frustum_hw    = frustum_hw,
    )

    if args.num_gpus == 1:
        # ---- Single-GPU: run directly in this process (no subprocess overhead)
        _run_worker(rank=0, **worker_kwargs)
    else:
        # ---- Multi-GPU: spawn one process per GPU, each handles a contiguous
        #      chunk.  mp.spawn blocks until all ranks have finished.
        logger.info(f'Spawning {args.num_gpus} worker processes ...')
        mp.spawn(
            _run_worker,
            args=(
                worker_kwargs['world_size'],
                worker_kwargs['args'],
                worker_kwargs['cfg_path'],
                worker_kwargs['infos'],
                worker_kwargs['all_infos'],
                worker_kwargs['global_start'],
                worker_kwargs['pred_dir'],
                worker_kwargs['quant_dir'],
                worker_kwargs['bev_grid'],
                worker_kwargs['sample_offsets'],
                worker_kwargs['grid_size'],
                worker_kwargs['downsample'],
                worker_kwargs['frustum_hw'],
            ),
            nprocs=args.num_gpus,
            join=True,
        )

    # ---- Merge partial JSON files written by each rank ---------------------
    occ_to_gt = {}
    for r in range(args.num_gpus):
        partial = osp.join(args.output_dir, f'_partial_json_rank{r:04d}.json')
        if osp.exists(partial):
            with open(partial) as f:
                occ_to_gt.update(json.load(f))
            os.remove(partial)
        else:
            logger.warning(f'Partial JSON for rank {r} not found: {partial}')

    json_path = osp.join(args.output_dir, 'occ_to_gt.json')
    with open(json_path, 'w') as fh:
        json.dump(occ_to_gt, fh, indent=2)

    logger.info(f'Saved JSON map     -> {json_path}  ({len(occ_to_gt)} entries)')
    logger.info(f'Saved OCC preds    -> {pred_dir}')
    if args.save_quant_inputs:
        logger.info(f'Saved quant inputs -> {quant_dir}')
    logger.info(f'Done: {len(occ_to_gt)} predictions saved.')


if __name__ == '__main__':
    main()