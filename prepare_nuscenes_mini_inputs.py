#!/usr/bin/env python3

'''
Usage:
    python3 samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_0528/prepare_nuscenes_mini_inputs.py \
        --nuscenes-root ./data/nuscenes \
        --output-dir ./samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/mini_data/flashocc_0528_val \
        --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/custom/flashocc-r50-M0_bevfusionocc_horizon_2.py \
        --version v1.0-trainval \
        --split val

依赖：nuscenes-devkit、numpy、torch、pyquaternion、Pillow
'''

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nuscenes_splits
from PIL import Image
from pyquaternion import Quaternion


CAM_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]

FH, FW       = 256, 704
SRC_H, SRC_W = 900, 1600
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_PTS    = 30000

QUANT_IMGS_SCALE = 0.0177
QUANT_UV_SCALE   = 0.064469
QUANT_ZCAM_SCALE = 0.00138971


def _dict_to_namespace(d):
    """递归将 dict 转为 types.SimpleNamespace，支持属性访问。"""
    if isinstance(d, dict):
        ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
        # 保留 dict 的 get 方法
        ns.get = d.get
        return ns
    if isinstance(d, (list, tuple)):
        return type(d)(_dict_to_namespace(v) for v in d)
    return d


def _load_py_config(config_path: str):
    """用 exec 加载 .py 配置文件，返回 SimpleNamespace（替代 mmcv.Config）。"""
    cfg_path = Path(config_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Config not found: {cfg_path}')
    cfg_dict: dict = {}
    with open(cfg_path, 'r', encoding='utf-8') as f:
        code = f.read()
    # 处理 _base_ 继承（简单支持）
    exec(compile(code, str(cfg_path), 'exec'), cfg_dict)
    # 移除内置项
    cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith('__')}
    return _dict_to_namespace(cfg_dict)


def quantize_s16(arr, scale, zero_point=0):
    q = np.round(arr.astype(np.float32) / scale) + zero_point
    return np.clip(q, -32768, 32767).astype(np.int16)


def write_float_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.float32).tofile(path)


def _sensor_to_ego_matrix(cs_rec: dict) -> np.ndarray:
    """calibrated_sensor 记录 → 4x4 sensor→ego 变换矩阵。"""
    R = Quaternion(cs_rec['rotation']).rotation_matrix.astype(np.float32)
    t = np.array(cs_rec['translation'], dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def prepare_inputs(nusc: NuScenes, sample: dict, cfg=None):
    """从 NuScenes API 构建模型输入，返回 (img_inputs, points, raw_imgs)。"""

    resize        = float(FW) / float(SRC_W)
    resize_w      = int(SRC_W * resize)
    resize_h      = int(SRC_H * resize)
    crop_h_offset = int((1 - 0.0) * resize_h) - FH
    crop_w_offset = int(max(0, resize_w - FW) / 2)
    crop          = (crop_w_offset, crop_h_offset,
                     crop_w_offset + FW, crop_h_offset + FH)

    imgs_list, sensor2egos_list, intrins_list, raw_imgs = [], [], [], []

    for cam_name in CAM_NAMES:
        cam_sd = nusc.get('sample_data', sample['data'][cam_name])
        cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])

        img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
        assert os.path.exists(img_path), f'Image not found: {img_path}'

        img = Image.open(img_path).convert('RGB')
        img = img.resize((resize_w, resize_h), Image.Resampling.BILINEAR)
        img = img.crop(crop)
        raw_imgs.append(img)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - IMG_MEAN) / IMG_STD
        img_np = img_np.transpose(2, 0, 1)
        imgs_list.append(img_np)

        K     = np.array(cam_cs['camera_intrinsic'], dtype=np.float32)
        K_adj = K.copy()
        K_adj[0, 0] *= resize
        K_adj[1, 1] *= resize
        K_adj[0, 2]  = K[0, 2] * resize - crop_w_offset
        K_adj[1, 2]  = K[1, 2] * resize - crop_h_offset
        intrins_list.append(K_adj)

        s2e = _sensor_to_ego_matrix(cam_cs)
        sensor2egos_list.append(s2e)

    imgs_f32    = np.stack(imgs_list,        axis=0)[np.newaxis]
    sensor2egos = np.stack(sensor2egos_list, axis=0)[np.newaxis]
    intrins_f32 = np.stack(intrins_list,     axis=0)[np.newaxis]
    bda_rot     = np.eye(3, dtype=np.float32)[np.newaxis]
    ego2sensors = np.linalg.inv(sensor2egos).astype(np.float32)

    pr_dummy = np.tile(np.eye(3, dtype=np.float32), (1, 6, 1, 1))
    pt_dummy = np.zeros((1, 6, 3), dtype=np.float32)

    # LiDAR points（通过 NuScenes API 获取路径和标定）
    lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lid_cs = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    lidar_path = os.path.join(nusc.dataroot, lid_sd['filename'])
    assert os.path.exists(lidar_path), f'LiDAR not found: {lidar_path}'

    pts_raw     = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    l2e_rot     = Quaternion(lid_cs['rotation']).rotation_matrix.astype(np.float32)
    l2e_tran    = np.array(lid_cs['translation'], dtype=np.float32)
    pts_xyz_ego = pts_raw[:, :3] @ l2e_rot.T + l2e_tran
    pts_ego_raw = np.concatenate([pts_xyz_ego, pts_raw[:, 3:4]], axis=1).astype(np.float32)

    N_raw = pts_ego_raw.shape[0]
    if N_raw >= N_PTS:
        pts_ego = pts_ego_raw[:N_PTS]
    else:
        pts_ego = np.concatenate([pts_ego_raw,
                                  np.zeros((N_PTS - N_raw, 4), dtype=np.float32)], axis=0)

    # Compute UV and z_cam
    _sensor2keyegos = torch.from_numpy(sensor2egos)
    _intrins        = torch.from_numpy(intrins_f32)
    _post_rots      = torch.from_numpy(pr_dummy)
    _post_trans     = torch.from_numpy(pt_dummy)
    _bda            = torch.from_numpy(bda_rot)
    _keyegos2sensor = torch.from_numpy(ego2sensors)
    _imgs           = torch.from_numpy(imgs_f32)

    data_config = cfg.data_config if cfg is not None else {'input_size': (FH, FW)}
    vt_cfg      = cfg.model.img_view_transformer
    # vt_cfg may be a SimpleNamespace (attribute access) or a dict (subscript access)
    def _vt_get(key, default=None):
        if isinstance(vt_cfg, dict):
            return vt_cfg.get(key, default)
        return getattr(vt_cfg, key, default)
    grid_config = _vt_get('grid_config')
    if isinstance(grid_config, types.SimpleNamespace):
        grid_config = vars(grid_config)
    downsample  = _vt_get('downsample', 16)
    num_samples = _vt_get('num_samples', (2, 2, 2))
    _dc_input   = data_config['input_size'] if isinstance(data_config, dict) else getattr(data_config, 'input_size', (FH, FW))
    input_size  = _vt_get('input_size', _dc_input)

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
    gox, goy, goz = torch.meshgrid(ox, oy, oz, indexing='ij')
    sample_offsets = torch.stack([gox, goy, goz], dim=-1).view(-1, 3).float()

    fH_feat = input_size[0] // downsample
    fW_feat = input_size[1] // downsample

    with torch.no_grad():
        uv_feat, z_cam = _compute_uv(
            bev_grid=bev_grid, sample_offsets=sample_offsets,
            frustum_hw=(fH_feat, fW_feat), grid_size=gs, downsample=downsample,
            sensor2ego=_sensor2keyegos, cam2imgs=_intrins,
            post_rots=_post_rots, post_trans=_post_trans,
            bda=_bda, ego2sensor=_keyegos2sensor,
        )

    img_inputs = [_imgs, uv_feat, z_cam]
    points = torch.from_numpy(pts_ego)

    return img_inputs, points, raw_imgs


def _compute_uv(bev_grid, sample_offsets, frustum_hw, grid_size, downsample,
                sensor2ego, cam2imgs, post_rots, post_trans, bda, ego2sensor):
    fH, fW = frustum_hw
    B = sensor2ego.shape[0]
    N = sensor2ego.shape[1]
    Dx = int(grid_size[0].item())
    Dy = int(grid_size[1].item())
    Dz = int(grid_size[2].item())
    S  = sample_offsets.shape[0]

    bda_inv = torch.inverse(bda)
    grid_flat = (
        bev_grid.to(sensor2ego)
        .view(1, Dx * Dy * Dz, 3)
        .expand(B, -1, -1)
    )
    centres_real_ego = (
        bda_inv @ grid_flat.transpose(-1, -2)
    ).transpose(-1, -2)

    offsets = sample_offsets.to(sensor2ego)
    all_points = centres_real_ego.unsqueeze(2) + offsets.reshape(1, 1, S, 3)
    all_points_flat = all_points.reshape(B, Dx * Dy * Dz * S, 3)
    P = Dx * Dy * Dz * S

    R = ego2sensor[:, :, :3, :3]
    t = ego2sensor[:, :, :3,  3]
    pts = all_points_flat.unsqueeze(1).expand(B, N, P, 3)
    p_cam = (
        R.reshape(B * N, 3, 3) @
        pts.reshape(B * N, P, 3).transpose(-1, -2)
    ).transpose(-1, -2).view(B, N, P, 3) + t.view(B, N, 1, 3)

    z_cam = p_cam[..., 2]
    z_safe = z_cam.clamp(min=1.0).unsqueeze(-1)
    p_cam_norm = p_cam / z_safe
    p_img = (
        cam2imgs.reshape(B * N, 3, 3) @
        p_cam_norm.reshape(B * N, P, 3).transpose(-1, -2)
    ).transpose(-1, -2).view(B, N, P, 3)
    uv_raw = p_img[..., :2]

    ones = torch.ones(B, N, P, 1, device=uv_raw.device, dtype=uv_raw.dtype)
    uvh  = torch.cat([uv_raw, ones], dim=-1)
    PR   = post_rots.reshape(B * N, 3, 3)
    PT   = post_trans.view(B, N, 1, 3)
    uv_aug = (
        PR @ uvh.reshape(B * N, P, 3).transpose(-1, -2)
    ).transpose(-1, -2).view(B, N, P, 3) + PT
    uv_full = uv_aug[..., :2]
    uv_feat = uv_full * (1.0 / downsample)

    return uv_feat, z_cam


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate flashocc_0528 runtime inputs from nuScenes'
    )
    p.add_argument('--nuscenes-root', default=None, help='nuScenes dataset root')
    p.add_argument('--output-dir', required=True, help='Output directory for sample bins')
    p.add_argument('--config', required=True, help='mmdet3d config path')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--split', default='val',
                   help='nuScenes split to use: train / val / test / mini_train / mini_val')
    p.add_argument('--max-samples', type=int, default=-1)
    p.add_argument('--frames-lst-name', default='frames.lst')
    p.add_argument('--infer-example-dir', default=None,
                   help='Single sample from infer intermediate products')
    p.add_argument('--infer-sample-dir-name', default='sample_infer_example')
    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.infer_example_dir and not args.nuscenes_root:
        raise SystemExit('必须指定 --nuscenes-root 或 --infer-example-dir')

    # Load config
    cfg = _load_py_config(args.config)

    nuscenes_root = args.nuscenes_root or ''

    frames_lines: list[str] = []

    if args.infer_example_dir:
        infer_p = Path(args.infer_example_dir).resolve()
        sample_dir = out_root / args.infer_sample_dir_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 使用 NuScenes API 获取第一个 sample
        assert nuscenes_root, '使用 --infer-example-dir 时也需要 --nuscenes-root'
        nusc = NuScenes(version=args.version, dataroot=nuscenes_root, verbose=False)
        first_scene = nusc.scene[0]
        sample = nusc.get('sample', first_scene['first_sample_token'])

        [imgs, uv_feat, z_cam], points, _ = prepare_inputs(nusc, sample, cfg=cfg)
        imgs_np    = imgs.numpy()
        uv_np      = uv_feat.numpy()
        zcam_np    = z_cam.numpy()
        points_np  = points.numpy()

        imgs_s16 = quantize_s16(imgs_np, scale=QUANT_IMGS_SCALE)
        uv_s16   = quantize_s16(uv_np,   scale=QUANT_UV_SCALE)
        zcam_s16 = quantize_s16(zcam_np,  scale=QUANT_ZCAM_SCALE)

        imgs_s16.tofile(sample_dir / 'imgs.bin')
        uv_s16.tofile(sample_dir / 'uv.bin')
        zcam_s16.tofile(sample_dir / 'zcam.bin')
        points_np.astype(np.float32).tofile(sample_dir / 'points.bin')

        frames_lines.append(str(sample_dir.resolve()) + '\n')
        num_samples = 1
    else:
        nusc = NuScenes(version=args.version, dataroot=nuscenes_root, verbose=False)

        # 按 split 过滤 scene
        split_scenes = set(nuscenes_splits.create_splits_scenes()[args.split])

        samples = []
        for sc in nusc.scene:
            if sc['name'] not in split_scenes:
                continue
            tok = sc['first_sample_token']
            while tok:
                s = nusc.get('sample', tok)
                samples.append(s)
                tok = s['next']
        samples.sort(key=lambda x: x['timestamp'])
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        print(f'Split "{args.split}": {len(samples)} samples from {len(split_scenes)} scenes')
        for i, sample in enumerate(samples):
            token_short = sample['token'][:8]
            sample_dir = out_root / f'sample_{i:04d}_{token_short}'
            sample_dir.mkdir(parents=True, exist_ok=True)

            [imgs, uv_feat, z_cam], points, _ = prepare_inputs(
                nusc, sample, cfg=cfg
            )
            imgs_np   = imgs.numpy()
            uv_np     = uv_feat.numpy()
            zcam_np   = z_cam.numpy()
            points_np = points.numpy()

            imgs_s16 = quantize_s16(imgs_np, scale=QUANT_IMGS_SCALE)
            uv_s16   = quantize_s16(uv_np,   scale=QUANT_UV_SCALE)
            zcam_s16 = quantize_s16(zcam_np,  scale=QUANT_ZCAM_SCALE)

            imgs_s16.tofile(sample_dir / 'imgs.bin')
            uv_s16.tofile(sample_dir / 'uv.bin')
            zcam_s16.tofile(sample_dir / 'zcam.bin')
            points_np.astype(np.float32).tofile(sample_dir / 'points.bin')

            frames_lines.append(str(sample_dir.resolve()) + '\n')

            if (i + 1) % 20 == 0:
                print(f'已处理 {i + 1}/{len(samples)}')

        num_samples = len(samples)

    lst_path = out_root / args.frames_lst_name
    with open(lst_path, 'w', encoding='utf-8') as f:
        f.writelines(frames_lines)

    meta = {
        'num_samples': num_samples,
        'output_dir': str(out_root),
        'frames_list': str(lst_path),
        'quant_scales': {
            'imgs': QUANT_IMGS_SCALE,
            'uv': QUANT_UV_SCALE,
            'zcam': QUANT_ZCAM_SCALE,
        },
        'input_size': [FH, FW],
        'src_size': [SRC_H, SRC_W],
        'n_pts': N_PTS,
    }
    with open(out_root / 'prepare_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'完成：{num_samples} 个样本目录，列表: {lst_path}')


if __name__ == '__main__':
    main()