#!/usr/bin/env python3
# Copyright (c) 2023 Horizon Robotics. All Rights Reserved.

'''
仿照 ``bev/flashocc_henet_lss_occ3d_nuscenes/prepare_nuscenes_mini_inputs.py``，
为 ``QATBevFusionMultiSensorPreProcessMethod`` 生成每帧样本目录及 ``frames.lst``。

Usage:
    # 从 nuScenes 生成（多扫点云与训练 num_sweeps 对齐，ego2img 与标定一致）
    python3 samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes/prepare_nuscenes_mini_inputs.py \
        --nuscenes-root ./data/nuscenes \
        --output-dir ./samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/mini_data/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes \
        --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes.py \
        --num-sweeps 9

    # 与训练 infer 中间产物对齐（以 demo/example 的 lidar.npy / lidar2img.npy / ego2img.npy / jpg 为准）
    python3 samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes/prepare_nuscenes_mini_inputs.py \
        --infer-example-dir ./demo/example \
        --output-dir ./samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/mini_data/bevfusion_infer_from_example \
        --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes.py

每行 ``frames.lst`` 为一个目录，内含 ``bevfusion_pointpillar_multisensor_preprocess.json`` 所列文件::

    lidar_points.bin              float32 点云，每点 5 维 → pillar → features/coors
    img.bin                       float32，logical [6,3,512,960]，运行时 quant 为 INT8
    queries_rebatch_grid.bin      float32 → INT16  [6,64,80,2]
    restore_bev_grid.bin          float32 → INT16  [1,256,128,2]
    reference_points_rebatch.bin  float32 → INT16  [6,5120,2,2]
    bev_pillar_counts.bin         float32 → INT8   [1,16384,1]

辅助张量优先由模型 ``export_reference_points`` 导出；无 CUDA / 无权重时可 ``--skip-export``
写零占位（仅打通字节数）。

依赖：nuscenes-devkit、numpy、opencv-python、torch、pyquaternion；导出需 horizon_plugin_pytorch、hat、GPU。
'''

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nuscenes_splits
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

try:
    import torch
except ImportError as e:
    raise SystemExit("需要安装 torch: " + str(e)) from e

try:
    from torchvision.transforms.functional import resize
except ImportError:
    from torchvision.transforms._functional_tensor import resize

CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

DEFAULT_AUX_SHAPES = {
    "img": (6, 3, 512, 960),
    "queries_rebatch_grid": (6, 64, 80, 2),
    "restore_bev_grid": (1, 256, 128, 2),
    "reference_points_rebatch": (6, 5120, 2, 2),
    "bev_pillar_counts": (1, 16384, 1),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="nuScenes-mini → BevFusion 多传感器预处理 bin（样本目录 + frames.lst）"
    )
    p.add_argument(
        "--nuscenes-root",
        default=None,
        help="nuScenes 根目录；与 --infer-example-dir 二选一",
    )
    p.add_argument("--output-dir", required=True, help="输出根目录（mini_data/...）")
    p.add_argument(
        "--config",
        required=True,
        help="bevfusion_pointpillar_henet_multisensor_multitask_nuscenes.py",
    )
    p.add_argument(
        "--tools-dir",
        default="samples/ai_toolchain/horizon_model_train_sample/scripts/tools",
        help="含模型构建脚本的目录（插入 sys.path）",
    )
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument(
        "--split",
        default=None,
        help=(
            "nuScenes split 过滤：train / val / test / mini_train / mini_val。"
            "不指定则使用全部 scene（与原有行为一致）。"
        ),
    )
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument(
        "--vt-input-hw",
        default="16,30",
        help="export_reference_points 的 vt_input_hw，默认与 feats 30x16 对应 (H,W)",
    )
    p.add_argument("--export-homo-levels", type=int, default=3)
    p.add_argument(
        "--export-homo-layout",
        choices=("auto", "six", "single"),
        default="auto",
        help="仅写入 prepare_meta.json 备注",
    )
    p.add_argument("--export-homo-cam-index", type=int, default=1)
    p.add_argument("--save-export-npy", action="store_true")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="不调 export；辅助 bin 按形状填零",
    )
    p.add_argument("--frames-lst-name", default="frames.lst")
    p.add_argument(
        "--infer-example-dir",
        default=None,
        help=(
            "与训练侧 infer 中间产物一致：目录内需含 lidar.npy、(6,4,4) lidar2img.npy、"
            "ego2img.npy 及按 img0_…img5_ 排序的 6 路 jpg；写出一份样本 bin 供与 demo/example 对齐"
        ),
    )
    p.add_argument(
        "--infer-sample-dir-name",
        default="sample_infer_example",
        help="使用 --infer-example-dir 时，在 output-dir 下创建的样本目录名",
    )
    p.add_argument(
        "--num-sweeps",
        type=int,
        default=9,
        help="与配置 NuscenesBevSequenceDataset num_sweeps 一致，沿 LIDAR_TOP 的 prev 链合并点云",
    )
    return p.parse_args()


def resize_homo(homo: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    view = np.eye(4, dtype=np.float64)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    return view @ homo


def crop_homo(homo: np.ndarray, offset: tuple[float, float]) -> np.ndarray:
    view = np.eye(4, dtype=np.float64)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    return np.matmul(view, homo)


def homography_resize_and_crop(
    lidar2img_6: np.ndarray,
    resize_size: tuple[int, int],
    orig_hw: tuple[int, int],
    hat_cfg,
) -> np.ndarray:
    rs = resize_size
    vs = resize_size
    if hat_cfg is not None:
        rs = tuple(int(x) for x in hat_cfg.resize_shape[1:])
        vs = tuple(int(x) for x in hat_cfg.val_data_shape[1:])
    scale = (rs[0] / orig_hw[0], rs[1] / orig_hw[1])
    homo = resize_homo(lidar2img_6.astype(np.float64), scale).astype(np.float32)
    top = int(rs[0] - vs[0])
    left = int((rs[1] - vs[1]) / 2)
    return crop_homo(homo, (left, top))


def expand_homo_to_levels_numpy(homo_6_4_4: np.ndarray, num_levels: int) -> np.ndarray:
    h = np.asarray(homo_6_4_4, dtype=np.float32)
    if h.shape != (6, 4, 4):
        raise ValueError(f"期望 (6,4,4)，实际 {h.shape}")
    core = h.reshape(1, 1, 6, 1, 4, 4)
    return np.tile(core, (1, num_levels, 1, 1, 1, 1))


def expand_single_cam_to_levels_numpy(
    homo_6_4_4: np.ndarray, num_levels: int, cam_index: int
) -> np.ndarray:
    h = np.asarray(homo_6_4_4, dtype=np.float32)
    if h.shape != (6, 4, 4):
        raise ValueError(f"期望 (6,4,4)，实际 {h.shape}")
    one = h[cam_index]
    core = one.reshape(1, 1, 1, 1, 4, 4)
    return np.tile(core, (1, num_levels, 1, 1, 1, 1))


def sensor_to_ego_matrix(cs_rec: dict) -> np.ndarray:
    R = Quaternion(cs_rec["rotation"]).rotation_matrix
    t = np.array(cs_rec["translation"], dtype=np.float64)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def ego_to_global_matrix(ep_rec: dict) -> np.ndarray:
    R = Quaternion(ep_rec["rotation"]).rotation_matrix
    t = np.array(ep_rec["translation"], dtype=np.float64)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def sensor_to_global_matrix(nusc: NuScenes, sample_data: dict) -> np.ndarray:
    cs = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sample_data["ego_pose_token"])
    sensor2ego = sensor_to_ego_matrix(cs)
    ego2global = ego_to_global_matrix(ep)
    return ego2global @ sensor2ego


def get_lidar2img_4x4(nusc: NuScenes, sample: dict, cam_name: str) -> np.ndarray:
    lid_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cam_sd = nusc.get("sample_data", sample["data"][cam_name])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    lidar2global = sensor_to_global_matrix(nusc, lid_sd)
    cam2global = sensor_to_global_matrix(nusc, cam_sd)
    lidar2cam = np.linalg.inv(cam2global) @ lidar2global
    intrinsic = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
    lidar2img = np.eye(4, dtype=np.float64)
    lidar2img[:3, :4] = intrinsic @ lidar2cam[:3, :4]
    return lidar2img


def stack_lidar2img(nusc: NuScenes, sample: dict) -> np.ndarray:
    mats = [get_lidar2img_4x4(nusc, sample, c) for c in CAM_NAMES]
    return np.stack(mats, axis=0).astype(np.float64)


def get_ego2img_4x4(nusc: NuScenes, sample: dict, cam_name: str) -> np.ndarray:
    """车体(ego)坐标系到像素：仅用相机 calibrated_sensor（cam→ego）与内参。

    不再用「lidar 的 ego_pose」与「相机的 global 链」拼 ego2cam，避免两路时间戳
    不一致时几何错误（板端参考 demo/example 的 lidar2img/ego2img 时应对齐此定义）。
    """
    cam_sd = nusc.get("sample_data", sample["data"][cam_name])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    intrinsic = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
    cam2ego = sensor_to_ego_matrix(cam_cs)
    ego2cam = np.linalg.inv(cam2ego)
    ego2img = np.eye(4, dtype=np.float64)
    ego2img[:3, :4] = intrinsic @ ego2cam[:3, :4]
    return ego2img


def stack_ego2img(nusc: NuScenes, sample: dict) -> np.ndarray:
    mats = [get_ego2img_4x4(nusc, sample, c) for c in CAM_NAMES]
    return np.stack(mats, axis=0).astype(np.float64)


def _lidar_sd_to_global(nusc: NuScenes, sd: dict) -> np.ndarray:
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return ego_to_global_matrix(ep) @ sensor_to_ego_matrix(cs)


def _transform_points_to_key_lidar(
    pts: np.ndarray, src_sd: dict, key_sd: dict, nusc: NuScenes
) -> np.ndarray:
    if src_sd["token"] == key_sd["token"]:
        return pts
    t_src = _lidar_sd_to_global(nusc, src_sd)
    t_key = _lidar_sd_to_global(nusc, key_sd)
    t_rel = np.linalg.inv(t_key) @ t_src
    out = np.asarray(pts, dtype=np.float32, order="C").copy()
    xyz = out[:, :3].astype(np.float64)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([xyz, ones], axis=1)
    mapped = (t_rel @ hom.T).T[:, :3].astype(np.float32)
    out[:, :3] = mapped
    return out


def load_lidar_points_f32(
    nusc: NuScenes, sample: dict, max_points: int, num_sweeps: int
) -> np.ndarray:
    """沿 LIDAR_TOP 的 prev 链合并多扫，与配置中 num_sweeps 对齐后截断到 max_points。"""
    key_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pieces: list[np.ndarray] = []
    curr_sd = key_sd
    for _ in range(max(1, int(num_sweeps))):
        if curr_sd.get("channel") and curr_sd["channel"] != "LIDAR_TOP":
            break
        path = Path(nusc.dataroot) / curr_sd["filename"]
        pc = LidarPointCloud.from_file(str(path))
        pts = pc.points.T.astype(np.float32)
        if pts.shape[1] < 5:
            pad = np.zeros((pts.shape[0], 5 - pts.shape[1]), dtype=np.float32)
            pts = np.concatenate([pts, pad], axis=1)
        elif pts.shape[1] > 5:
            pts = pts[:, :5]
        pts = _transform_points_to_key_lidar(pts, curr_sd, key_sd, nusc)
        pieces.append(pts)
        prev = (curr_sd.get("prev") or "").strip()
        if not prev:
            break
        curr_sd = nusc.get("sample_data", prev)
    cat = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
    if cat.shape[0] > max_points:
        cat = cat[:max_points]
    return cat.reshape(-1)


def process_one_cam_bevfusion(
    img_path: str, resize_hw: tuple[int, int], pad_divisor: int, image_pad_fn
):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[0], rgb.shape[1]
    chw = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
    resized = resize(chw, list(resize_hw))
    padded = image_pad_fn(resized, "chw", None, pad_divisor, 0.0)
    return padded, (orig_h, orig_w)


def build_image_tensor_bevfusion(
    nusc: NuScenes, sample: dict, nusc_root: str, resize_hw: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """返回 (6,3,H,W) float32，与训练管线 BgrToYuv444(rgb_input=True)+Normalize(128) 一致。"""
    try:
        import horizon_plugin_pytorch as horizon
        from hat.data.transforms.functional_img import image_pad
    except ImportError as e:
        raise RuntimeError(
            "图像预处理需要 horizon_plugin_pytorch 与 hat。请在训练 Docker 中运行。"
        ) from e

    views: list[torch.Tensor] = []
    first_orig = None
    for cam in CAM_NAMES:
        sd = nusc.get("sample_data", sample["data"][cam])
        ip = os.path.join(nusc_root, sd["filename"])
        t, orig_hw = process_one_cam_bevfusion(ip, resize_hw, 32, image_pad)
        if first_orig is None:
            first_orig = orig_hw
        t_u8 = t.round().clamp(0, 255).to(dtype=torch.uint8)
        yuv = horizon.nn.functional.bgr_to_yuv444(t_u8.unsqueeze(0), True).squeeze(0).float()
        views.append((yuv - 128.0) / 128.0)
    stacked = torch.stack(views, dim=0).detach().cpu().numpy().astype(np.float32)
    return stacked, first_orig


def sorted_infer_jpg_paths(infer_dir: Path) -> list[str]:
    jpgs = [p for p in infer_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]

    def keyfn(p: Path) -> int:
        m = re.search(r"img(\d+)_", p.name, re.I)
        return int(m.group(1)) if m else 10**9

    return [str(p) for p in sorted(jpgs, key=keyfn)]


def build_image_tensor_from_paths(
    img_paths: list[str], resize_hw: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """与 build_image_tensor_bevfusion 相同图像管线，路径来自 infer 目录（如 demo/example）。"""
    try:
        import horizon_plugin_pytorch as horizon
        from hat.data.transforms.functional_img import image_pad
    except ImportError as e:
        raise RuntimeError(
            "图像预处理需要 horizon_plugin_pytorch 与 hat。请在训练 Docker 中运行。"
        ) from e

    views: list[torch.Tensor] = []
    first_orig: tuple[int, int] | None = None
    for ip in img_paths:
        t, orig_hw = process_one_cam_bevfusion(ip, resize_hw, 32, image_pad)
        if first_orig is None:
            first_orig = orig_hw
        t_u8 = t.round().clamp(0, 255).to(dtype=torch.uint8)
        yuv = horizon.nn.functional.bgr_to_yuv444(t_u8.unsqueeze(0), True).squeeze(0).float()
        views.append((yuv - 128.0) / 128.0)
    if not img_paths:
        raise ValueError("img_paths 为空，无法构建图像张量")
    stacked = torch.stack(views, dim=0).detach().cpu().numpy().astype(np.float32)
    assert first_orig is not None
    return stacked, first_orig
    if name in ref_dict:
        v = ref_dict[name]
        return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    suffixes = (name, name.replace("_rebatch", ""), "quant_" + name)
    for k, v in ref_dict.items():
        for s in suffixes:
            if k == s or k.endswith(s) or s in k:
                return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    return None


def resolve_export_reference_fn(module):
    if module is None:
        return None
    root_fn = getattr(module, "export_reference_points", None)
    if callable(root_fn):
        return root_fn

    def _score(cls_name: str) -> int:
        n = cls_name.lower()
        if n == "bevfusion":
            return -100
        if "fusion" in n:
            return -50
        if "bevformer" in n:
            return -40
        if "maptr" in n or "sparse_head" in n:
            return 50
        if "neck" in n or "backbone" in n:
            return 30
        return 0

    candidates: list[tuple[int, object]] = []
    for m in module.modules():
        fn = getattr(m, "export_reference_points", None)
        if callable(fn):
            candidates.append((_score(type(m).__name__), fn))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1] if candidates else None


def _ref_dict_from_export(ref) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if ref is None:
        return out
    if isinstance(ref, dict):
        for logical in DEFAULT_AUX_SHAPES:
            if logical == "img":
                continue
            arr = _match_export_tensor(logical, ref)
            if arr is not None:
                out[logical] = np.asarray(arr, dtype=np.float32)
        return out
    if isinstance(ref, (list, tuple)):
        keys_order = [
            "queries_rebatch_grid",
            "restore_bev_grid",
            "reference_points_rebatch",
            "bev_pillar_counts",
        ]
        idx = 0
        for t in ref:
            if idx >= len(keys_order):
                break
            if torch.is_tensor(t) and t.numel() > 0:
                k = keys_order[idx]
                arr = t.detach().cpu().numpy().astype(np.float32)
                tgt_sh = DEFAULT_AUX_SHAPES[k]
                if arr.size == int(np.prod(tgt_sh)):
                    arr = arr.reshape(tgt_sh)
                out[k] = arr
                idx += 1
    return out


def export_auxiliary_tensors(
    model,
    img_chw6: np.ndarray,
    lidar2img_6: np.ndarray,
    ego2img_6: np.ndarray,
    resize_size: tuple[int, int],
    orig_hw: tuple[int, int],
    vt_hw: tuple[int, int],
    hat_cfg=None,
    homo_num_levels: int = 3,
    ego_single_cam_index: int = 1,
    save_debug_npy_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    lidar_h = homography_resize_and_crop(lidar2img_6, resize_size, orig_hw, hat_cfg)
    ego_h = homography_resize_and_crop(ego2img_6, resize_size, orig_hw, hat_cfg)
    img_t = torch.from_numpy(img_chw6).float().cuda()

    ld_np = np.asarray(lidar_h, dtype=np.float32)
    eg_np = np.asarray(ego_h, dtype=np.float32)

    if save_debug_npy_dir is not None:
        dbg = Path(save_debug_npy_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        np.save(dbg / "lidar_h_6x4x4.npy", ld_np)
        np.save(dbg / "ego_h_6x4x4.npy", eg_np)

    export_fn = resolve_export_reference_fn(model)
    if export_fn is None:
        return {}

    vt_arg: tuple[int, int] | list[int] = vt_hw
    if hat_cfg is not None:
        vcfg = getattr(hat_cfg, "vt_input_hw", None)
        if vcfg is not None:
            vt_arg = tuple(int(x) for x in vcfg)

    def _try_inputs(inputs: dict, label: str) -> dict[str, np.ndarray] | None:
        last_e: BaseException | None = None
        for suffix, caller in (
            ("+vt(tuple)", lambda: export_fn(inputs, vt_arg)),
            ("+vt(list)", lambda: export_fn(inputs, list(vt_arg))),
            (" no vt", lambda: export_fn(inputs)),
        ):
            try:
                ref = caller()
                if ref is not None:
                    out = _ref_dict_from_export(ref)
                    if out:
                        print(f"[info] export_reference_points 成功（{label}{suffix}）")
                        return out
            except TypeError:
                continue
            except Exception as e:
                last_e = e
        if last_e is not None:
            print(f"[warn] {label} 失败: {last_e}")
        return None

    def _seq_meta(lidar_xy, ego_xy) -> dict:
        return {
            "img": img_t,
            "seq_meta": [
                {
                    "meta": [{"scene": "runtime_prepare"}],
                    "lidar2img": [lidar_xy],
                    "ego2img": [ego_xy],
                }
            ],
        }

    out = _try_inputs(
        _seq_meta(ld_np, eg_np),
        "seq_meta lidar2img/ego (6,4,4) np",
    )
    if out:
        return out

    lt = torch.from_numpy(ld_np.copy()).cuda().float().contiguous()
    et = torch.from_numpy(eg_np.copy()).cuda().float().contiguous()
    out = _try_inputs(_seq_meta(lt, et), "seq_meta (6,4,4) cuda")
    if out:
        return out

    base_in = _seq_meta(lt, et)
    for n_pts, tag in ((4096, "4k"), (1024, "1k"), (128, "128")):
        inp = dict(base_in)
        inp["points"] = [torch.zeros(n_pts, 5, dtype=torch.float32, device=img_t.device)]
        out = _try_inputs(inp, f"seq_meta+points(zero {tag})")
        if out:
            return out

    level_candidates: list[int] = []
    for lv in (homo_num_levels, 1, 2, 4):
        if lv >= 1 and lv not in level_candidates:
            level_candidates.append(lv)

    for L in level_candidates:
        try:
            lidar_m = expand_homo_to_levels_numpy(lidar_h, L)
            ego_m = expand_homo_to_levels_numpy(ego_h, L)
            lidar_s = expand_single_cam_to_levels_numpy(lidar_h, L, ego_single_cam_index)
            ego_s = expand_single_cam_to_levels_numpy(ego_h, L, ego_single_cam_index)
        except ValueError:
            continue

        if save_debug_npy_dir is not None:
            dbg = Path(save_debug_npy_dir)
            np.save(dbg / f"fallback_lidar_multicam_L{L}.npy", lidar_m)
            np.save(dbg / f"fallback_ego_single_L{L}.npy", ego_s)

        combos = (
            ("lidar(1,L,6,...)+ego(1,L,1,...)", lidar_m, ego_s),
            ("lidar+ego 均多相机", lidar_m, ego_m),
            ("lidar+ego 均单相机", lidar_s, ego_s),
        )
        for desc, lx, ey in combos:
            out = _try_inputs(_seq_meta(lx, ey), f"fallback L={L} {desc}")
            if out:
                return out
            inp = dict(_seq_meta(lx, ey))
            inp["points"] = [
                torch.zeros(1024, 5, dtype=torch.float32, device=img_t.device)
            ]
            out = _try_inputs(inp, f"fallback L={L} {desc}+points1024")
            if out:
                return out

    return {}


def write_float_bin(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.float32).tofile(path)


def init_model_for_reference_export(config_path: Path, tools_dir: Path):
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    import horizon_plugin_pytorch as horizon
    from hat.registry import RegistryContext, build_from_registry
    from hat.utils.config import Config

    cfg = Config.fromfile(str(config_path))
    horizon.march.set_march(cfg.march)

    def _try_predictor(pred, label: str):
        try:
            pipeline = build_from_registry(pred["model_convert_pipeline"])
            m = build_from_registry(pred["model"])
            m = pipeline(m)
            m.eval()
            m.cuda()
            if resolve_export_reference_fn(m) is not None:
                return m, label
        except Exception as e:
            print(f"[warn] {label} 构建失败: {e}")
        return None, ""

    def _run_partial_qat_pipeline(label: str, allowed_types: tuple[str, ...]):
        qat = getattr(cfg, "qat_predictor", None)
        if not qat:
            return None, ""
        try:
            mdl = build_from_registry(qat["model"])
            pl_cfg = copy.deepcopy(qat["model_convert_pipeline"])
            convs = [c for c in pl_cfg.get("converters", []) if c.get("type") in allowed_types]
            if not convs:
                return None, ""
            pl_cfg["converters"] = convs
            pipeline = build_from_registry(pl_cfg)
            mdl = pipeline(mdl)
            mdl.eval()
            mdl.cuda()
            if resolve_export_reference_fn(mdl) is not None:
                return mdl, label
        except Exception as e:
            print(f"[warn] {label} 构建失败: {e}")
        return None, ""

    def _try_cfg_model(attr: str, label: str):
        mdict = getattr(cfg, attr, None)
        if not mdict:
            return None, ""
        try:
            mdl = build_from_registry(mdict)
            mdl.eval()
            mdl.cuda()
            if resolve_export_reference_fn(mdl) is not None:
                return mdl, label
        except Exception as e:
            print(f"[warn] {label} 构建失败: {e}")
        return None, ""

    def _qat_bare():
        qat = getattr(cfg, "qat_predictor", None)
        if not qat:
            return None, ""
        try:
            mdl = build_from_registry(qat["model"])
            mdl.eval()
            mdl.cuda()
            if resolve_export_reference_fn(mdl) is not None:
                return mdl, "qat_bare_model"
        except Exception as e:
            print(f"[warn] qat bare 构建失败: {e}")
        return None, ""

    model = None
    strategy = "none"
    with RegistryContext():
        fp = getattr(cfg, "float_predictor", None)
        if fp is not None:
            model, strategy = _try_predictor(fp, "float_predictor")
        if model is None:
            model, strategy = _run_partial_qat_pipeline(
                "qat_rep_deploy_plus_load",
                ("RepModel2Deploy", "LoadCheckpoint"),
            )
        if model is None:
            model, strategy = _run_partial_qat_pipeline("qat_loadcheckpoint_only", ("LoadCheckpoint",))
        if model is None:
            model, strategy = _try_cfg_model("model", "cfg.model")
        if model is None:
            model, strategy = _try_cfg_model("deploy_model", "cfg.deploy_model")
        if model is None:
            model, strategy = _qat_bare()

    if model is not None:
        print(f"[info] export_reference_points 可用，策略: {strategy}")
    else:
        print("[warn] 无 export_reference_points；辅助 bin 将填零。")
    return model, cfg, strategy


def _write_logical_aux_bins(
    sample_dir: Path, aux_np: dict[str, np.ndarray], sample_index: int, skip_export: bool
) -> None:
    logical_files = [
        ("queries_rebatch_grid", "queries_rebatch_grid.bin"),
        ("restore_bev_grid", "restore_bev_grid.bin"),
        ("reference_points_rebatch", "reference_points_rebatch.bin"),
        ("bev_pillar_counts", "bev_pillar_counts.bin"),
    ]
    for logical, fname in logical_files:
        if logical in aux_np:
            arr = aux_np[logical]
        else:
            arr = np.zeros(DEFAULT_AUX_SHAPES[logical], dtype=np.float32)
            if sample_index == 0 and not skip_export:
                print(
                    f"[warn] export 未返回 {logical}，已零填充 shape={DEFAULT_AUX_SHAPES[logical]}"
                )
        write_float_bin(sample_dir / fname, arr)


def main(args=None):
    if args is None:
        args = parse_args()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.infer_example_dir and not args.nuscenes_root:
        raise SystemExit("必须指定 --nuscenes-root 或 --infer-example-dir")

    vt_parts = [int(x.strip()) for x in args.vt_input_hw.split(",")]
    if len(vt_parts) != 2:
        raise ValueError("--vt-input-hw 需两个整数，如 16,30")
    vt_hw = (vt_parts[0], vt_parts[1])

    model = None
    export_strategy = None
    hat_cfg = None

    if not args.skip_export:
        if not torch.cuda.is_available():
            raise RuntimeError("导出需要 CUDA，或使用 --skip-export")
        cfg_path = Path(args.config).resolve()
        tools_dir = Path(args.tools_dir).resolve()
        model, hat_cfg, export_strategy = init_model_for_reference_export(cfg_path, tools_dir)
        if hat_cfg is not None and getattr(hat_cfg, "vt_input_hw", None) is not None:
            v = hat_cfg.vt_input_hw
            vt_hw = tuple(int(x) for x in v)
            print(f"[info] vt_input_hw={vt_hw}")

    frames_lines: list[str] = []

    if args.infer_example_dir:
        infer_p = Path(args.infer_example_dir).resolve()
        for name in ("lidar.npy", "lidar2img.npy", "ego2img.npy"):
            if not (infer_p / name).is_file():
                raise FileNotFoundError(f"缺少 {infer_p / name}")
        sample_dir = out_root / args.infer_sample_dir_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        lidar = np.load(infer_p / "lidar.npy")
        if lidar.ndim != 2 or lidar.shape[1] < 5:
            raise ValueError(f"lidar.npy 期望形状 (N,>=5)，实际 {lidar.shape}")
        pts = lidar[:, :5].astype(np.float32)
        if pts.shape[0] > 40000:
            pts = pts[:40000]
        write_float_bin(sample_dir / "lidar_points.bin", pts.reshape(-1))

        resize_size = (512, 960)
        jpg_paths = sorted_infer_jpg_paths(infer_p)
        if len(jpg_paths) < 6:
            print(f"[warn] 仅找到 {len(jpg_paths)} 张 jpg，期望 6 路相机顺序 img0_…img5_")
        img_np, orig_hw = build_image_tensor_from_paths(jpg_paths, resize_size)
        write_float_bin(sample_dir / "img.bin", img_np)

        lidar2img = np.asarray(np.load(infer_p / "lidar2img.npy"), dtype=np.float64)
        ego2img = np.asarray(np.load(infer_p / "ego2img.npy"), dtype=np.float64)
        if lidar2img.ndim == 2 and lidar2img.size == 96:
            lidar2img = lidar2img.reshape(6, 4, 4)
        if ego2img.ndim == 2 and ego2img.size == 96:
            ego2img = ego2img.reshape(6, 4, 4)
        if lidar2img.shape[:2] != (6, 4) or ego2img.shape[:2] != (6, 4):
            raise ValueError(
                f"lidar2img/ego2img 期望 (6,4,4)，实际 {lidar2img.shape} / {ego2img.shape}"
            )

        aux_np: dict[str, np.ndarray] = {}
        if not args.skip_export:
            exported = export_auxiliary_tensors(
                model,
                img_np,
                lidar2img.astype(np.float32),
                ego2img.astype(np.float32),
                resize_size,
                orig_hw,
                vt_hw,
                hat_cfg=hat_cfg,
                homo_num_levels=args.export_homo_levels,
                ego_single_cam_index=args.export_homo_cam_index,
                save_debug_npy_dir=(
                    (sample_dir / "_export_debug") if args.save_export_npy else None
                ),
            )
            aux_np.update(exported or {})
        _write_logical_aux_bins(sample_dir, aux_np, 0, args.skip_export)
        frames_lines.append(str(sample_dir.resolve()) + "\n")
        num_samples = 1
    else:
        nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)

        # 按 split 过滤 scene（参考 flashocc_0528）
        split_scenes: set | None = None
        if args.split:
            split_scenes = set(nuscenes_splits.create_splits_scenes()[args.split])

        samples = []
        for sc in nusc.scene:
            if split_scenes is not None and sc["name"] not in split_scenes:
                continue
            tok = sc["first_sample_token"]
            while tok:
                s = nusc.get("sample", tok)
                samples.append(s)
                tok = s["next"]
        samples.sort(key=lambda x: x["timestamp"])
        if args.max_samples > 0:
            samples = samples[: args.max_samples]

        for i, sample in enumerate(samples):
            token_short = sample["token"][:8]
            sample_dir = out_root / f"sample_{i:04d}_{token_short}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            lidar_flat = load_lidar_points_f32(
                nusc, sample, max_points=40000, num_sweeps=args.num_sweeps
            )
            write_float_bin(sample_dir / "lidar_points.bin", lidar_flat)

            resize_size = (512, 960)
            img_np, orig_hw = build_image_tensor_bevfusion(
                nusc, sample, args.nuscenes_root, resize_size
            )
            write_float_bin(sample_dir / "img.bin", img_np)

            aux_np = {}
            if not args.skip_export:
                exported = export_auxiliary_tensors(
                    model,
                    img_np,
                    stack_lidar2img(nusc, sample),
                    stack_ego2img(nusc, sample),
                    resize_size,
                    orig_hw,
                    vt_hw,
                    hat_cfg=hat_cfg,
                    homo_num_levels=args.export_homo_levels,
                    ego_single_cam_index=args.export_homo_cam_index,
                    save_debug_npy_dir=(
                        (sample_dir / "_export_debug") if args.save_export_npy else None
                    ),
                )
                aux_np.update(exported or {})

            _write_logical_aux_bins(sample_dir, aux_np, i, args.skip_export)

            frames_lines.append(str(sample_dir.resolve()) + "\n")

            if (i + 1) % 20 == 0:
                print(f"已处理 {i + 1}/{len(samples)}")

        num_samples = len(samples)

    lst_path = out_root / args.frames_lst_name
    with open(lst_path, "w", encoding="utf-8") as f:
        f.writelines(frames_lines)

    meta = {
        "num_samples": num_samples,
        "output_dir": str(out_root),
        "frames_list": str(lst_path),
        "aux_shapes": {k: list(v) for k, v in DEFAULT_AUX_SHAPES.items()},
        "vt_input_hw": list(vt_hw),
        "export_homo_levels": args.export_homo_levels,
        "export_homo_layout": args.export_homo_layout,
        "skip_export": args.skip_export,
        "export_model_strategy": export_strategy,
        "infer_example_dir": args.infer_example_dir,
        "num_sweeps": args.num_sweeps,
        "split": args.split,
    }
    with open(out_root / "prepare_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"完成：{num_samples} 个样本目录，列表: {lst_path}")


if __name__ == "__main__":
    main()
