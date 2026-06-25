import argparse
import copy
import logging
import multiprocessing as mp
import os
import pickle
import sys
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from PIL import Image
from torch import Tensor
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*non-writable.*")

REPO_ROOT = "/data01/chenmu/bev_lss_v2"
sys.path.insert(0, REPO_ROOT)

import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.nn.functional import bgr_to_yuv444 as _bgr_to_yuv444
from horizon_plugin_pytorch.nn.quantized.functional_impl import (
    _voxelization as horizon_voxelization,
)
from pyquaternion import Quaternion

from hat.registry import OBJECT_REGISTRY

from hat.models.model_convert.converters import Float2QAT
from hat.models.model_convert.ckpt_converters import LoadCheckpoint

NUSCENES_ROOT = "/data01/chenmu/data/nuscenes-horizon"
VAL_PKL = os.path.join(NUSCENES_ROOT, "nuscenes_infos_val.pkl")

OCC3D_SEG_CLASS = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]
OCC3D_SEG_CLASS_FULL = OCC3D_SEG_CLASS + ["free"]
NUM_CLASSES = len(OCC3D_SEG_CLASS)
IGNORE_INDEX = 17

RESIZE_SHAPE = (540, 960)
CROP_SHAPE = (512, 960)

POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
VOXEL_SIZE = [0.4, 0.4, 6.4]
MAX_POINTS_IN_VOXEL = 20
MAX_VOXELS_TRAIN = 30000
MAX_VOXELS_VAL = 40000
HBM_VOXELS = 30000
NORM_RANGE = [-40, -40, -1, 0, 40, 40, 5.4, 255.0]
NORM_DIMS = [0, 1, 2, 3]

CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

HBM_QUANT_PARAMS = {
    "features": {"scale": 0.00771821, "dtype": "s8"},
    "coors": {"scale": None, "dtype": "s32"},
    "img": {"scale": 0.00686275, "dtype": "s8"},
    "ego2img": {"scale": None, "dtype": "f32"},
    "points0": {"scale": 0.0078125, "dtype": "s16"},
    "points1": {"scale": 0.015625, "dtype": "s16"},
}

OCC_COLOR_MAP = np.array([
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_registry_initialized = False


def _ensure_registry():
    
    global _registry_initialized
    if _registry_initialized:
        return
    from hat.registry import register_default_config
    register_default_config()
    _registry_initialized = True


def _instantiate(config_dict):
    """Recursively instantiate objects from config dicts.

    Local replacement for ``hat.registry.build_from_registry``.
    """
    _ensure_registry()

    if not isinstance(config_dict, dict):
        return config_dict

    if "type" not in config_dict:
        return {k: _instantiate(v) for k, v in config_dict.items()}

    cls_type = config_dict["type"]
    cls = OBJECT_REGISTRY.get(cls_type)

    kwargs = {}
    for k, v in config_dict.items():
        if k == "type":
            continue
        if isinstance(v, dict) and "type" in v:
            kwargs[k] = _instantiate(v)
        elif isinstance(v, (list, tuple)):
            kwargs[k] = type(v)(
                _instantiate(item) if isinstance(item, dict) and "type" in item
                else item
                for item in v
            )
        else:
            kwargs[k] = v

    return cls(**kwargs)


def _compute_ego2img(cam: dict) -> np.ndarray:
    """Compute ego→image projection matrix from camera calibration.

    Replicates ``NuscenesSample._get_homography_by_cam``.

    Args:
        cam: dict with keys ``sensor2ego_translation`` (3-list),
             ``sensor2ego_rotation`` (4-list quaternion), and
             ``cam_intrinsic`` (3×3 ndarray).

    Returns:
        ego2img: (4, 4) float64 ndarray.
    """
    s2e_t = np.array(cam["sensor2ego_translation"], dtype=np.float64)
    s2e_r = np.array(cam["sensor2ego_rotation"], dtype=np.float64)
    rotation = Quaternion(s2e_r).rotation_matrix

    ego2sensor_r = np.linalg.inv(rotation)
    ego2sensor_t = s2e_t @ ego2sensor_r.T
    ego2sensor = np.eye(4, dtype=np.float64)
    ego2sensor[:3, :3] = ego2sensor_r.T
    ego2sensor[3, :3] = -ego2sensor_t

    K = np.array(cam["cam_intrinsic"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: K.shape[0], : K.shape[1]] = K

    ego2img = viewpad @ ego2sensor.T
    return ego2img


def _compute_lidar2ego(info: dict) -> np.ndarray:
    l2e_t = np.array(info["lidar2ego_translation"], dtype=np.float64)
    l2e_r = np.array(info["lidar2ego_rotation"], dtype=np.float64)
    l2e_m = np.eye(4, dtype=np.float64)
    l2e_m[:3, :3] = Quaternion(l2e_r).rotation_matrix
    l2e_m[:3, 3] = l2e_t
    return l2e_m


def load_sample(idx: int, info: dict) -> dict:
    imgs = []
    ego2imgs = []
    for cam_name in CAM_NAMES:
        cam = info["cams"][cam_name]
        img_path = os.path.join(NUSCENES_ROOT, cam["data_path"])
        img = Image.open(img_path).convert("RGB")
        imgs.append(img)
        ego2imgs.append(_compute_ego2img(cam))

    lidar_path = info["lidar_path"]
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points[:, 4] = 0
    points = points[:, [0, 1, 2, 3, 4]]

    lidar2ego = _compute_lidar2ego(info)

    occ_path = os.path.join(NUSCENES_ROOT, info["occ_path"], "labels.npz")
    occ_labels = np.load(occ_path)
    gt_occ_info = {
        "voxel_semantics": occ_labels["semantics"],
        "mask_lidar": occ_labels["mask_lidar"],
        "mask_camera": occ_labels["mask_camera"],
    }

    return {
        "img": imgs,
        "ego2img": ego2imgs,
        "points": points,
        "lidar2ego": lidar2ego,
        "gt_occ_info": gt_occ_info,
    }


def _lidar2ego_points(points: np.ndarray, lidar2ego: np.ndarray) -> np.ndarray:
    xyz = points[:, :3].astype(np.float64)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([xyz, ones], axis=1).T
    transformed = (lidar2ego.astype(np.float64) @ points_h).T
    points_ego = transformed[:, :3].astype(points.dtype)
    if points.shape[1] > 3:
        points_ego = np.concatenate([points_ego, points[:, 3:]], axis=1)
    return points_ego


def _resize_img(img: Image.Image, target_hw: Tuple[int, int]) -> Image.Image:
    return TVF.resize(img, list(target_hw), TVF.InterpolationMode.BICUBIC)


def _resize_mat(mat: np.ndarray, orig_hw: Tuple[int, int],
                new_hw: Tuple[int, int]) -> np.ndarray:
    sy = new_hw[0] / orig_hw[0]
    sx = new_hw[1] / orig_hw[1]
    view = np.eye(4, dtype=mat.dtype)
    view[0, 0] = sx
    view[1, 1] = sy
    return view @ mat


def _crop_mat(mat: np.ndarray, left: float, top: float) -> np.ndarray:
    
    view = np.eye(4, dtype=mat.dtype)
    view[0, 2] = -left
    view[1, 2] = -top
    return view @ mat


def _pil_to_tensor(img: Image.Image) -> Tensor:
    return TVF.pil_to_tensor(img)


def _bgr_to_yuv444_tensor(img: Tensor, rgb_input: bool = False) -> Tensor:
    batched = img.unsqueeze(0)
    if batched.dtype != torch.uint8:
        batched = batched.to(torch.uint8)
    yuv = _bgr_to_yuv444(batched, rgb_input)
    return yuv.squeeze(0)


def _normalize(img: Tensor, mean: float = 128.0, std: float = 128.0) -> Tensor:
    return (img.float() - mean) / std


def apply_test_transforms(data: dict) -> dict:
    data["points"] = _lidar2ego_points(data["points"], data["lidar2ego"])

    imgs = data["img"]
    ego2imgs = data["ego2img"]
    for i, img in enumerate(imgs):
        orig_h, orig_w = img.size[1], img.size[0]
        imgs[i] = _resize_img(img, RESIZE_SHAPE)
        ego2imgs[i] = _resize_mat(ego2imgs[i], (orig_h, orig_w), RESIZE_SHAPE)

    target_h, target_w = CROP_SHAPE
    for i, img in enumerate(imgs):
        w, h = img.size
        top = h - target_h
        left = (w - target_w) / 2.0
        imgs[i] = TVF.crop(img, int(top), int(left), target_h, target_w)
        ego2imgs[i] = _crop_mat(ego2imgs[i], left, top)

    for i, img in enumerate(imgs):
        t = _pil_to_tensor(img)
        t = _bgr_to_yuv444_tensor(t, rgb_input=True)
        t = _normalize(t, mean=128.0, std=128.0)
        imgs[i] = t

    data["img"] = imgs
    data["ego2img"] = ego2imgs
    return data


def _get_paddings_indicator(actual_num: Tensor, max_num: int, axis: int = 0) -> Tensor:
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num_t = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device
    ).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num_t
    return paddings_indicator


def _voxel_feature_encoder(
    norm_range: Tensor,
    norm_dims: List[int],
    features: Tensor,
    num_points_in_voxel: Tensor,
) -> Tensor:
    half = len(norm_range) // 2
    for idx, dim in enumerate(norm_dims):
        start = norm_range[idx]
        norm = norm_range[idx + half] - norm_range[idx]
        features[:, :, dim] = (features[:, :, dim] - start) / norm

    voxel_count = features.shape[1]
    mask = _get_paddings_indicator(num_points_in_voxel, voxel_count, axis=0)
    mask = mask.unsqueeze(-1).type_as(features)
    features = features * mask

    features = features.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
    return features


def voxelize_lidar(
    points_list: List[torch.Tensor],
    pc_range: List[float],
    voxel_size: List[float],
    max_points_in_voxel: int,
    max_voxels: int,
    norm_range: List[float],
    norm_dims: List[int],
    is_deploy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Voxelize LiDAR point clouds

    This combines ``BatchVoxelization.forward`` and
    ``CenterPointPreProcess.forward``.

    Args:
        points_list: List of (N, 5) tensors on GPU.
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        voxel_size: [vx, vy, vz]
        max_points_in_voxel: Max points per voxel (e.g. 20).
        max_voxels: Max number of voxels.
        norm_range: [min_x, min_y, min_z, min_intensity,
                     max_x, max_y, max_z, max_intensity]
        norm_dims: Dimensions to normalize, e.g. [0, 1, 2, 3].
        is_deploy: Whether in deploy mode (passed as ``use_max`` to voxelization).

    Returns:
        features: (1, 5, max_points, num_voxels) float tensor.
        coors_batch: (num_voxels, 4) int32 tensor — padded (batch_idx, z, y, x).
    """
    device = points_list[0].device
    pc_range_t = torch.tensor(pc_range, device=device)
    voxel_size_t = torch.tensor(voxel_size, device=device)
    norm_range_t = torch.tensor(norm_range, device=device)

    voxel_lst: List[torch.Tensor] = []
    coors_lst: List[torch.Tensor] = []
    num_pts_lst: List[torch.Tensor] = []

    for points in points_list:
        voxels, coors, num_pts = horizon_voxelization(
            points,
            voxel_size=voxel_size_t,
            pc_range=pc_range_t,
            max_points_per_voxel=max_points_in_voxel,
            max_voxels=max_voxels,
            use_max=is_deploy,
        )
        voxel_lst.append(voxels)
        coors_lst.append(coors)
        num_pts_lst.append(num_pts)

    voxel_feature = torch.cat(voxel_lst, dim=0)
    num_points_per_voxel = torch.cat(num_pts_lst, dim=0)

    coors_batch_lst = []
    for i, coor in enumerate(coors_lst):
        coor_pad = F.pad(coor, (1, 0), mode="constant", value=float(i))
        coors_batch_lst.append(coor_pad)
    coors_batch = torch.cat(coors_batch_lst, dim=0).long()

    features = _voxel_feature_encoder(
        norm_range=norm_range_t,
        norm_dims=norm_dims,
        features=voxel_feature,
        num_points_in_voxel=num_points_per_voxel,
    )

    return features, coors_batch


def load_config_vars(config_path: str):
    
    import importlib.util

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(tools_dir)
    os.chdir(scripts_dir)

    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def build_deploy_model_with_calib_weights(
    cfg, ckpt_dir: str, checkpoint_name: str, device: str
):
    
    horizon.march.set_march(cfg.march)

    deploy_model = _instantiate(cfg.deploy_model)
    deploy_model.train()

    calib_ckpt = os.path.join(ckpt_dir, checkpoint_name)

    pipeline_cfg = copy.deepcopy(cfg.deploy_model_convert_pipeline)
    qat_cfg = None
    load_ckpt_cfg = None
    for c in pipeline_cfg["converters"]:
        if c["type"] == "Float2QAT":
            qat_cfg = copy.deepcopy(c)
        elif c["type"] == "LoadCheckpoint":
            load_ckpt_cfg = copy.deepcopy(c)

    if qat_cfg is None or load_ckpt_cfg is None:
        raise RuntimeError("Could not find Float2QAT/LoadCheckpoint in pipeline config")

    load_ckpt_cfg["checkpoint_path"] = calib_ckpt

    qat_kwargs = {k: v for k, v in qat_cfg.items() if k != "type"}
    for k in list(qat_kwargs.keys()):
        if isinstance(qat_kwargs[k], dict) and "type" in qat_kwargs[k]:
            qat_kwargs[k] = _instantiate(qat_kwargs[k])

    qat_converter = Float2QAT(**qat_kwargs)
    deploy_model = qat_converter(deploy_model)

    load_kwargs = {k: v for k, v in load_ckpt_cfg.items() if k != "type"}
    load_converter = LoadCheckpoint(**load_kwargs)
    deploy_model = load_converter(deploy_model)

    deploy_model.eval()
    deploy_model = deploy_model.to(device)
    horizon.quantization.set_fake_quantize(
        deploy_model, horizon.quantization.FakeQuantState.VALIDATION
    )
    return deploy_model


def prepare_deploy_inputs(
    sample: dict,
    deploy_model,
    cfg,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Prepare the 6 deploy-model inputs, all on GPU.

    Steps (all pure Python / PyTorch):
    1. Apply test transforms (resize, crop, colour convert, normalize)
    2. Stack camera images + ego2img matrices
    3. Voxelize LiDAR points (inline, no CenterPointPreProcess)
    4. Compute LSS reference points
    """
    sample = apply_test_transforms(sample)

    img_list = sample["img"]
    ego2img_list = sample["ego2img"]

    img = torch.stack(img_list, dim=0).to(device)
    ego2img = torch.stack(
        [torch.from_numpy(m.astype(np.float32)) for m in ego2img_list], dim=0
    ).to(device)

    points_np = sample["points"]
    points_tensor = torch.from_numpy(points_np.astype(np.float32)).float().to(device)
    with torch.no_grad():
        features, coords = voxelize_lidar(
            points_list=[points_tensor],
            pc_range=POINT_CLOUD_RANGE,
            voxel_size=VOXEL_SIZE,
            max_points_in_voxel=MAX_POINTS_IN_VOXEL,
            max_voxels=MAX_VOXELS_VAL,
            norm_range=NORM_RANGE,
            norm_dims=NORM_DIMS,
            is_deploy=True,
        )

    vt_input_hw = cfg.vt_input_hw
    ref_points = deploy_model.camera_net.export_reference_points(
        {"img": img, "ego2img": ego2img}, feat_wh=tuple(vt_input_hw)
    )

    return {
        "features": features,
        "coors": coords.to(torch.int32),
        "img": img,
        "ego2img": ego2img,
        "points0": ref_points["points0"].to(device),
        "points1": ref_points["points1"].to(device),
    }


def run_inference(deploy_model, deploy_data: dict) -> torch.Tensor:
    with torch.no_grad():
        results = deploy_model(deploy_data, compile_model=True)

    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    if isinstance(result, (list, tuple)):
        occ_preds = result[0]
    else:
        occ_preds = result

    if isinstance(occ_preds, (list, tuple)):
        occ_preds = occ_preds[0]

    return occ_preds.squeeze()


class _MeanIOUState:


    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.intersect = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)
        self.pred_label = torch.zeros(num_classes)
        self.label = torch.zeros(num_classes)

    def update(self, label: Tensor, preds: Tensor):
        mask = label != IGNORE_INDEX
        pred_label = preds[mask].float()
        label_f = label[mask].float()

        intersect = pred_label[pred_label == label_f]

        area_intersect = torch.histc(
            intersect, bins=self.num_classes, max=self.num_classes - 1
        )
        area_pred_label = torch.histc(
            pred_label, bins=self.num_classes, max=self.num_classes - 1
        )
        area_label = torch.histc(
            label_f, bins=self.num_classes, max=self.num_classes - 1
        )
        area_union = area_pred_label + area_label - area_intersect

        self.intersect += area_intersect
        self.union += area_union
        self.pred_label += area_pred_label
        self.label += area_label

    def get_states(self):
        return (
            self.intersect.clone(),
            self.union.clone(),
            self.pred_label.clone(),
            self.label.clone(),
        )


class _Occ3dMeanIOUState:

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.float64
        )

    def update(self, label: Tensor, preds: Tensor):
        n_cl = self.num_classes
        label = label.reshape(-1).long()
        pred_label = preds.reshape(-1).long()
        valid_mask = (label >= 0) & (label < n_cl)
        label_valid = label[valid_mask]
        pred_valid = pred_label[valid_mask]
        indices = n_cl * label_valid + pred_valid
        batch_hist = torch.bincount(
            indices, minlength=n_cl * n_cl
        ).reshape(n_cl, n_cl).to(dtype=torch.float64)
        self.confusion_matrix += batch_hist

    def get_states(self):
        return (self.confusion_matrix.clone(),)


def update_metric(metric: _MeanIOUState, occ_pred: Tensor, sample: dict):
    
    gt_info = sample["gt_occ_info"]
    gt_semantics = gt_info["voxel_semantics"]
    camera_mask = gt_info["mask_camera"]

    camera_mask_t = torch.from_numpy(camera_mask.copy()).to(torch.bool)
    masked_gt = torch.from_numpy(gt_semantics.copy())[camera_mask_t].reshape(-1).long()
    masked_pred = occ_pred[camera_mask_t].reshape(-1).long().cpu()

    metric.update(label=masked_gt, preds=masked_pred)


def _occ2img(semantics):
    H, W, D = semantics.shape
    free_id = 17
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id
    for i in range(D):
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]
    viz = OCC_COLOR_MAP[semantics_2d][..., :3]
    viz = cv2.resize(viz, dsize=(800, 800), interpolation=cv2.INTER_NEAREST)
    viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
    return viz


def _quantize_to_bytes(tensor, scale, dtype_str):
    arr = tensor.cpu().numpy()
    if dtype_str == "s8":
        q = np.round(arr.astype(np.float64) / scale)
        q = np.clip(q, -128, 127).astype(np.int8)
        return q.tobytes()
    elif dtype_str == "s16":
        q = np.round(arr.astype(np.float64) / scale)
        q = np.clip(q, -32768, 32767).astype(np.int16)
        return q.tobytes()
    elif dtype_str == "s32":
        return arr.astype(np.int32).tobytes()
    elif dtype_str == "f32":
        return arr.astype(np.float32).tobytes()
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _get_display_images(sample):
    return np.stack([np.array(img) for img in sample["img"]], axis=0)


def _save_sample_outputs(output_dir, sample_idx, token, deploy_data, occ_pred, display_imgs):
    stem = f"sample_{sample_idx:06d}_{token[:16]}"
    out_dir = os.path.join(output_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    qin_dir = os.path.join(out_dir, "quant_inputs")
    os.makedirs(qin_dir, exist_ok=True)

    features = deploy_data["features"]
    n_voxels = features.shape[-1]
    if n_voxels < HBM_VOXELS:
        pad = torch.zeros(1, 5, 20, HBM_VOXELS - n_voxels, dtype=features.dtype, device=features.device)
        features = torch.cat([features, pad], dim=-1)
    elif n_voxels > HBM_VOXELS:
        features = features[..., :HBM_VOXELS]
    qp = HBM_QUANT_PARAMS["features"]
    with open(os.path.join(qin_dir, "features.bin"), "wb") as f:
        f.write(_quantize_to_bytes(features, qp["scale"], qp["dtype"]))

    coors = deploy_data["coors"]
    if coors.shape[0] < HBM_VOXELS:
        pad = torch.zeros(HBM_VOXELS - coors.shape[0], 4, dtype=coors.dtype, device=coors.device)
        coors = torch.cat([coors, pad], dim=0)
    elif coors.shape[0] > HBM_VOXELS:
        coors = coors[:HBM_VOXELS]
    qp = HBM_QUANT_PARAMS["coors"]
    with open(os.path.join(qin_dir, "coors.bin"), "wb") as f:
        f.write(_quantize_to_bytes(coors, qp["scale"], qp["dtype"]))

    for key in ["img", "ego2img", "points0", "points1"]:
        qp = HBM_QUANT_PARAMS[key]
        with open(os.path.join(qin_dir, f"{key}.bin"), "wb") as f:
            f.write(_quantize_to_bytes(deploy_data[key], qp["scale"], qp["dtype"]))

    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for cam_idx, cam_name in enumerate(CAM_NAMES):
        img_bgr = display_imgs[cam_idx][..., ::-1]
        cv2.imwrite(os.path.join(vis_dir, f"{cam_name}.png"), img_bgr)
    occ_img = _occ2img(occ_pred.cpu().numpy().astype(np.int32))
    cv2.imwrite(os.path.join(vis_dir, "occ_topdown.png"), occ_img)


def _save_tensor_ele(x: Tensor, save_index_list: list) -> Tensor:
    _save_tensor = [x[index: index + 1] for index in save_index_list]
    return torch.cat(_save_tensor, dim=0)


def _tensor_nan_mean(x: Tensor) -> Tensor:
    tmp_value = x.cpu().numpy()
    _mean_val = np.nanmean(tmp_value)
    return x.new_tensor(_mean_val)


def aggregate_metric_states(
    states_list: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
    seg_class: List[str],
    ignore_index: int,
):
    
    num_classes = len(seg_class)
    global_save_index_list = [
        i for i in range(num_classes) if i != ignore_index
    ]

    intersect = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    pred_label = torch.zeros(num_classes)
    label = torch.zeros(num_classes)

    for s in states_list:
        intersect += s[0].cpu()
        union += s[1].cpu()
        pred_label += s[2].cpu()
        label += s[3].cpu()

    all_acc = (
        _save_tensor_ele(intersect, global_save_index_list).sum()
        / _save_tensor_ele(label, global_save_index_list).sum()
    )
    acc = intersect / label
    iou = intersect / union

    summary_str = "~~~~ MeanIOU Summary metrics ~~~~\n"
    summary_str += "Summary:\n"
    line_format = "{:<15} {:>10} {:>10} {:>10}\n"
    summary_str += line_format.format("Scope", "mIoU", "mAcc", "aAcc")

    miou = _tensor_nan_mean(_save_tensor_ele(iou, global_save_index_list))
    macc = _tensor_nan_mean(_save_tensor_ele(acc, global_save_index_list))
    summary_str += line_format.format(
        "global",
        "{:.2f}".format(miou.cpu().item() * 100),
        "{:.2f}".format(macc.cpu().item() * 100),
        "{:.2f}".format(all_acc * 100),
    )

    summary_str += "Per Class Results:\n"
    line_format = "{:<15} {:>10} {:>10}\n"
    summary_str += line_format.format("Class", "IoU", "Acc")

    for i in range(num_classes):
        summary_str += line_format.format(
            seg_class[i],
            "{:.2f}".format(iou[i].cpu().item() * 100),
            "{:.2f}".format(acc[i].cpu().item() * 100),
        )
    logger.info(summary_str)

    return miou.cpu().item() * 100, iou


def aggregate_occ3d_states(states_list, seg_class, exclude_classes):
    num_classes = len(seg_class)
    include_indices = [i for i in range(num_classes) if i not in exclude_classes]

    hist = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for s in states_list:
        hist += s[0].cpu()

    hist = hist.float()
    intersect = torch.diag(hist)
    gt_count = hist.sum(dim=1)
    pred_count = hist.sum(dim=0)
    union = gt_count + pred_count - intersect

    iou = intersect / union
    iou[union == 0] = float("nan")
    acc = intersect / gt_count
    acc[gt_count == 0] = float("nan")

    inc = include_indices
    iou_sel = iou[inc]
    acc_sel = acc[inc]

    miou_vec = iou_sel.numpy()
    macc_vec = acc_sel.numpy()
    miou_val = np.nanmean(miou_vec)
    macc_val = np.nanmean(macc_vec)
    all_acc_val = (intersect[inc].sum() / (gt_count[inc].sum() + 1e-8)).item()

    summary_str = "~~~~ Occ3dMeanIOU Summary metrics ~~~~\n"
    summary_str += "Summary:\n"
    line_format = "{:<15} {:>10} {:>10} {:>10}\n"
    summary_str += line_format.format("Scope", "mIoU", "mAcc", "aAcc")
    summary_str += line_format.format(
        "global",
        "{:.2f}".format(miou_val * 100),
        "{:.2f}".format(macc_val * 100),
        "{:.2f}".format(all_acc_val * 100),
    )

    summary_str += "Per Class Results:\n"
    line_format = "{:<15} {:>10} {:>10}\n"
    summary_str += line_format.format("Class", "IoU", "Acc")
    for i in range(num_classes):
        iou_str = "{:.2f}".format(iou[i].cpu().item() * 100)
        acc_str = "{:.2f}".format(acc[i].cpu().item() * 100)
        summary_str += line_format.format(seg_class[i], iou_str, acc_str)
    logger.info(summary_str)

    return miou_val * 100, iou


def evaluate_single_gpu(
    config_path: str,
    ckpt_dir: str,
    checkpoint_name: str,
    device: str,
    max_samples: int,
    indices: List[int],
    pbar_position: int = 0,
    output_dir: str = None,
    metric_type: str = "histc",
):

    logging.getLogger("hat").setLevel(logging.WARNING)
    logging.getLogger("horizon_plugin_pytorch").setLevel(logging.WARNING)

    cfg = load_config_vars(config_path)
    deploy_model = build_deploy_model_with_calib_weights(
        cfg, ckpt_dir, checkpoint_name, device
    )

    with open(VAL_PKL, "rb") as f:
        pkl_data = pickle.load(f)
    all_infos = pkl_data["infos"]

    if metric_type == "occ3d":
        metric = _Occ3dMeanIOUState(num_classes=len(OCC3D_SEG_CLASS_FULL))
    else:
        metric = _MeanIOUState(num_classes=NUM_CLASSES)

    desc = f"GPU {device}"
    for idx in tqdm(indices, desc=desc, position=pbar_position, leave=False):
        info = all_infos[idx]
        sample = load_sample(idx, info)
        if output_dir is not None:
            display_imgs = _get_display_images(sample)
        deploy_data = prepare_deploy_inputs(sample, deploy_model, cfg, device)
        occ_pred = run_inference(deploy_model, deploy_data)
        update_metric(metric, occ_pred, sample)
        if output_dir is not None:
            _save_sample_outputs(output_dir, idx, info.get("token", ""), deploy_data, occ_pred, display_imgs)
        del deploy_data, occ_pred

    return metric.get_states()


def _worker_entry(
    rank: int,
    gpu_id: int,
    config_path: str,
    ckpt_dir: str,
    checkpoint_name: str,
    max_samples: int,
    indices: List[int],
    result_queue: mp.Queue,
    output_dir: str = None,
    metric_type: str = "histc",
):

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    try:
        states = evaluate_single_gpu(
            config_path=config_path,
            ckpt_dir=ckpt_dir,
            checkpoint_name=checkpoint_name,
            device=device,
            max_samples=max_samples,
            indices=indices,
            pbar_position=rank,
            output_dir=output_dir,
            metric_type=metric_type,
        )
        np_states = tuple(s.cpu().numpy().copy() for s in states)
        result_queue.put((rank, np_states))
    except Exception as e:
        logger.error("Worker %d (GPU %d) failed: %s", rank, gpu_id, e, exc_info=True)
        result_queue.put((rank, None))


def evaluate_multi_gpu(
    gpu_ids: List[int],
    config_path: str,
    ckpt_dir: str,
    checkpoint_name: str,
    max_samples: int,
    output_dir: str = None,
    metric_type: str = "histc",
):

    world_size = len(gpu_ids)

    with open(VAL_PKL, "rb") as f:
        pkl_data = pickle.load(f)
    total_samples = len(pkl_data["infos"])
    max_samples = max_samples if max_samples > 0 else total_samples

    logger.info("Total samples: %d  |  GPUs: %d  |  Per GPU: ~%d",
                max_samples, world_size,
                (max_samples + world_size - 1) // world_size)

    all_indices = list(range(max_samples))
    shards = []
    for r in range(world_size):
        shards.append(all_indices[r::world_size])

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_entry,
            args=(
                rank, gpu_id, config_path, ckpt_dir, checkpoint_name,
                max_samples, shards[rank], result_queue, output_dir, metric_type,
            ),
        )
        p.start()
        processes.append(p)

    states_list = [None] * world_size
    for _ in range(world_size):
        rank, np_states = result_queue.get()
        if np_states is None:
            raise RuntimeError(f"Worker {rank} failed — check logs above.")
        states_list[rank] = tuple(torch.from_numpy(a) for a in np_states)

    for p in processes:
        p.join()

    if metric_type == "occ3d":
        miou_val, iou = aggregate_occ3d_states(
            states_list, OCC3D_SEG_CLASS_FULL, [17]
        )
        seg_class_display = OCC3D_SEG_CLASS_FULL
    else:
        miou_val, iou = aggregate_metric_states(
            states_list, OCC3D_SEG_CLASS, IGNORE_INDEX
        )
        seg_class_display = OCC3D_SEG_CLASS

    logger.info("=" * 60)
    logger.info("Calibrated Deploy Model mIoU (multi-GPU): %.4f%%", miou_val)
    logger.info("=" * 60)
    logger.info("Per-class IoU:")
    for i, cls_name in enumerate(seg_class_display):
        logger.info("  %-20s: %.4f", cls_name, iou[i].cpu().item() * 100)

    return miou_val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify calibrated deploy model mIoU — raw data, no registry"
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to config file (e.g. ../configs/occ/flashocc_xxx.py)",
    )
    parser.add_argument(
        "--ckpt-dir", "-d", type=str, required=True,
        help="Directory containing calibration checkpoint",
    )
    parser.add_argument(
        "--checkpoint", "-p", type=str, default="calibration-checkpoint-best.pth.tar",
        help="Checkpoint filename inside ckpt-dir",
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=0,
        help="Max samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help='Comma-separated GPU indices (e.g. "0,1,2,3")',
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Single-device override (single-GPU mode only)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="If set, save quantized HBM inputs and occ visualization per sample",
    )
    parser.add_argument(
        "--metric", type=str, default="histc", choices=["histc", "occ3d"],
        help="Metric type: histc (17-bin torch.histc) or occ3d (18x18 confusion matrix)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
    ckpt_dir = os.path.join(script_dir, args.ckpt_dir) if not os.path.isabs(args.ckpt_dir) else args.ckpt_dir

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    logger.info("Config:       %s", config_path)
    logger.info("Checkpoint:   %s/%s", ckpt_dir, args.checkpoint)
    logger.info("GPUs:         %s", gpu_ids)
    logger.info("Max samples:  %s", args.max_samples if args.max_samples > 0 else "all")
    logger.info("Output dir:   %s", args.output_dir or "(none)")
    logger.info("Metric:       %s", args.metric)

    if len(gpu_ids) == 1:
        device = args.device or f"cuda:{gpu_ids[0]}"

        with open(VAL_PKL, "rb") as f:
            pkl_data = pickle.load(f)
        total = len(pkl_data["infos"])
        max_samples = args.max_samples if args.max_samples > 0 else total

        states = evaluate_single_gpu(
            config_path=config_path,
            ckpt_dir=ckpt_dir,
            checkpoint_name=args.checkpoint,
            device=device,
            max_samples=max_samples,
            indices=list(range(max_samples)),
            pbar_position=0,
            output_dir=args.output_dir,
            metric_type=args.metric,
        )
        if args.metric == "occ3d":
            miou_val, iou = aggregate_occ3d_states(
                [states], OCC3D_SEG_CLASS_FULL, [17]
            )
            seg_class_display = OCC3D_SEG_CLASS_FULL
        else:
            miou_val, iou = aggregate_metric_states(
                [states], OCC3D_SEG_CLASS, IGNORE_INDEX
            )
            seg_class_display = OCC3D_SEG_CLASS

        logger.info("=" * 60)
        logger.info("Calibrated Deploy Model mIoU (single-GPU): %.4f%%", miou_val)
        logger.info("=" * 60)
        logger.info("Per-class IoU:")
        for i, cls_name in enumerate(seg_class_display):
            logger.info("  %-20s: %.4f", cls_name, iou[i].cpu().item() * 100)
    else:
        miou_val = evaluate_multi_gpu(
            gpu_ids=gpu_ids,
            config_path=config_path,
            ckpt_dir=ckpt_dir,
            checkpoint_name=args.checkpoint,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            metric_type=args.metric,
        )

    # logger.info("")
    # logger.info("To compare with calibration predictor, run:")
    # logger.info(
    #     "  cd /data01/chenmu/bev_lss_v2/release_package/scripts/tools && "
    #     "python3 predict.py -c %s -s calibration", args.config
    # )

    return miou_val


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()