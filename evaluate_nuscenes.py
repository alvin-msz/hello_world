"""Standalone evaluation tool for nuscenes results."""
import argparse
import glob
import json
import logging
import os
import pickle

import numpy as np

from hat.utils.config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

'''
Usage:
python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/evaluate_nuscenes.py \
  --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes.py \
  --results-file ./tmp_eval/bevfusion_qat/meta/results_nusc.json \
  --data-root /open_explorer/data/nuscenes \
  --output-dir ./tmp_eval/bevfusion_qat/meta \
  --occ-dir ./tmp_eval/bevfusion_qat/meta/occ_preds
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate nuscenes results")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="config file path")
    parser.add_argument("--results-file", type=str, default=None,
                        help="path to results_nusc.json (required for mAP evaluation)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="nuscenes dataset root directory (required for mAP evaluation)")
    parser.add_argument("--output-dir", type=str, default="./metric_results",
                        help="directory to save evaluation results")
    parser.add_argument("--eval-map", action="store_true", default=True,
                        help="evaluate mAP (detection)")
    parser.add_argument("--eval-miou", action="store_true", default=True,
                        help="evaluate mIOU (segmentation)")
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        help="nuscenes dataset version")
    parser.add_argument(
        "--occ-dir", type=str, default=None,
        help="directory containing per-frame OCC pkl files (occ_rank*.pkl). "
             "Each pkl has keys 'gt' and 'pred' as flattened numpy arrays.",
    )
    parser.add_argument(
        "--ignore-index", type=int, default=None,
        help="ignore index for OCC evaluation (overrides config). "
             "Typically 17 for occ3d-nuscenes.",
    )
    parser.add_argument(
        "--frames-lst", type=str, default=None,
        help="path to frames.lst that defines the order of bin files. "
             "Each line is a path like .../sample_XXXX_<token_prefix>. "
             "Required for bin-mode OCC evaluation to align pred/GT order.",
    )
    parser.add_argument(
        "--mask-type", type=str, default="camera",
        choices=["camera", "lidar", "union"],
        help="Visibility mask to apply in bin-mode OCC evaluation. "
             "'camera' (default): mask_camera only -- matches camera-only models "
             "(FlashOcc, BEVFormer). "
             "'lidar': mask_lidar only -- matches lidar-only models. "
             "'union': mask_lidar | mask_camera -- matches fusion models (BEVFusion).",
    )
    parser.add_argument(
        "--pred-dtype", type=str, default="int16",
        choices=["int16", "uint8", "int8", "int32"],
        help="NumPy dtype used to read the bin pred file (default: int16). "
             "Use uint8 if the BPU model outputs uint8 argmax.",
    )
    return parser.parse_args()


def evaluate_map(results_file, data_root, version, output_dir, cfg):
    """Evaluate detection mAP using NuscenesMetric."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    logger.info("Evaluating detection mAP...")

    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    eval_cfg = config_factory("detection_cvpr_2019")
    nusc_eval = DetectionEval(
        nusc,
        config=eval_cfg,
        result_path=results_file,
        eval_set="val",
        output_dir=output_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main()

    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info(f"mAP: {metrics_summary['mean_ap']:.4f}")
    logger.info(f"NDS: {metrics_summary['nd_score']:.4f}")
    return metrics_summary


def _find_ignore_index_in_cfg(cfg):
    """Recursively search config for ignore_index in metrics dicts."""
    # Check top-level
    ignore_index = getattr(cfg, "ignore_index", None)
    if ignore_index is not None:
        return ignore_index

    # Check occ_val_nuscenes_metric
    occ_val_metric = getattr(cfg, "occ_val_nuscenes_metric", None)
    if occ_val_metric and isinstance(occ_val_metric, dict):
        if "ignore_index" in occ_val_metric:
            return occ_val_metric["ignore_index"]

    # Check val_occ_metric
    val_occ_metric = getattr(cfg, "val_occ_metric", None)
    if val_occ_metric and isinstance(val_occ_metric, dict):
        if "ignore_index" in val_occ_metric:
            return val_occ_metric["ignore_index"]

    # Search through predictor configs (float_predictor, qat_predictor, etc.)
    for attr_name in dir(cfg):
        if "predictor" not in attr_name:
            continue
        try:
            predictor = getattr(cfg, attr_name)
        except Exception:
            continue
        if not isinstance(predictor, dict):
            continue
        metrics = predictor.get("metrics", None)
        if isinstance(metrics, dict) and "ignore_index" in metrics:
            return metrics["ignore_index"]

    # Check bpu_eval_metric (tuple of metrics)
    bpu_eval_metric = getattr(cfg, "bpu_eval_metric", None)
    if bpu_eval_metric and isinstance(bpu_eval_metric, (tuple, list)):
        for m in bpu_eval_metric:
            if hasattr(m, "ignore_index"):
                return m.ignore_index
            if isinstance(m, dict) and "ignore_index" in m:
                return m["ignore_index"]

    # Search val_metrics list
    val_metrics = getattr(cfg, "val_metrics", None)
    if val_metrics and isinstance(val_metrics, (tuple, list)):
        for m in val_metrics:
            if isinstance(m, dict) and "ignore_index" in m:
                return m["ignore_index"]

    return None


def evaluate_miou(occ_dir, output_dir, cfg, override_ignore_index=None,
                  data_root=None, version="v1.0-trainval", frames_lst=None,
                  mask_type="camera", pred_dtype="int16"):
    """Evaluate segmentation mIOU from per-frame OCC pkl or bin directory.

    pkl mode: each file has dict with 'gt' and 'pred'.
    bin mode: each file has flattened argmax pred;
              GT is loaded from nuscenes dataset via data_root.
              frames_lst defines the order of bin files.

    mask_type: controls which visibility mask to apply in bin mode.
        'camera' (default) -- mask_camera only (camera models: FlashOcc, BEVFormer)
        'lidar'            -- mask_lidar only (lidar-only models)
        'union'            -- mask_lidar | mask_camera (fusion models: BEVFusion)
    """
    logger.info(f"Evaluating OCC mIOU from directory: {occ_dir}")

    occ_pkl_files = sorted(glob.glob(os.path.join(occ_dir, "occ_rank*.pkl")))
    occ_bin_files = sorted(glob.glob(os.path.join(occ_dir, "occ_rank*.bin")))

    use_bin = False
    if occ_pkl_files:
        occ_files = occ_pkl_files
        logger.info(f"Found {len(occ_files)} OCC pkl files")
    elif occ_bin_files:
        occ_files = occ_bin_files
        use_bin = True
        logger.info(f"Found {len(occ_files)} OCC bin files")
    else:
        logger.warning(f"No occ_rank*.pkl or occ_rank*.bin files found in {occ_dir}")
        return None

    # --- Load GT for bin mode from nuscenes dataset ---
    gt_list = None
    mask_list = None
    if use_bin:
        if data_root is None:
            logger.error("--data-root is required for bin-mode OCC evaluation")
            return None
        if frames_lst is None:
            logger.error("--frames-lst is required for bin-mode OCC evaluation "
                         "to align pred/GT order")
            return None
        result = _load_occ_gt_by_frames_lst(frames_lst, data_root, version,
                                             mask_type=mask_type)
        if result is None:
            logger.error("Failed to load OCC ground truth from nuscenes dataset")
            return None
        gt_list, mask_list = result
        if len(gt_list) == 0:
            logger.error("Failed to load OCC ground truth from nuscenes dataset")
            return None
        if len(gt_list) != len(occ_files):
            logger.warning(
                f"GT count ({len(gt_list)}) != bin file count ({len(occ_files)}). "
                f"Will use min({len(gt_list)}, {len(occ_files)}) samples."
            )

    seg_classes_name = getattr(cfg, "seg_classes_name", None)
    if seg_classes_name is None:
        seg_classes_name = getattr(cfg, "seg_class_names", None)
    if seg_classes_name is None:
        seg_classes_name = getattr(cfg, "occ3d_seg_class", None)
    if seg_classes_name is None:
        # Default occ3d-nuscenes 17 semantic classes + free
        seg_classes_name = [
            "others", "barrier", "bicycle", "bus", "car",
            "construction_vehicle", "motorcycle", "pedestrian",
            "traffic_cone", "trailer", "truck", "driveable_surface",
            "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
            "free",
        ]
        logger.info("Using default occ3d-nuscenes class names (18 classes)")

    if override_ignore_index is not None:
        ignore_index = override_ignore_index
    else:
        ignore_index = _find_ignore_index_in_cfg(cfg)
        if ignore_index is None:
            ignore_index = -1
    logger.info(f"Using ignore_index={ignore_index}")

    if seg_classes_name is not None:
        num_classes = len(seg_classes_name)
        if ignore_index >= num_classes:
            num_classes = ignore_index + 1
    else:
        num_classes = None
    confusion_matrix = None
    loaded = 0

    for idx, fpath in enumerate(occ_files):
        try:
            if use_bin:
                np_dtype = np.dtype(pred_dtype)
                pred = np.fromfile(fpath, dtype=np_dtype).astype(np.int64)
                if gt_list is not None and idx < len(gt_list):
                    gt = gt_list[idx].flatten().astype(np.int64)
                else:
                    continue
                # Skip empty GT (from "not found" tokens)
                if gt.size == 0:
                    logger.warning(f"Skipping {fpath}: GT is empty for index {idx}")
                    continue
                # Apply visibility mask (type controlled by --mask-type)
                vis_mask = mask_list[idx] if mask_list is not None and idx < len(mask_list) else None
                # --- First-frame diagnostics ---
                if idx == 0:
                    logger.info(
                        f"[Diag frame 0] pred.size={pred.size} dtype={np_dtype}, "
                        f"gt.size={gt.size}, "
                        f"mask={'None' if vis_mask is None else f'size={vis_mask.size} True={vis_mask.sum()}'}"
                    )
                    logger.info(
                        f"[Diag frame 0] pred[:8]={pred[:8].tolist()}  "
                        f"gt[:8]={gt[:8].tolist()}"
                    )
                    if vis_mask is not None and vis_mask.size == pred.size == gt.size:
                        p_masked = pred[vis_mask]
                        g_masked = gt[vis_mask]
                        logger.info(
                            f"[Diag frame 0] after mask -> pred[:8]={p_masked[:8].tolist()}  "
                            f"gt[:8]={g_masked[:8].tolist()}"
                        )
                        gt_cls, gt_cnt = np.unique(g_masked, return_counts=True)
                        pd_cls, pd_cnt = np.unique(p_masked, return_counts=True)
                        logger.info(f"[Diag frame 0] gt class dist: { {int(c): int(n) for c, n in zip(gt_cls[:10], gt_cnt[:10])} }")
                        logger.info(f"[Diag frame 0] pred class dist: { {int(c): int(n) for c, n in zip(pd_cls[:10], pd_cnt[:10])} }")
                if vis_mask is not None:
                    if vis_mask.size == pred.size == gt.size:
                        pred = pred[vis_mask]
                        gt = gt[vis_mask]
                    else:
                        logger.warning(
                            f"Mask size ({vis_mask.size}) != pred/gt size "
                            f"({pred.size}/{gt.size}) at index {idx}, skipping mask"
                        )
                # Validate size match
                if pred.size != gt.size:
                    logger.warning(
                        f"Skipping {fpath}: pred size ({pred.size}) != "
                        f"gt size ({gt.size}) at index {idx}"
                    )
                    continue
            else:
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                pred = np.array(data["pred"])
                gt = np.array(data["gt"])
        except Exception as e:
            logger.warning(f"Skipping corrupted OCC file {fpath}: {e}")
            continue

        valid = gt != ignore_index
        pred = pred[valid]
        gt = gt[valid]

        if pred.size == 0:
            continue

        if num_classes is None:
            num_classes = int(max(pred.max(), gt.max())) + 1

        if confusion_matrix is None:
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        pred = np.clip(pred, 0, num_classes - 1)
        gt = np.clip(gt, 0, num_classes - 1)

        np.add.at(confusion_matrix, (gt, pred), 1)
        loaded += 1

    if confusion_matrix is None:
        logger.warning("No valid OCC samples found.")
        return None

    logger.info(f"Loaded {loaded} valid OCC frames")

    intersection = np.diag(confusion_matrix)
    union = (
        confusion_matrix.sum(axis=1)
        + confusion_matrix.sum(axis=0)
        - intersection
    )
    iou = intersection / np.maximum(union, 1).astype(np.float64)

    valid_classes = list(range(num_classes))
    if 0 <= ignore_index < num_classes:
        valid_classes = [c for c in valid_classes if c != ignore_index]
    valid_iou = iou[valid_classes]
    miou = float(np.nanmean(valid_iou))

    class_labels = (
        list(seg_classes_name) if seg_classes_name
        else [str(i) for i in range(num_classes)]
    )

    miou_results = {
        "mIOU": miou,
        "num_frames": loaded,
        "per_class_iou": {
            name: float(iou[i]) for i, name in enumerate(class_labels)
            if i in valid_classes
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "miou_results.json")
    with open(out_path, "w") as f:
        json.dump(miou_results, f, indent=2)

    logger.info(f"mIOU: {miou:.4f}  (over {loaded} frames)")
    for name, val in miou_results["per_class_iou"].items():
        logger.info(f"  {name}: {val:.4f}")

    return miou_results


def _load_occ_gt_by_frames_lst(frames_lst, data_root, version, mask_type="camera"):
    """Load OCC GT ordered by frames.lst token prefixes.

    Each line in frames.lst looks like:
        /media/.../sample_XXXX_<token_prefix>
    where <token_prefix> is the first 8 chars of the nuscenes sample token.

    mask_type: controls which mask(s) to load from labels.npz.
        'camera' (default) -- mask_camera only  (FlashOcc, BEVFormer)
        'lidar'            -- mask_lidar only
        'union'            -- mask_lidar | mask_camera  (BEVFusion multisensor)

    Returns:
        (gt_list, mask_list): gt is flattened full-grid array;
        mask is a boolean array selecting the visible voxels, or None.
    """
    # Parse token prefixes from frames.lst
    with open(frames_lst, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    logger.info(f"Read {len(lines)} entries from {frames_lst}")

    token_prefixes = []
    for line in lines:
        basename = os.path.basename(line)
        # e.g. "sample_0000_30e55a3e" -> token prefix "30e55a3e"
        parts = basename.split("_")
        if len(parts) >= 3:
            token_prefixes.append(parts[-1])
        else:
            token_prefixes.append(basename)

    # Load GT files from occ3d gts/ directory
    occ_gt_dir = os.path.join(data_root, "gts")
    if not os.path.isdir(occ_gt_dir):
        occ_gt_dir = os.path.join(data_root, "occ3d", "gts")
    if not os.path.isdir(occ_gt_dir):
        logger.error(f"Cannot find OCC GT directory at {data_root}/gts or "
                     f"{data_root}/occ3d/gts")
        return None

    # Build lookup: token (or token prefix) -> gt directory path
    # Structure: gts/<scene_dir>/<token>/labels.npz
    gt_dir_lookup = {}
    for scene_dir in os.listdir(occ_gt_dir):
        scene_path = os.path.join(occ_gt_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue
        for token_dir in os.listdir(scene_path):
            token_path = os.path.join(scene_path, token_dir)
            gt_file = os.path.join(token_path, "labels.npz")
            if os.path.exists(gt_file):
                gt_dir_lookup[token_dir] = token_path
    logger.info(f"Found {len(gt_dir_lookup)} GT files in {occ_gt_dir}")

    # Build prefix -> full token mapping
    prefix_to_full = {}
    for full_token in gt_dir_lookup:
        for plen in (8, 10, 12, len(full_token)):
            prefix = full_token[:plen]
            if prefix not in prefix_to_full:
                prefix_to_full[prefix] = full_token

    gt_list = []
    mask_list = []
    not_found = 0
    mask_found_count = 0
    for prefix in token_prefixes:
        # Try exact match first, then prefix match
        full_token = prefix_to_full.get(prefix, None)
        if full_token is None:
            if prefix in gt_dir_lookup:
                full_token = prefix
        if full_token is None:
            not_found += 1
            if not_found <= 5:
                logger.warning(f"OCC GT not found for token prefix {prefix}")
            gt_list.append(np.array([], dtype=np.int64))
            mask_list.append(None)
            continue
        token_path = gt_dir_lookup[full_token]
        gt_path = os.path.join(token_path, "labels.npz")
        try:
            gt_data = np.load(gt_path)
            if "semantics" in gt_data:
                gt = gt_data["semantics"]
            elif "labels" in gt_data:
                gt = gt_data["labels"]
            else:
                gt = list(gt_data.values())[0]
            if gt.ndim == 3:
                gt = gt.transpose(0, 1, 2)

            # Load visibility masks from labels.npz.
            # Which mask(s) to use is determined by mask_type, matching
            # the training pipeline's update_metric logic:
            #   camera-only model  -> mask_camera only   (FlashOcc, BEVFormer)
            #   lidar-only model   -> mask_lidar only
            #   fusion model       -> mask_lidar | mask_camera  (BEVFusion)
            mask_camera = None
            mask_lidar = None
            if "mask_camera" in gt_data:
                m = gt_data["mask_camera"]
                if m.ndim == 3:
                    m = m.transpose(0, 1, 2)
                mask_camera = m.flatten().astype(bool)
            if "mask_lidar" in gt_data:
                m = gt_data["mask_lidar"]
                if m.ndim == 3:
                    m = m.transpose(0, 1, 2)
                mask_lidar = m.flatten().astype(bool)

            if mask_type == "camera":
                mask = mask_camera
            elif mask_type == "lidar":
                mask = mask_lidar
            elif mask_type == "union":
                if mask_camera is not None and mask_lidar is not None:
                    mask = mask_lidar | mask_camera
                else:
                    mask = mask_camera if mask_camera is not None else mask_lidar
            else:
                mask = mask_camera

            if mask is not None:
                mask_found_count += 1

            gt_list.append(gt.flatten())
            mask_list.append(mask)
        except Exception as e:
            logger.warning(f"Failed to load GT {gt_path}: {e}")
            gt_list.append(np.array([], dtype=np.int64))
            mask_list.append(None)

    if not_found > 5:
        logger.warning(f"Total {not_found} GT files not found (suppressed after 5)")
    logger.info(f"Loaded {len(gt_list)} OCC GT samples by frames.lst order "
                f"({mask_found_count} with camera mask)")
    return gt_list, mask_list


def _load_occ_gt_from_nuscenes(data_root, version, cfg):
    """Load OCC ground truth from nuscenes dataset.

    Attempts to use the config's val dataset to load GT occ labels.
    Falls back to loading from standard occ3d-nuscenes GT path.
    """
    # Try standard occ3d-nuscenes GT path
    occ_gt_dir = os.path.join(data_root, "gts")
    if not os.path.isdir(occ_gt_dir):
        occ_gt_dir = os.path.join(data_root, "occ3d", "gts")

    if os.path.isdir(occ_gt_dir):
        return _load_occ_gt_from_occ3d_dir(occ_gt_dir, data_root, version)

    logger.warning(
        f"Cannot find OCC GT directory at {data_root}/gts or {data_root}/occ3d/gts. "
        "Please ensure occ3d-nuscenes GT data is available."
    )
    return None


def _load_occ_gt_from_occ3d_dir(occ_gt_dir, data_root, version):
    """Load OCC GT from occ3d-nuscenes gts/ directory, ordered by val split."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    logger.info(f"Loading OCC GT from {occ_gt_dir}")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    splits = create_splits_scenes()
    val_scenes = splits.get("val", [])
    val_scene_tokens = set()
    scene_name_by_token = {}
    for scene in nusc.scene:
        if scene["name"] in val_scenes:
            val_scene_tokens.add(scene["token"])
            scene_name_by_token[scene["token"]] = scene["name"]

    # Build a mapping from sample token to scene name for path construction
    sample_to_scene_name = {}
    for sample in nusc.sample:
        if sample["scene_token"] in val_scene_tokens:
            sample_to_scene_name[sample["token"]] = scene_name_by_token[sample["scene_token"]]

    # Build a lookup of all available GT files for faster searching
    # Structure: gts/scene-XXXX/token/labels.npz
    gt_file_lookup = {}
    for scene_dir in os.listdir(occ_gt_dir):
        scene_path = os.path.join(occ_gt_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue
        for token_dir in os.listdir(scene_path):
            gt_file = os.path.join(scene_path, token_dir, "labels.npz")
            if os.path.exists(gt_file):
                gt_file_lookup[token_dir] = gt_file
    logger.info(f"Found {len(gt_file_lookup)} GT files in {occ_gt_dir}")

    # Collect val sample tokens in order
    val_sample_tokens = []
    for sample in nusc.sample:
        if sample["scene_token"] in val_scene_tokens:
            val_sample_tokens.append(sample["token"])

    gt_list = []
    not_found = 0
    for token in val_sample_tokens:
        gt_path = gt_file_lookup.get(token, None)
        if gt_path is None:
            not_found += 1
            if not_found <= 5:
                logger.warning(f"OCC GT not found for token {token}")
            gt_list.append(np.array([], dtype=np.int64))
            continue
        try:
            gt_data = np.load(gt_path)
            # occ3d-nuscenes stores as 'semantics'
            if "semantics" in gt_data:
                gt = gt_data["semantics"].flatten()
            elif "labels" in gt_data:
                gt = gt_data["labels"].flatten()
            else:
                gt = list(gt_data.values())[0].flatten()
            gt_list.append(gt)
        except Exception as e:
            logger.warning(f"Failed to load GT {gt_path}: {e}")
            gt_list.append(np.array([], dtype=np.int64))

    if not_found > 5:
        logger.warning(f"Total {not_found} GT files not found (suppressed after 5)")
    logger.info(f"Loaded {len(gt_list)} OCC GT samples from val split")
    return gt_list


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    results = {}

    if args.eval_map:
        if not args.results_file or not args.data_root:
            logger.warning(
                "--results-file and --data-root are required for mAP evaluation, "
                "skipping mAP evaluation."
            )
        else:
            map_results = evaluate_map(
                args.results_file,
                args.data_root,
                args.version,
                args.output_dir,
                cfg,
            )
            if map_results:
                results["detection"] = map_results

    if args.eval_miou:
        if not args.occ_dir:
            logger.warning(
                "--occ-dir not specified, skipping mIOU evaluation. "
                "Pass --occ-dir <path/to/occ_preds> to enable it."
            )
        else:
            miou_results = evaluate_miou(
                args.occ_dir, args.output_dir, cfg, args.ignore_index,
                data_root=args.data_root, version=args.version,
                frames_lst=args.frames_lst,
                mask_type=args.mask_type,
                pred_dtype=args.pred_dtype,
            )
            if miou_results:
                results["segmentation"] = miou_results

    logger.info(f"Evaluation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()