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
python3 samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_0528/prepare_nuscenes_mini_inputs.py \
    --nuscenes-root ./data/nuscenes  --output-dir ./samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/mini_data/flashocc_0528  \
    --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/custom/flashocc-r50-M0_bevfusionocc_horizon_2.py  \
    --version v1.0-trainval  --split val
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate nuscenes results")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="config file path")
    parser.add_argument("--results-file", type=str, required=True,
                        help="path to results_nusc.json")
    parser.add_argument("--data-root", type=str, required=True,
                        help="nuscenes dataset root directory")
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


def evaluate_miou(occ_dir, output_dir, cfg, override_ignore_index=None):
    """Evaluate segmentation mIOU from per-frame OCC pkl or bin directory.

    Supported file formats:
      - occ_rank*.pkl : dict with keys 'gt' and 'pred' (flattened numpy arrays)
      - occ_rank*.bin : raw int16 pred array (board-side output, no gt bundled)
        In bin mode, a paired gt file occ_gt*.bin or occ_rank*_gt.bin is expected,
        OR a single gt_all.bin / gt_all.pkl containing all gt frames in order.
    """
    logger.info(f"Evaluating OCC mIOU from directory: {occ_dir}")

    pkl_files = sorted(glob.glob(os.path.join(occ_dir, "occ_rank*.pkl")))
    bin_files = sorted(glob.glob(os.path.join(occ_dir, "occ_rank*.bin")))

    if pkl_files:
        logger.info(f"Found {len(pkl_files)} OCC pkl files")
        return _evaluate_miou_from_pkl(pkl_files, output_dir, cfg, override_ignore_index)
    elif bin_files:
        logger.info(f"Found {len(bin_files)} OCC bin files")
        return _evaluate_miou_from_bin(bin_files, output_dir, cfg, override_ignore_index)
    else:
        logger.warning(f"No occ_rank*.pkl or occ_rank*.bin files found in {occ_dir}")
        return None


def _get_miou_params(cfg, override_ignore_index):
    seg_classes_name = getattr(cfg, "seg_classes_name", None)
    if seg_classes_name is None:
        seg_classes_name = getattr(cfg, "seg_class_names", None)
    if seg_classes_name is None:
        seg_classes_name = getattr(cfg, "occ3d_seg_class", None)

    if override_ignore_index is not None:
        ignore_index = override_ignore_index
    else:
        ignore_index = _find_ignore_index_in_cfg(cfg)
        if ignore_index is None:
            ignore_index = -1
    logger.info(f"Using ignore_index={ignore_index}")

    num_classes = None
    if seg_classes_name is not None:
        num_classes = len(seg_classes_name)
        if ignore_index >= num_classes:
            num_classes = ignore_index + 1

    return seg_classes_name, ignore_index, num_classes


def _compute_and_save_miou(confusion_matrix, loaded, num_classes, ignore_index,
                            seg_classes_name, output_dir):
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


def _evaluate_miou_from_pkl(pkl_files, output_dir, cfg, override_ignore_index):
    seg_classes_name, ignore_index, num_classes = _get_miou_params(cfg, override_ignore_index)
    confusion_matrix = None
    loaded = 0

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"Skipping corrupted OCC file {pkl_path}: {e}")
            continue

        pred = np.array(data["pred"])
        gt = np.array(data["gt"])

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
    return _compute_and_save_miou(
        confusion_matrix, loaded, num_classes, ignore_index, seg_classes_name, output_dir
    )


def _evaluate_miou_from_bin(bin_files, output_dir, cfg, override_ignore_index):
    """Load board-side int16 pred bin files and paired gt bin/pkl files.

    GT lookup order for each occ_rank0_NNNNNN.bin:
      1. occ_gt0_NNNNNN.bin  (int16, same layout)
      2. occ_rank0_NNNNNN_gt.bin
      3. gt_all.bin  (all frames concatenated as int16, indexed by frame order)
      4. gt_all.pkl  (list of flattened arrays)
    """
    import re
    seg_classes_name, ignore_index, num_classes = _get_miou_params(cfg, override_ignore_index)
    confusion_matrix = None
    loaded = 0

    occ_dir = os.path.dirname(bin_files[0])

    # Pre-load gt_all if present
    gt_all_list = None
    gt_all_bin_path = os.path.join(occ_dir, "gt_all.bin")
    gt_all_pkl_path = os.path.join(occ_dir, "gt_all.pkl")
    if os.path.exists(gt_all_pkl_path):
        with open(gt_all_pkl_path, "rb") as f:
            gt_all_list = pickle.load(f)
        logger.info(f"Loaded gt_all.pkl with {len(gt_all_list)} entries")
    elif os.path.exists(gt_all_bin_path):
        gt_all_arr = np.fromfile(gt_all_bin_path, dtype=np.int16)
        logger.info(f"Loaded gt_all.bin with {gt_all_arr.size} int16 elements")
    else:
        gt_all_arr = None

    for frame_idx, pred_path in enumerate(bin_files):
        # --- load pred ---
        pred = np.fromfile(pred_path, dtype=np.int16).astype(np.int32)

        # --- locate gt ---
        gt = None
        basename = os.path.basename(pred_path)  # occ_rank0_000001.bin
        m = re.search(r"(\d+)\.bin$", basename)
        n_str = m.group(1) if m else None

        # try occ_gt0_NNNNNN.bin
        if n_str:
            candidate = os.path.join(occ_dir, f"occ_gt0_{n_str}.bin")
            if os.path.exists(candidate):
                gt = np.fromfile(candidate, dtype=np.int16).astype(np.int32)

        # try occ_rank0_NNNNNN_gt.bin
        if gt is None and n_str:
            candidate = os.path.join(occ_dir, basename.replace(".bin", "_gt.bin"))
            if os.path.exists(candidate):
                gt = np.fromfile(candidate, dtype=np.int16).astype(np.int32)

        # try gt_all
        if gt is None:
            if gt_all_list is not None:
                if frame_idx < len(gt_all_list):
                    gt = np.array(gt_all_list[frame_idx], dtype=np.int32).flatten()
            elif gt_all_arr is not None:
                frame_size = pred.size
                start = frame_idx * frame_size
                if start + frame_size <= gt_all_arr.size:
                    gt = gt_all_arr[start:start + frame_size].astype(np.int32)

        if gt is None:
            logger.warning(
                f"No gt found for {pred_path}, skipping. "
                "Provide occ_gt0_NNNNNN.bin, *_gt.bin, gt_all.bin, or gt_all.pkl."
            )
            continue

        valid = gt != ignore_index
        pred_v = pred[valid]
        gt_v = gt[valid]

        if pred_v.size == 0:
            continue

        if num_classes is None:
            num_classes = int(max(pred_v.max(), gt_v.max())) + 1

        if confusion_matrix is None:
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        pred_v = np.clip(pred_v, 0, num_classes - 1)
        gt_v = np.clip(gt_v, 0, num_classes - 1)
        np.add.at(confusion_matrix, (gt_v, pred_v), 1)
        loaded += 1

    if confusion_matrix is None:
        logger.warning("No valid OCC samples found from bin files.")
        return None

    logger.info(f"Loaded {loaded} valid OCC frames from bin files")
    return _compute_and_save_miou(
        confusion_matrix, loaded, num_classes, ignore_index, seg_classes_name, output_dir
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    results = {}

    if args.eval_map:
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
                args.occ_dir, args.output_dir, cfg, args.ignore_index
            )
            if miou_results:
                results["segmentation"] = miou_results

    logger.info(f"Evaluation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()