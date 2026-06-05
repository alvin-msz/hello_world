"""Predict and save detection + OCC results for offline evaluation."""
import argparse
import glob
import json
import logging
import os
import shutil
import sys
import pickle
import threading
import tempfile

import numpy as np
import torch

from hat.utils.config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---- 用共享文件收集跨进程 OCC 结果 ----
_OCC_RESULTS_DIR = os.environ.get("_OCC_RESULTS_DIR", "")


def _install_miou_patch():
    """在每个进程（包括子进程）中安装 MeanIOU patch."""
    if not _OCC_RESULTS_DIR:
        return
    try:
        from hat.metrics.mean_iou import MeanIOU
        _original_update = MeanIOU._original_update = getattr(
            MeanIOU, "_original_update", MeanIOU.update
        )
        _counter = {"n": 0}
        _lock = threading.Lock()

        def _patched_update(self, label=None, preds=None, **kwargs):
            if label is not None and preds is not None:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
                with _lock:
                    _counter["n"] += 1
                    n = _counter["n"]
                data = {
                    "gt": label.cpu().numpy().flatten(),
                    "pred": preds.cpu().numpy().flatten(),
                }
                # 用 NamedTemporaryFile 写入同目录，然后 rename
                # delete=False 保证文件不会被自动删除
                occ_dir = _OCC_RESULTS_DIR
                try:
                    os.makedirs(occ_dir, exist_ok=True)
                    fd = tempfile.NamedTemporaryFile(
                        dir=occ_dir,
                        prefix=f".occ_rank{rank}_{n:06d}_",
                        suffix=".pkl",
                        delete=False,
                    )
                    try:
                        pickle.dump(data, fd)
                        fd.flush()
                        os.fsync(fd.fileno())
                    finally:
                        fd.close()
                    final_path = os.path.join(
                        occ_dir, f"occ_rank{rank}_{n:06d}.pkl"
                    )
                    os.rename(fd.name, final_path)
                except Exception as e:
                    logger.warning(
                        f"[Rank {rank}] Failed to save OCC sample {n}: {e}"
                    )
                    # 清理残留
                    try:
                        os.remove(fd.name)
                    except Exception:
                        pass
                if n % 100 == 0:
                    logger.info(
                        f"[Rank {rank}] Collected {n} OCC samples"
                    )
            return _original_update(self, label=label, preds=preds, **kwargs)

        MeanIOU.update = _patched_update
    except ImportError:
        pass


# 自动在 import 时安装 patch（子进程也会执行）
if _OCC_RESULTS_DIR:
    _install_miou_patch()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--device-ids", "-ids", type=str, default=None,
        help="GPU device ids like '0,1,2,3', will use config's device_ids if not set",
    )
    parser.add_argument("--stage", type=str, default="float")
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="checkpoint path, will use config's ckpt if not set",
    )
    return parser.parse_args()


def save_results(config_path, output_dir, stage="float", device_ids=None, ckpt=None):
    """Run predict and save both detection and OCC results."""
    os.makedirs(output_dir, exist_ok=True)

    # 使用绝对路径，避免子进程工作目录不同导致找不到目录
    occ_preds_dir = os.path.abspath(os.path.join(output_dir, "occ_preds"))
    os.makedirs(occ_preds_dir, exist_ok=True)

    # 直接写到最终目录，不再用临时中转目录
    os.environ["_OCC_RESULTS_DIR"] = occ_preds_dir
    _install_miou_patch()

    # ---- Monkey-patch NuscenesMetric 以捕获检测结果保存路径 ----
    det_save_paths = []
    try:
        from hat.metrics import NuscenesMetric
        original_nusc_compute = NuscenesMetric.compute

        def patched_nusc_compute(self, *args, **kwargs):
            result = original_nusc_compute(self, *args, **kwargs)
            if hasattr(self, "save_prefix") and self.save_prefix:
                candidate = os.path.join(self.save_prefix, "results_nusc.json")
                if os.path.exists(candidate):
                    det_save_paths.append(candidate)
                    logger.info(f"Detected results_nusc.json at: {candidate}")
            return result

        NuscenesMetric.compute = patched_nusc_compute
        has_nusc_patch = True
    except (ImportError, AttributeError):
        logger.warning("NuscenesMetric not found, skipping detection capture")
        has_nusc_patch = False

    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        if this_dir not in sys.path:
            sys.path.insert(0, this_dir)

        startup_env = os.environ.get("HAT_STARTUP_IMPORTS", "")
        if "predict_export_json" not in startup_env:
            os.environ["HAT_STARTUP_IMPORTS"] = (
                (startup_env + "," if startup_env else "") + "predict_export_json"
            )

        from predict import predict
        predict(
            stage=stage,
            config=config_path,
            device_ids=device_ids,
            ckpt=ckpt,
        )
    finally:
        if has_nusc_patch:
            NuscenesMetric.compute = original_nusc_compute
        os.environ.pop("_OCC_RESULTS_DIR", None)
        os.environ.pop("HAT_STARTUP_IMPORTS", None)

    # ---- 清理残留的临时文件 ----
    for tmp_f in glob.glob(os.path.join(occ_preds_dir, ".occ_rank*")):
        try:
            os.remove(tmp_f)
        except OSError:
            pass

    occ_files = sorted(glob.glob(os.path.join(occ_preds_dir, "occ_rank*.pkl")))
    occ_count = len(occ_files)

    if occ_count > 0:
        logger.info(f"Saved {occ_count} OCC pkl files to {occ_preds_dir}")
    else:
        logger.warning("No OCC results collected")

    # ---- 复制检测结果到 output_dir ----
    if det_save_paths:
        src = det_save_paths[-1]
        dst = os.path.join(output_dir, "results_nusc.json")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            logger.info(f"Copied detection results: {src} -> {dst}")
        else:
            logger.info(f"Detection results already at: {dst}")
    else:
        cfg = Config.fromfile(config_path)
        task_name = getattr(cfg, "task_name", None)
        if task_name:
            candidate = os.path.join(
                "./metric_results", task_name, "results_nusc.json"
            )
            if os.path.exists(candidate):
                dst = os.path.join(output_dir, "results_nusc.json")
                if os.path.abspath(candidate) != os.path.abspath(dst):
                    shutil.copy2(candidate, dst)
                    logger.info(f"Copied detection results: {candidate} -> {dst}")
        else:
            candidates = sorted(
                glob.glob("./metric_results/*/results_nusc.json"),
                key=os.path.getmtime,
                reverse=True,
            )
            if candidates:
                dst = os.path.join(output_dir, "results_nusc.json")
                if os.path.abspath(candidates[0]) != os.path.abspath(dst):
                    shutil.copy2(candidates[0], dst)
                    logger.info(f"Copied detection results: {candidates[0]} -> {dst}")
            else:
                logger.warning("No detection results_nusc.json found")

    # ---- 汇总 ----
    summary = {
        "has_detection": os.path.exists(os.path.join(output_dir, "results_nusc.json")),
        "has_occ": occ_count > 0,
        "occ_samples": occ_count,
        "occ_preds_dir": occ_preds_dir if occ_count > 0 else None,
    }
    with open(os.path.join(output_dir, "export_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Export summary: {summary}")
    logger.info("Done!")


if __name__ == "__main__":
    args = parse_args()
    save_results(
        args.config,
        args.output_dir,
        args.stage,
        args.device_ids,
        args.ckpt,
    )