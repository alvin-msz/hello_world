"""OccSegMetric: OCC mIOU metric matching evaluate_nuscenes.py's computation.

This is a drop-in replacement for hat.metrics.mean_iou.MeanIOU that uses the
exact same confusion-matrix-based mIOU computation as evaluate_nuscenes.py.

Key behaviors (matching evaluate_nuscenes.py):
  - Keeps ignore_index voxels in the confusion matrix so wrong predictions
    on free-space voxels still penalize semantic classes.
  - Only excludes ignore_index from the final mIoU mean (nanmean over
    non-ignore classes).
  - Clips pred/gt to [0, num_classes-1] before accumulation.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OccSegMetric:
    """OCC segmentation metric matching evaluate_nuscenes.py logic.

    Args:
        seg_class: List of class name strings. Determines num_classes.
                   If None, num_classes is inferred from data.
        ignore_index: Class index to exclude from mIoU mean (default 17
                      for occ3d-nuscenes "free" class).

    Interface (drop-in compatible with MeanIOU):
        - update(label, preds): accumulate a batch of flattened label/preds
        - compute(): return dict with "mIOU" key
        - reset(): clear accumulated confusion matrix
    """

    def __init__(self, seg_class=None, ignore_index=17):
        self.seg_class = seg_class
        self.ignore_index = ignore_index
        self._num_classes = len(seg_class) if seg_class is not None else None
        self.confusion_matrix = None
        self._loaded = 0

    @property
    def name(self):
        return "OccSegMetric"

    def update(self, label=None, preds=None, **kwargs):
        """Accumulate predictions into confusion matrix.

        Args:
            label: Ground-truth labels, 1-D array-like (numpy or torch tensor).
            preds: Predicted labels, 1-D array-like (numpy or torch tensor).
        """
        if label is None or preds is None:
            return

        # Convert torch tensors to numpy
        if hasattr(label, "cpu"):
            label = label.cpu().numpy()
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()

        gt = np.asarray(label, dtype=np.int64).flatten()
        pred = np.asarray(preds, dtype=np.int64).flatten()

        if gt.size == 0 or pred.size == 0:
            return

        # Infer num_classes from data if not set
        if self._num_classes is None:
            self._num_classes = int(max(pred.max(), gt.max())) + 1
        # Expand if ignore_index exceeds current num_classes
        if self.ignore_index >= self._num_classes:
            self._num_classes = self.ignore_index + 1

        # clip to valid range [0, num_classes-1]
        pred = np.clip(pred, 0, self._num_classes - 1)
        gt = np.clip(gt, 0, self._num_classes - 1)

        # Filter out invalid entries (negative values shouldn't exist after
        # clipping, but keep the guard for safety)
        valid = (gt >= 0) & (gt < self._num_classes)
        pred = pred[valid]
        gt = gt[valid]

        if pred.size == 0:
            return

        if self.confusion_matrix is None:
            self.confusion_matrix = np.zeros(
                (self._num_classes, self._num_classes), dtype=np.int64
            )

        np.add.at(self.confusion_matrix, (gt, pred), 1)
        self._loaded += 1

    def compute(self):
        """Compute mIOU from accumulated confusion matrix.

        In DDP mode, confusion matrices from all ranks are all-reduced
        (summed) before computing the final mIOU, so the result matches
        evaluate_nuscenes.py on the full dataset.

        Returns:
            dict with keys:
                "mIOU": float, mean IoU over non-ignore classes
                "per_class_iou": dict, class_name -> IoU value
        """
        if self.confusion_matrix is None or self._loaded == 0:
            logger.warning("OccSegMetric: no data accumulated")
            return {"mIOU": 0.0}

        # --- DDP sync: all-reduce confusion matrices across ranks ---
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            # NCCL backend requires GPU tensors — move to CUDA, sync, then
            # move back to CPU for numpy computation.
            cm_tensor = torch.from_numpy(
                self.confusion_matrix.astype(np.int64)
            ).cuda()
            dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
            confusion_matrix = cm_tensor.cpu().numpy()

            loaded_tensor = torch.tensor(
                [self._loaded], dtype=torch.int64
            ).cuda()
            dist.all_reduce(loaded_tensor, op=dist.ReduceOp.SUM)
            total_loaded = int(loaded_tensor.cpu().item())
        else:
            confusion_matrix = self.confusion_matrix
            total_loaded = self._loaded

        num_classes = confusion_matrix.shape[0]

        intersection = np.diag(confusion_matrix)
        union = (
            confusion_matrix.sum(axis=1)
            + confusion_matrix.sum(axis=0)
            - intersection
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = intersection.astype(np.float64) / union.astype(np.float64)
        iou[union == 0] = np.nan

        valid_classes = list(range(num_classes))
        if 0 <= self.ignore_index < num_classes:
            valid_classes = [
                c for c in valid_classes if c != self.ignore_index
            ]

        miou = float(np.nanmean(iou[valid_classes]))

        class_labels = (
            list(self.seg_class)
            if self.seg_class
            else [str(i) for i in range(num_classes)]
        )

        per_class_iou = {}
        for i in valid_classes:
            name = class_labels[i] if i < len(class_labels) else str(i)
            val = float(iou[i]) if not np.isnan(iou[i]) else 0.0
            per_class_iou[name] = val

        result = {
            "mIOU": miou,
            "per_class_iou": per_class_iou,
        }

        logger.info(
            f"OccSegMetric mIOU: {miou:.4f} (over {total_loaded} frames "
            f"across all ranks)"
        )
        for name, val in per_class_iou.items():
            logger.info(f"  {name}: {val:.4f}")

        return result

    def get(self):
        """Return (name, value) for HAT MetricUpdater._log compatibility.

        Called by MetricUpdater.on_epoch_end → _log(m, prefix).
        """
        result = self.compute()
        return ("mIOU", result.get("mIOU", 0.0))

    def reset(self):
        """Reset accumulated confusion matrix."""
        self.confusion_matrix = None
        self._loaded = 0

    # ---- nn.Module compatibility (for HAT Predictor which calls m.to(device)) ----
    def to(self, *args, **kwargs):
        """No-op for device compatibility with HAT Predictor.set_device."""
        return self

    def cuda(self, *args, **kwargs):
        """No-op for device compatibility."""
        return self

    def cpu(self, *args, **kwargs):
        """No-op for device compatibility."""
        return self

    def train(self, mode=True):
        """No-op for module compatibility."""
        return self

    def eval(self):
        """No-op for module compatibility."""
        return self

    def parameters(self, recurse=True):
        """Return empty iterator for module compatibility."""
        return iter(())

    def named_parameters(self, prefix="", recurse=True):
        """Return empty iterator for module compatibility."""
        return iter(())

    def modules(self):
        """Yield self for module compatibility."""
        yield self

    def state_dict(self, *args, **kwargs):
        """Return empty dict for module compatibility."""
        return {}

    def load_state_dict(self, state_dict, strict=True):
        """No-op for module compatibility."""
        pass
