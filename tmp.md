 File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 651, in fit
    self.on_epoch_end(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 451, in on_epoch_end
    super(LoopBase, self).on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 116, in on_epoch_end
    cb.on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 228, in on_epoch_end
    self._do_val(epoch_id, model, ema_model, device, val_metrics)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 191, in _do_val
    self.predictor.fit()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/predictor.py", line 121, in fit
    super().fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 840, in __call__
    batch_end_callback(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 136, in on_batch_end
    cb.on_batch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/metric_updater.py", line 283, in on_batch_end
    self.metric_update_func(metrics, batch, model_outs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_argmax_bpu.py", line 556, in update_metric
    metrics[idx].update(batch, _get_centerpoint_preds(model_outs[idx]))
  File "/usr/local/lib/python3.10/dist-packages/torchmetrics/metric.py", line 248, in wrapped_func
    return update(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/metrics/nuscenes_metric.py", line 259, in update
    meta = meta[self.meta_key]
KeyError: 'meta'

2026-07-07 11:53:40,043 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 651, in fit
    self.on_epoch_end(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 451, in on_epoch_end
    super(LoopBase, self).on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 116, in on_epoch_end
    cb.on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 228, in on_epoch_end
    self._do_val(epoch_id, model, ema_model, device, val_metrics)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 191, in _do_val
    self.predictor.fit()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/predictor.py", line 121, in fit
    super().fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 840, in __call__
    batch_end_callback(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 136, in on_batch_end
    cb.on_batch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/metric_updater.py", line 283, in on_batch_end
    self.metric_update_func(metrics, batch, model_outs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_argmax_bpu.py", line 556, in update_metric
    metrics[idx].update(batch, _get_centerpoint_preds(model_outs[idx]))
  File "/usr/local/lib/python3.10/dist-packages/torchmetrics/metric.py", line 248, in wrapped_func
    return update(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/metrics/nuscenes_metric.py", line 259, in update
    meta = meta[self.meta_key]
KeyError: 'meta'

2026-07-07 11:53:40,144 ERROR [ddp_trainer.py:463] Node[2] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 651, in fit
    self.on_epoch_end(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 451, in on_epoch_end
    super(LoopBase, self).on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 116, in on_epoch_end
    cb.on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 228, in on_epoch_end
    self._do_val(epoch_id, model, ema_model, device, val_metrics)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 191, in _do_val
    self.predictor.fit()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/predictor.py", line 121, in fit
    super().fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 840, in __call__
    batch_end_callback(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 136, in on_batch_end
    cb.on_batch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/metric_updater.py", line 283, in on_batch_end
    self.metric_update_func(metrics, batch, model_outs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_argmax_bpu.py", line 556, in update_metric
    metrics[idx].update(batch, _get_centerpoint_preds(model_outs[idx]))
  File "/usr/local/lib/python3.10/dist-packages/torchmetrics/metric.py", line 248, in wrapped_func
    return update(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/metrics/nuscenes_metric.py", line 259, in update
    meta = meta[self.meta_key]
KeyError: 'meta'

2026-07-07 11:53:40,263 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 651, in fit
    self.on_epoch_end(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 451, in on_epoch_end
    super(LoopBase, self).on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 116, in on_epoch_end
    cb.on_epoch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 228, in on_epoch_end
    self._do_val(epoch_id, model, ema_model, device, val_metrics)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/validation.py", line 191, in _do_val
    self.predictor.fit()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/predictor.py", line 121, in fit
    super().fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 840, in __call__
    batch_end_callback(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 136, in on_batch_end
    cb.on_batch_end(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/callbacks/metric_updater.py", line 283, in on_batch_end
    self.metric_update_func(metrics, batch, model_outs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/configs/lidar_bevfusion/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes_argmax_bpu.py", line 556, in update_metric
    metrics[idx].update(batch, _get_centerpoint_preds(model_outs[idx]))
  File "/usr/local/lib/python3.10/dist-packages/torchmetrics/metric.py", line 248, in wrapped_func
    return update(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/metrics/nuscenes_metric.py", line 259, in update
    meta = meta[self.meta_key]
KeyError: 'meta'

W0707 11:53:42.494000 5716 torch/multiprocessing/spawn.py:169] Terminating process 5857 via signal SIGTERM
W0707 11:53:42.496000 5716 torch/multiprocessing/spawn.py:169] Terminating process 5858 via signal SIGTERM
W0707 11:53:42.497000 5716 torch/multiprocessing/spawn.py:169] Terminating process 5860 via signal SIGTERM
ERROR:__main__:train failed! process 2 terminated with exit code 1
Traceback (most recent call last):
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 287, in <module>
    raise e
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 273, in <module>
    train(
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 254, in train
    launch(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 426, in launch
    mp.spawn(
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/spawn.py", line 340, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/spawn.py", line 296, in start_processes
    while not context.join():
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/spawn.py", line 204, in join
    raise ProcessExitedException(
torch.multiprocessing.spawn.ProcessExitedException: process 2 terminated with exit code 1