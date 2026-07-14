2026-07-14 13:33:46,654 ERROR [ddp_trainer.py:463] Node[2] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 790, in __call__
    model_outs = model(*_as_list(batch_i))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1643, in forward
    else self._run_ddp_forward(*inputs, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1459, in _run_ddp_forward
    return self.module(*inputs, **kwargs)  # type: ignore[index]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 116, in forward
    results = self.post_process(data, pts_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 123, in post_process
    result = bev_decoder(pts_feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevfusion_centerpoint_bboxes_adapter_opt.py", line 577, in forward
    out = super().forward(feats, meta)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 148, in forward
    return self._post_process(meta, pred)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 168, in _post_process
    loss = self._loss(target)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 317, in _loss
    cls_loss = self.loss_cls(**cls_target)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
TypeError: ClampedGaussianFocalLoss.forward() missing 2 required positional arguments: 'pred' and 'target'

2026-07-14 13:33:46,675 ERROR [ddp_trainer.py:463] Node[3] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 790, in __call__
    model_outs = model(*_as_list(batch_i))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1643, in forward
    else self._run_ddp_forward(*inputs, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1459, in _run_ddp_forward
    return self.module(*inputs, **kwargs)  # type: ignore[index]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 116, in forward
    results = self.post_process(data, pts_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 123, in post_process
    result = bev_decoder(pts_feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevfusion_centerpoint_bboxes_adapter_opt.py", line 577, in forward
    out = super().forward(feats, meta)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 148, in forward
    return self._post_process(meta, pred)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 168, in _post_process
    loss = self._loss(target)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 317, in _loss
    cls_loss = self.loss_cls(**cls_target)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
TypeError: ClampedGaussianFocalLoss.forward() missing 2 required positional arguments: 'pred' and 'target'

2026-07-14 13:33:46,684 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 790, in __call__
    model_outs = model(*_as_list(batch_i))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1643, in forward
    else self._run_ddp_forward(*inputs, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1459, in _run_ddp_forward
    return self.module(*inputs, **kwargs)  # type: ignore[index]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 116, in forward
    results = self.post_process(data, pts_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 123, in post_process
    result = bev_decoder(pts_feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevfusion_centerpoint_bboxes_adapter_opt.py", line 577, in forward
    out = super().forward(feats, meta)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 148, in forward
    return self._post_process(meta, pred)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 168, in _post_process
    loss = self._loss(target)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 317, in _loss
    cls_loss = self.loss_cls(**cls_target)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
TypeError: ClampedGaussianFocalLoss.forward() missing 2 required positional arguments: 'pred' and 'target'

2026-07-14 13:33:46,798 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 557, in fit
    self.batch_processor(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/deterministic.py", line 253, in wrapper
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/processors/processor.py", line 790, in __call__
    model_outs = model(*_as_list(batch_i))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1643, in forward
    else self._run_ddp_forward(*inputs, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parallel/distributed.py", line 1459, in _run_ddp_forward
    return self.module(*inputs, **kwargs)  # type: ignore[index]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 116, in forward
    results = self.post_process(data, pts_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 123, in post_process
    result = bev_decoder(pts_feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevfusion_centerpoint_bboxes_adapter_opt.py", line 577, in forward
    out = super().forward(feats, meta)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 148, in forward
    return self._post_process(meta, pred)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 168, in _post_process
    loss = self._loss(target)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/view_fusion/decoder.py", line 317, in _loss
    cls_loss = self.loss_cls(**cls_target)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
TypeError: ClampedGaussianFocalLoss.forward() missing 2 required positional arguments: 'pred' and 'target'

W0714 13:33:49.193000 135093 torch/multiprocessing/spawn.py:169] Terminating process 135234 via signal SIGTERM
W0714 13:33:49.194000 135093 torch/multiprocessing/spawn.py:169] Terminating process 135237 via signal SIGTERM
ERROR:__main__:train failed! process 1 terminated with exit code 1
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
torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with exit code 1