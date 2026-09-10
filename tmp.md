2026-09-10 09:49:54,690 ERROR [ddp_trainer.py:463] Node[3] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 175, in forward
    pred, result = super(BevformerOccDetDecoder, self).forward(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 165, in forward
    return [occ_preds], self._post_process(occ_preds, data)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 135, in _post_process
    return self._loss(occ_preds, voxel_semantics, vis, ceiling)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 112, in _loss
    * sem_scal_loss(
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 87, in sem_scal_loss
    loss_cls = loss_cls + _bce_ones(nominator / p_sum)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 24, in _bce_ones
    return F.binary_cross_entropy(prob, torch.ones_like(prob))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 3569, in binary_cross_entropy
    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.

2026-09-10 09:49:54,690 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 175, in forward
    pred, result = super(BevformerOccDetDecoder, self).forward(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 165, in forward
    return [occ_preds], self._post_process(occ_preds, data)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 135, in _post_process
    return self._loss(occ_preds, voxel_semantics, vis, ceiling)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 112, in _loss
    * sem_scal_loss(
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 87, in sem_scal_loss
    loss_cls = loss_cls + _bce_ones(nominator / p_sum)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 24, in _bce_ones
    return F.binary_cross_entropy(prob, torch.ones_like(prob))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 3569, in binary_cross_entropy
    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.

2026-09-10 09:49:54,690 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 175, in forward
    pred, result = super(BevformerOccDetDecoder, self).forward(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 165, in forward
    return [occ_preds], self._post_process(occ_preds, data)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 135, in _post_process
    return self._loss(occ_preds, voxel_semantics, vis, ceiling)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 112, in _loss
    * sem_scal_loss(
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 87, in sem_scal_loss
    loss_cls = loss_cls + _bce_ones(nominator / p_sum)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 24, in _bce_ones
    return F.binary_cross_entropy(prob, torch.ones_like(prob))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 3569, in binary_cross_entropy
    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.

2026-09-10 09:49:54,691 ERROR [ddp_trainer.py:463] Node[2] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 175, in forward
    pred, result = super(BevformerOccDetDecoder, self).forward(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/flashocc/decoder.py", line 165, in forward
    return [occ_preds], self._post_process(occ_preds, data)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 135, in _post_process
    return self._loss(occ_preds, voxel_semantics, vis, ceiling)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/bevformer_occ_union_mask.py", line 112, in _loss
    * sem_scal_loss(
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 87, in sem_scal_loss
    loss_cls = loss_cls + _bce_ones(nominator / p_sum)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/occ_scal_loss.py", line 24, in _bce_ones
    return F.binary_cross_entropy(prob, torch.ones_like(prob))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 3569, in binary_cross_entropy
    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.

W0910 09:49:56.616000 27 torch/multiprocessing/spawn.py:169] Terminating process 169 via signal SIGTERM
W0910 09:49:56.617000 27 torch/multiprocessing/spawn.py:169] Terminating process 170 via signal SIGTERM
W0910 09:49:56.618000 27 torch/multiprocessing/spawn.py:169] Terminating process 171 via signal SIGTERM
ERROR:__main__:train failed! process 0 terminated with exit code 1
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
torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with exit code 1