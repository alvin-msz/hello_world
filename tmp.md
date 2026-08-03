2026-08-03 08:41:48,914 ERROR [ddp_trainer.py:463] Node[3] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 99, in forward
    camera_feature = self.forward_camera_feature(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 91, in forward_camera_feature
    bev_feat = self.camera_net.view_transformer(feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 877, in forward
    bev_emb = self.get_bev_embed(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 845, in get_bev_embed
    bev_embed = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 400, in forward
    ) = self.get_encoder_inputs(mlvl_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 360, in get_encoder_inputs
    feat = self.addcams_embeds[lvl].add(feat, cams_embeds)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 334, in __getitem__
    return self._modules[self._get_abs_string_index(idx)]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 316, in _get_abs_string_index
    raise IndexError(f"index {idx} is out of range")
IndexError: index 1 is out of range

2026-08-03 08:41:48,914 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 99, in forward
    camera_feature = self.forward_camera_feature(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 91, in forward_camera_feature
    bev_feat = self.camera_net.view_transformer(feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 877, in forward
    bev_emb = self.get_bev_embed(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 845, in get_bev_embed
    bev_embed = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 400, in forward
    ) = self.get_encoder_inputs(mlvl_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 360, in get_encoder_inputs
    feat = self.addcams_embeds[lvl].add(feat, cams_embeds)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 334, in __getitem__
    return self._modules[self._get_abs_string_index(idx)]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 316, in _get_abs_string_index
    raise IndexError(f"index {idx} is out of range")
IndexError: index 1 is out of range

2026-08-03 08:41:48,914 ERROR [ddp_trainer.py:463] Node[2] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 99, in forward
    camera_feature = self.forward_camera_feature(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 91, in forward_camera_feature
    bev_feat = self.camera_net.view_transformer(feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 877, in forward
    bev_emb = self.get_bev_embed(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 845, in get_bev_embed
    bev_embed = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 400, in forward
    ) = self.get_encoder_inputs(mlvl_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 360, in get_encoder_inputs
    feat = self.addcams_embeds[lvl].add(feat, cams_embeds)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 334, in __getitem__
    return self._modules[self._get_abs_string_index(idx)]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 316, in _get_abs_string_index
    raise IndexError(f"index {idx} is out of range")
IndexError: index 1 is out of range

2026-08-03 08:41:48,914 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 99, in forward
    camera_feature = self.forward_camera_feature(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 91, in forward_camera_feature
    bev_feat = self.camera_net.view_transformer(feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 877, in forward
    bev_emb = self.get_bev_embed(
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 845, in get_bev_embed
    bev_embed = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 400, in forward
    ) = self.get_encoder_inputs(mlvl_feats)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/encoder.py", line 360, in get_encoder_inputs
    feat = self.addcams_embeds[lvl].add(feat, cams_embeds)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 334, in __getitem__
    return self._modules[self._get_abs_string_index(idx)]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 316, in _get_abs_string_index
    raise IndexError(f"index {idx} is out of range")
IndexError: index 1 is out of range

W0803 08:41:50.621000 32 torch/multiprocessing/spawn.py:169] Terminating process 109 via signal SIGTERM
W0803 08:41:50.622000 32 torch/multiprocessing/spawn.py:169] Terminating process 110 via signal SIGTERM
W0803 08:41:50.622000 32 torch/multiprocessing/spawn.py:169] Terminating process 112 via signal SIGTERM
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