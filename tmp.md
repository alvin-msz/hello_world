2026-07-06 14:08:58,480 INFO [monitor.py:131] Node[0] Epoch[2] Step[200-399] Cost Time: 273.169s Speed: 2.93 samples/sec Remaining Time: 14:31:49 Remaining step percent: 90.72%
2026-07-06 14:08:59,644 INFO [metric_updater.py:360] Node[0] Epoch[2] Step[399] GlobalStep[3917] loss_bevfusion_pointpillar_henet_multisensor_multitask_nuscenes: loss_cls[0.2020] loss_bbox[0.4469] d0.loss_cls[0.2761] d0.loss_bbox[0.5519] d1.loss_cls[0.2717] d1.loss_bbox[0.4798] d2.loss_cls[0.2503] d2.loss_bbox[0.4610] d3.loss_cls[0.2214] d3.loss_bbox[0.4533] d4.loss_cls[0.2047] d4.loss_bbox[0.4486] loss_occ[1.4812] 
2026-07-06 14:09:22,145 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 96, in forward
    lidar_feature = self.forward_lidar_feature(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 85, in forward_lidar_feature
    x = self.lidar_net.neck(x)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/necks/second_neck.py", line 109, in forward
    out = self.cat.cat(ups, dim=1)
  File "/usr/local/lib/python3.10/dist-packages/torch/ao/nn/quantized/modules/functional_modules.py", line 82, in cat
    r = torch.cat(x, dim=dim)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 22.19 MiB is free. Process 739711 has 2.49 GiB memory in use. Process 1025873 has 20.97 GiB memory in use. Of the allocated memory 1.81 GiB is allocated by PyTorch, and 128.30 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

W0706 14:09:36.633000 101577 torch/multiprocessing/spawn.py:169] Terminating process 101719 via signal SIGTERM
W0706 14:09:36.636000 101577 torch/multiprocessing/spawn.py:169] Terminating process 101720 via signal SIGTERM
W0706 14:09:36.637000 101577 torch/multiprocessing/spawn.py:169] Terminating process 101721 via signal SIGTERM
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