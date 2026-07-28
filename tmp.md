  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 185, in train_entrance
    trainer = build_from_registry(trainer)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 414, in build_from_registry
    return _impl(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 397, in _impl
    obj = build_from_cfg(OBJECT_REGISTRY, x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/calibrator.py", line 120, in __init__
    super(Calibrator, self).__init__(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 274, in __init__
    self.model = model_convert_pipeline(self.model)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/model_convert/pipelines.py", line 63, in __call__
    model = converter(model)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/model_convert/converters.py", line 396, in __call__
    model = horizon.quantization.prepare_qat_fx(
  File "/usr/local/lib/python3.10/dist-packages/horizon_plugin_pytorch/utils/typeguard.py", line 1096, in wrapper
    retval = func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/horizon_plugin_pytorch/quantization/quantize_fx.py", line 605, in prepare_qat_fx
    model = _prepare_fx(
  File "/usr/local/lib/python3.10/dist-packages/horizon_plugin_pytorch/quantization/quantize_fx.py", line 297, in _prepare_fx
    graph = tracer.trace(model)
  File "/usr/local/lib/python3.10/dist-packages/horizon_plugin_pytorch/fx/tracer.py", line 248, in trace
    return super(CustomTracer, self).trace(root, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/fx/_symbolic_trace.py", line 843, in trace
    (self.create_arg(fn(*args)),),
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 96, in forward
    lidar_feature = self.forward_lidar_feature(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 71, in forward_lidar_feature
    batch_size=len(example["points"]),
  File "/usr/local/lib/python3.10/dist-packages/torch/fx/proxy.py", line 556, in __len__
    raise RuntimeError(
RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call torch.fx.wrap('len') at module scope

[rank0]:[W728 10:20:23.264609737 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
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