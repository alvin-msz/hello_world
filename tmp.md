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

[rank0]:[W729 08:19:58.228583069 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
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

python3 samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py --stage calibration --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/detection/centerpoint/bevformer_pointpillar_nuscenes_opt_resume.py --device-ids 7


mAP: 0.4715
mATE: 0.4047
mASE: 0.2797
mAOE: 0.4515
mAVE: 0.3662
mAAE: 0.1922
NDS: 0.5663
Eval time: 105.0s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.834	0.198	0.158	0.140	0.317	0.203
truck	0.450	0.408	0.211	0.171	0.320	0.232
bus	0.597	0.413	0.206	0.188	0.723	0.249
trailer	0.341	0.613	0.230	0.529	0.259	0.191
construction_vehicle	0.154	0.794	0.479	1.068	0.151	0.338
pedestrian	0.743	0.188	0.284	0.391	0.245	0.092
motorcycle	0.470	0.333	0.276	0.635	0.554	0.214
bicycle	0.135	0.391	0.299	0.875	0.361	0.019
traffic_cone	0.486	0.265	0.346	nan	nan	nan
barrier	0.505	0.445	0.306	0.067	nan	nan
2026-07-28 17:04:53,054 INFO [nuscenes_metric.py:388] Node[0] NDS: 0.5663, mAP:0.4715
car_AP: [0.5]:0.7239  [1.0]:0.8393  [2.0]:0.8782  [4.0]:0.8947 
truck_AP: [0.5]:0.2638  [1.0]:0.4223  [2.0]:0.5371  [4.0]:0.5778 
trailer_AP: [0.5]:0.0699  [1.0]:0.2759  [2.0]:0.4358  [4.0]:0.5806 
bus_AP: [0.5]:0.3394  [1.0]:0.5691  [2.0]:0.7230  [4.0]:0.7583 
construction_vehicle_AP: [0.5]:0.0025  [1.0]:0.0876  [2.0]:0.2274  [4.0]:0.2982 
bicycle_AP: [0.5]:0.0934  [1.0]:0.1371  [2.0]:0.1471  [4.0]:0.1608 
motorcycle_AP: [0.5]:0.3319  [1.0]:0.4811  [2.0]:0.5213  [4.0]:0.5462 
pedestrian_AP: [0.5]:0.6797  [1.0]:0.7339  [2.0]:0.7678  [4.0]:0.7916 
traffic_cone_AP: [0.5]:0.4032  [1.0]:0.4633  [2.0]:0.5139  [4.0]:0.5626 
barrier_AP: [0.5]:0.2718  [1.0]:0.4914  [2.0]:0.6083  [4.0]:0.6477