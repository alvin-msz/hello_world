2026-07-21 10:42:59,010 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 185, in train_entrance
    trainer = build_from_registry(trainer)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 414, in build_from_registry
    return _impl(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in _impl
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in <genexpr>
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 397, in _impl
    obj = build_from_cfg(OBJECT_REGISTRY, x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/transforms/lidar_utils/sample_ops.py", line 80, in __init__
    with open(info_path, "rb") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'data/bevfusion_nuscenes/nuscenes_dbinfos_train.pkl'

[-1]
{'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}
2026-07-21 10:42:59,024 ERROR [ddp_trainer.py:463] Node[3] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 185, in train_entrance
    trainer = build_from_registry(trainer)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 414, in build_from_registry
    return _impl(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in _impl
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in <genexpr>
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 397, in _impl
    obj = build_from_cfg(OBJECT_REGISTRY, x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/transforms/lidar_utils/sample_ops.py", line 80, in __init__
    with open(info_path, "rb") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'data/bevfusion_nuscenes/nuscenes_dbinfos_train.pkl'

[-1]
{'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}
2026-07-21 10:42:59,051 ERROR [ddp_trainer.py:463] Node[2] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 185, in train_entrance
    trainer = build_from_registry(trainer)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 414, in build_from_registry
    return _impl(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in _impl
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in <genexpr>
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 397, in _impl
    obj = build_from_cfg(OBJECT_REGISTRY, x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/transforms/lidar_utils/sample_ops.py", line 80, in __init__
    with open(info_path, "rb") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'data/bevfusion_nuscenes/nuscenes_dbinfos_train.pkl'

[-1]
{'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}
2026-07-21 10:42:59,054 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 185, in train_entrance
    trainer = build_from_registry(trainer)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 414, in build_from_registry
    return _impl(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in _impl
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 353, in <genexpr>
    x = type(x)((_impl(x_i) for x_i in x))
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in _impl
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 370, in <genexpr>
    build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 397, in _impl
    obj = build_from_cfg(OBJECT_REGISTRY, x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/transforms/lidar_utils/sample_ops.py", line 80, in __init__
    with open(info_path, "rb") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'data/bevfusion_nuscenes/nuscenes_dbinfos_train.pkl'

[rank0]:[W721 10:42:59.903866878 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0721 10:42:59.982000 63074 torch/multiprocessing/spawn.py:169] Terminating process 63215 via signal SIGTERM
W0721 10:42:59.983000 63074 torch/multiprocessing/spawn.py:169] Terminating process 63216 via signal SIGTERM
W0721 10:42:59.984000 63074 torch/multiprocessing/spawn.py:169] Terminating process 63217 via signal SIGTERM
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
