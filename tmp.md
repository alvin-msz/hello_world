2026-07-21 10:48:18,020 ERROR [ddp_trainer.py:463] Node[3] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 391, in _impl
    obj = _build_dataset(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 261, in _build_dataset
    obj = build_from_cfg(OBJECT_REGISTRY, cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1975, in __init__
    super(NuscenesBevSequenceDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1723, in __init__
    super(NuscenesBevDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/package_helper.py", line 243, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1516, in __init__
    self.pack_type = get_packtype_from_path(data_path)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/pack_type/utils.py", line 36, in get_packtype_from_path
    assert os.path.exists(path), f"{path} does not exist!"
AssertionError: data/bevfusion_nuscenes/train_lmdb/ does not exist!

After filter database:
load 56455 traffic_cone database infos
load 60691 truck database infos
load 296423 car database infos
load 149198 pedestrian database infos
load 19195 movable_object.pushable_pullable database infos
load 10622 construction_vehicle database infos
load 102476 barrier database infos
load 2120 movable_object.debris database infos
load 8094 motorcycle database infos
load 7565 bicycle database infos
load 11662 bus database infos
load 2259 static_object.bicycle_rack database infos
load 18165 trailer database infos
load 751 human.pedestrian.stroller database infos
load 619 animal database infos
load 352 human.pedestrian.personal_mobility database infos
load 492 human.pedestrian.wheelchair database infos
load 11 vehicle.emergency.ambulance database infos
load 498 vehicle.emergency.police database infos
2026-07-21 10:48:18,075 ERROR [ddp_trainer.py:463] Node[2] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 391, in _impl
    obj = _build_dataset(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 261, in _build_dataset
    obj = build_from_cfg(OBJECT_REGISTRY, cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1975, in __init__
    super(NuscenesBevSequenceDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1723, in __init__
    super(NuscenesBevDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/package_helper.py", line 243, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1516, in __init__
    self.pack_type = get_packtype_from_path(data_path)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/pack_type/utils.py", line 36, in get_packtype_from_path
    assert os.path.exists(path), f"{path} does not exist!"
AssertionError: data/bevfusion_nuscenes/train_lmdb/ does not exist!

After filter database:
load 56455 traffic_cone database infos
load 60691 truck database infos
load 296423 car database infos
load 149198 pedestrian database infos
load 19195 movable_object.pushable_pullable database infos
load 10622 construction_vehicle database infos
load 102476 barrier database infos
load 2120 movable_object.debris database infos
load 8094 motorcycle database infos
load 7565 bicycle database infos
load 11662 bus database infos
load 2259 static_object.bicycle_rack database infos
load 18165 trailer database infos
load 751 human.pedestrian.stroller database infos
load 619 animal database infos
load 352 human.pedestrian.personal_mobility database infos
load 492 human.pedestrian.wheelchair database infos
load 11 vehicle.emergency.ambulance database infos
load 498 vehicle.emergency.police database infos
2026-07-21 10:48:18,261 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 391, in _impl
    obj = _build_dataset(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 261, in _build_dataset
    obj = build_from_cfg(OBJECT_REGISTRY, cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1975, in __init__
    super(NuscenesBevSequenceDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1723, in __init__
    super(NuscenesBevDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/package_helper.py", line 243, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1516, in __init__
    self.pack_type = get_packtype_from_path(data_path)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/pack_type/utils.py", line 36, in get_packtype_from_path
    assert os.path.exists(path), f"{path} does not exist!"
AssertionError: data/bevfusion_nuscenes/train_lmdb/ does not exist!

After filter database:
load 56455 traffic_cone database infos
load 60691 truck database infos
load 296423 car database infos
load 149198 pedestrian database infos
load 19195 movable_object.pushable_pullable database infos
load 10622 construction_vehicle database infos
load 102476 barrier database infos
load 2120 movable_object.debris database infos
load 8094 motorcycle database infos
load 7565 bicycle database infos
load 11662 bus database infos
load 2259 static_object.bicycle_rack database infos
load 18165 trailer database infos
load 751 human.pedestrian.stroller database infos
load 619 animal database infos
load 352 human.pedestrian.personal_mobility database infos
load 492 human.pedestrian.wheelchair database infos
load 11 vehicle.emergency.ambulance database infos
load 498 vehicle.emergency.police database infos
2026-07-21 10:48:18,735 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 391, in _impl
    obj = _build_dataset(x)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 261, in _build_dataset
    obj = build_from_cfg(OBJECT_REGISTRY, cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/registry.py", line 248, in build_from_cfg
    instance = obj_cls(**cfg)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1975, in __init__
    super(NuscenesBevSequenceDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1723, in __init__
    super(NuscenesBevDataset, self).__init__(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/package_helper.py", line 243, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1516, in __init__
    self.pack_type = get_packtype_from_path(data_path)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/pack_type/utils.py", line 36, in get_packtype_from_path
    assert os.path.exists(path), f"{path} does not exist!"
AssertionError: data/bevfusion_nuscenes/train_lmdb/ does not exist!

[rank0]:[W721 10:48:19.840593899 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0721 10:48:19.428000 63772 torch/multiprocessing/spawn.py:169] Terminating process 63912 via signal SIGTERM
W0721 10:48:19.430000 63772 torch/multiprocessing/spawn.py:169] Terminating process 63913 via signal SIGTERM
W0721 10:48:19.432000 63772 torch/multiprocessing/spawn.py:169] Terminating process 63914 via signal SIGTERM
ERROR:__main__:train failed! process 3 terminated with exit code 1
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
torch.multiprocessing.spawn.ProcessExitedException: process 3 terminated with exit code 1
