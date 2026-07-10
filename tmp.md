2026-07-10 11:13:12,936 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 217, in __init__
    super(DistributedDataParallelTrainer, self).__init__(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/trainer.py", line 92, in __init__
    super(Trainer, self).__init__(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 290, in __init__
    self._resume_from_checkpoint(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 361, in _resume_from_checkpoint
    assert (
AssertionError: Resume only when number of devices is consistent

2026-07-10 11:13:13,120 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
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
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 217, in __init__
    super(DistributedDataParallelTrainer, self).__init__(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/trainer.py", line 92, in __init__
    super(Trainer, self).__init__(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 290, in __init__
    self._resume_from_checkpoint(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 361, in _resume_from_checkpoint
    assert (
AssertionError: Resume only when number of devices is consistent

[rank0]:[W710 11:13:13.426761195 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0710 11:13:14.289000 279137 torch/multiprocessing/spawn.py:169] Terminating process 279279 via signal SIGTERM
W0710 11:13:14.291000 279137 torch/multiprocessing/spawn.py:169] Terminating process 279280 via signal SIGTERM
W0710 11:13:14.292000 279137 torch/multiprocessing/spawn.py:169] Terminating process 279281 via signal SIGTERM
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