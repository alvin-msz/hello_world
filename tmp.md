2026-07-16 10:02:41,431 ERROR [ddp_trainer.py:463] Node[0] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/predict.py", line 146, in predict_entrance
    load_pred_ckpt_func(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/checkpoint.py", line 420, in load_state_dict
    raise ValueError("set allow_miss=True to skip this check")
ValueError: set allow_miss=True to skip this check

2026-07-16 10:02:41,432 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/predict.py", line 146, in predict_entrance
    load_pred_ckpt_func(
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/checkpoint.py", line 420, in load_state_dict
    raise ValueError("set allow_miss=True to skip this check")
ValueError: set allow_miss=True to skip this check

[rank0]:[W716 10:02:41.495490400 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0716 10:02:42.401000 9018 torch/multiprocessing/spawn.py:169] Terminating process 9159 via signal SIGTERM
W0716 10:02:42.402000 9018 torch/multiprocessing/spawn.py:169] Terminating process 9160 via signal SIGTERM
W0716 10:02:42.403000 9018 torch/multiprocessing/spawn.py:169] Terminating process 9162 via signal SIGTERM
ERROR:__main__:predict failed! process 2 terminated with exit code 1
Traceback (most recent call last):
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/predict.py", line 241, in <module>
    raise e
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/predict.py", line 228, in <module>
    predict(
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/predict.py", line 213, in predict
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