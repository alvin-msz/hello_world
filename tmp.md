/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
2026-07-07 08:49:20,433 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 542, in fit
    _, batch = next(self.data_loader_pr)
  File "/usr/local/lib/python3.10/dist-packages/hat/profiler/profilers.py", line 263, in profile_iterable
    value = next(iterator)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/generator.py", line 20, in prefetch_iterator
    for val in it:
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1480, in _next_data
    return self._process_data(data)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1505, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.10/dist-packages/torch/_utils.py", line 733, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 2.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/collates/nusc_collates.py", line 182, in collate_nuscenes_sequencev2
    collate_data = [d[idx][key] for d in batch]
  File "/usr/local/lib/python3.10/dist-packages/hat/data/collates/nusc_collates.py", line 182, in <listcomp>
    collate_data = [d[idx][key] for d in batch]
KeyError: 'ego_bboxes_labels'


W0707 08:49:22.082000 157134 torch/multiprocessing/spawn.py:169] Terminating process 157275 via signal SIGTERM
W0707 08:49:22.083000 157134 torch/multiprocessing/spawn.py:169] Terminating process 157277 via signal SIGTERM
W0707 08:49:22.084000 157134 torch/multiprocessing/spawn.py:169] Terminating process 157278 via signal SIGTERM
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
