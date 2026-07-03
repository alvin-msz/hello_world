  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 542, in fit
    _, batch = next(self.data_loader_pr)
  File "/usr/local/lib/python3.10/dist-packages/hat/profiler/profilers.py", line 263, in profile_iterable
    value = next(iterator)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/generator.py", line 15, in prefetch_iterator
    last = next(it)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1480, in _next_data
    return self._process_data(data)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1505, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.10/dist-packages/torch/_utils.py", line 733, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1985, in __getitem__
    data = super().__getitem__(idx)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1762, in __getitem__
    data = self.transforms(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/core/compose_transform.py", line 10, in __call__
    img = t(img)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/transforms/flashocc_transforms.py", line 295, in __call__
    semantics = torch.tensor(data["gt_occ_info"]["voxel_semantics"], dtype=torch.uint8)
KeyError: 'gt_occ_info'


2026-07-03 16:34:02,725 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 542, in fit
    _, batch = next(self.data_loader_pr)
  File "/usr/local/lib/python3.10/dist-packages/hat/profiler/profilers.py", line 263, in profile_iterable
    value = next(iterator)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/generator.py", line 15, in prefetch_iterator
    last = next(it)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1480, in _next_data
    return self._process_data(data)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1505, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.10/dist-packages/torch/_utils.py", line 733, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1985, in __getitem__
    data = super().__getitem__(idx)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/datasets/nuscenes_dataset.py", line 1762, in __getitem__
    data = self.transforms(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/core/compose_transform.py", line 10, in __call__
    img = t(img)
  File "/usr/local/lib/python3.10/dist-packages/hat/data/transforms/flashocc_transforms.py", line 295, in __call__
    semantics = torch.tensor(data["gt_occ_info"]["voxel_semantics"], dtype=torch.uint8)
KeyError: 'gt_occ_info'


W0703 16:34:03.683000 55319 torch/multiprocessing/spawn.py:169] Terminating process 55461 via signal SIGTERM
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