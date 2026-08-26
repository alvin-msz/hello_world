/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/maptr/sparse_head.py:157: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/maptr/sparse_decoder.py:121: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/maptr/sparse_decoder.py:139: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
2026-08-26 10:08:39,384 WARNING default upsampling behavior when mode=bilinear is changed to align_corners=False since torch 0.4.0. Please specify align_corners=True if the old behavior is desired. 
Traceback (most recent call last):
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/infer_float_occ_one_frame.py", line 764, in <module>
    main()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/infer_float_occ_one_frame.py", line 744, in main
    model = build_float_model(cfg, ckpt, device)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/infer_float_occ_one_frame.py", line 381, in build_float_model
    torch.load(ckpt_path, map_location="cpu"),
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 1470, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([scalar])` or the `torch.serialization.safe_globals([scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.