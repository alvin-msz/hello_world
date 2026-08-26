/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/infer_float_occ_one_frame.py", line 772, in <module>
    main()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/infer_float_occ_one_frame.py", line 753, in main
    outs = model(model_input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 99, in forward
    camera_feature = self.forward_camera_feature(data)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/structures/bevfusion.py", line 91, in forward_camera_feature
    bev_feat = self.camera_net.view_transformer(feats, data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 876, in forward
    ) = self.point_sampling(ref3d, self.pc_range, img_meta, im_shape)
  File "/usr/local/lib/python3.10/dist-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/bevformer/view_transformer.py", line 233, in point_sampling
    ego2img = np.asarray(ego2img)
  File "/usr/local/lib/python3.10/dist-packages/torch/_tensor.py", line 1194, in __array__
    return self.numpy()
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.