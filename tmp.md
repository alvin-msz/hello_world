python3 -c "
import torch
ckpt = torch.load('./tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes/float-checkpoint-last.pth.tar', map_location='cpu')
print('devices:', ckpt.get('devices'))
print('epoch:', ckpt.get('epoch'))
print('step:', ckpt.get('step'))
"
Traceback (most recent call last):
  File "<string>", line 3, in <module>
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 1470, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([scalar])` or the `torch.serialization.safe_globals([scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.