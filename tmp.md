python3 -c "
import torch
ckpt = torch.load(
    './tmp_models/bevfusion_centerpoint_pointpillar_henet_multisensor_multitask_nuscenes/float-checkpoint-last.pth.tar',
    map_location='cpu',
    weights_only=False,
)
print('devices:', ckpt.get('devices'))
print('epoch:', ckpt.get('epoch'))
print('step:', ckpt.get('step'))
print('keys:', list(ckpt.keys()))
"
devices: 4
epoch: 24
step: 28152
keys: ['epoch', 'step', 'devices', 'grad_scaler', 'state_dict', 'horizon-plugin-version', 'optimizer']
