docker exec -it s100_g352 bash
cd /open_explorer
pip freeze > /root/pip-freeze-cu126-backup.txt
python3 -c "import torch; print('before:', torch.__version__)"

python3 -m pip install -U pip

pip uninstall -y torch torchvision torchaudio
# 卸载所有旧的 nvidia-cuda-* 12.6 依赖（避免和 12.8 版本并存冲突）
pip uninstall -y \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
pip uninstall -y horizon-plugin-pytorch horizon-plugin-profiler

pip install torch==2.8.0 torchvision==0.23.0
pip install --force-reinstall --no-deps \
  /open_explorer/package/host/ai_toolchain/horizon_plugin_pytorch-3.1.5+cu128.torch280-cp310-cp310-linux_x86_64.whl \
  /open_explorer/package/host/ai_toolchain/horizon_plugin_profiler-3.1.5-py3-none-any.whl

# 验证升级效果
python3 - <<'EOF'
import torch
print("torch  :", torch.__version__)              # 2.8.0
print("cuda   :", torch.version.cuda)             # 12.8
print("archs  :", torch.cuda.get_arch_list())     # 必须含 'sm_120'
print("cap    :", torch.cuda.get_device_capability())  # (12, 0)

import horizon_plugin_pytorch as h
print("plugin :", h.__version__)                  # 3.1.5+cu128.torch280

import hat
print("hat ok")                                    # 纯 Python，应正常

# 实测一个 CUDA kernel，确认真能在这张 sm_120 卡上跑
x = torch.randn(512, 512, device="cuda")
y = torch.randn(512, 512, device="cuda")
print("matmul ok:", (x @ y).sum().item())

img = torch.randint(0, 255, (1, 3, 512, 960), dtype=torch.uint8, device="cuda")
yuv = h.nn.functional.bgr_to_yuv444(img, True)
print("horizon op ok:", yuv.shape, yuv.device)
EOF