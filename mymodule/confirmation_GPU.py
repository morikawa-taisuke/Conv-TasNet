import torch

print('GPU availability:', torch.cuda.is_available())  # GPU availability:True :GPUが使える場合True, 使えない場合False
print('torch version:', torch.__version__)  # torch version:1.4.0 torchのバージョン
print("CUDA version: ",torch.version.cuda)  # CUDAのバージョン
print('Number of GPUs available in PyTorch:', torch.cuda.device_count())  # Number of GPUs available in PyTorch:1 使えるGPUの個数?
print('GPU name:', torch.cuda.get_device_name())  # GPU name:NVIDIA GeForce RTX 3060 Ti 使えるGPUの種類
