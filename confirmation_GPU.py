import torch

print('torch version:', torch.__version__)  # torchのバージョン
print('GPU availability:', torch.cuda.is_available())  # GPUが使える場合True, 使えない場合False
print('Number of GPUs available in PyTorch:', torch.cuda.device_count())  # 使えるGPUの個数?
print('GPU name:', torch.cuda.get_device_name())  # PCに搭載されているGPUの名前
