import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand(1000, 1000).cuda()
    y = x @ x

    print("Tensor device:", y.device)
    print("SUCCESS: Code is running on GPU")
else:
    print("Tensor device: CPU")
    print("GPU is NOT being used")