import torch
import torch.backends.cudnn as cudnn
# import pytorch3d

# print("Python version:", platform.python_version())
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("cuDNN version:", cudnn.version())
# print("PyTorch3D version:", pytorch3d.__version__)
