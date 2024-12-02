import torch
import os
import sys
import torch
import torch.nn.functional as F

def get_environment_info():
    # Python 版本
    print("Python Version:", sys.version)
    
    # PyTorch 版本
    print("PyTorch Version:", torch.__version__)
    
    # CUDA 是否可用
    print("Is CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        # CUDA 版本
        print("CUDA Version:", torch.version.cuda)
        
        # cuDNN 版本
        print("cuDNN Version:", torch.backends.cudnn.version())
        
        # 可用的 GPU 数量
        print("Number of GPUs Available:", torch.cuda.device_count())
        
        # GPU 名称
        print("GPU Name(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 当前使用的 GPU ID 和名称
        print("Current GPU ID:", torch.cuda.current_device())
        print("Current GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # 环境变量
    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        if 'CUDA' in key or 'cudnn' in key:
            print(f"{key}: {value}")

'''
测试 norm 方法是否相同
'''
def test_norm():
    B = 10
    rot = torch.randn(B, 4)

    # yihao 方法
    rot = F.normalize(rot, p=2, dim=1)

    # 4dgs 方法
    rot_4dgs = torch.nn.functional.normalize(rot)
    
    # 测试是否相同
    if torch.abs(rot - rot_4dgs).max() < 1e-5:
        print("norm is the same.")
    else:
        print("norm is different.")



if __name__ == "__main__":
    # get_environment_info()
    test_norm()


