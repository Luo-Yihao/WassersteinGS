from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from debug_utils import debug_print

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        # debug_print("手动修改相机参数R,T")
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask

                '''
                手动修改相机参数
                '''
                # 手动修改相机参数

                # tilt_angle = np.radians(90)  # 45度俯视角，可以根据需要调整
                # tilt_matrix = np.array([
                #     [1, 0, 0],
                #     [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
                #     [0, np.sin(tilt_angle), np.cos(tilt_angle)]
                # ], dtype=R.dtype)

                # # 将倾斜矩阵应用到原始旋转矩阵上
                # R = np.dot(tilt_matrix, R)
                # # R = tilt_matrix

                # # 调整相机位置（可选）
                # x_adjustment = 0.0  # 向左移动相机，可以根据需要调整
                # y_adjustment = 0.0  # 向上移动相机，可以根据需要调整
                # z_adjustment = 0.0  # 向前移动相机，可以根据需要调整
                # T[0] += x_adjustment
                # T[1] += y_adjustment
                # T[2] += z_adjustment
                '''
                手动修改相机参数
                ''' 

            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
