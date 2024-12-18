import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
import numpy as np


# waterstein distance of two gaussians modified by Junli
class WassersteinGaussian(nn.Module):
    def __init__(self):
        super(WassersteinGaussian, self).__init__()

    def forward(self, loc1, scale1, rot_matrix1, loc2, scale2=None, rot_matrix2=None, cov2=None):
        """
        计算两个高斯分布之间的 Wasserstein 距离
        loc1, loc2: Bx3
        scale1, scale2: Bx3
        rot_matrix1, rot_matrix2: Bx3x3
        """
        # 确保输入的尺度参数为非负
        scale1 = torch.clamp(scale1, min=1e-8)
        if scale2 is not None:
            scale2 = torch.clamp(scale2, min=1e-8)

        # 计算位置差异
        loc_diff2 = torch.sum((loc1 - loc2)**2, dim=-1)

        # Wasserstein 距离 Tr(C1 + C2 - 2(C1^0.5 * C2 * C1^0.5)^0.5)
        cov1_sqrt_diag = torch.sqrt(scale1).diag_embed()  # Bx3x3

        if cov2 is None:
            assert rot_matrix2 is not None and scale2 is not None
            cov2 = torch.bmm(rot_matrix2, torch.bmm(torch.diag_embed(scale2), rot_matrix2.transpose(1, 2)))  # 协方差矩阵 Bx3x3

        cov2_R1 = torch.bmm(rot_matrix1.transpose(1, 2), cov2).matmul(rot_matrix1)  # Bx3x3
        # E = cv1^0.5 * cv2 * cv1^0.5
        E = torch.bmm(torch.bmm(cov1_sqrt_diag, cov2_R1), cov1_sqrt_diag)  # Bx3x3

        # 确保 E 是对称的
        E = (E + E.transpose(1, 2)) / 2

        # 添加数值稳定性处理
        epsilon = 1e-8
        batch_size, dim, _ = E.size()
        identity = torch.eye(dim, device=E.device).expand(batch_size, dim, dim)
        E = E + epsilon * identity

        # 计算特征值并进行裁剪
        E_eign = torch.linalg.eigvalsh(E)
        E_eign = torch.clamp(E_eign, min=1e-8)

        # 计算 E 的平方根的迹
        E_sqrt_trace = torch.sqrt(E_eign).sum(dim=-1)

        # # 计算 E 的平方根的迹，dengjunli修改
        # U, S, Vh = torch.linalg.svd(E)
        # S = torch.clamp(S, min=1e-8)
        # E_sqrt_trace = torch.sqrt(S).sum(dim=-1)

        # 计算协方差的 Wasserstein 距离
        CovWasserstein = scale1.sum(dim=-1) + cov2.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 2 * E_sqrt_trace
        
        # 确保 CovWasserstein 非负
        CovWasserstein = torch.clamp(CovWasserstein, min=0)

        # 添加调试信息
        # if torch.isnan(CovWasserstein).any() or torch.isinf(CovWasserstein).any():
        #     print("CovWasserstein contains NaN or Inf")
        #     print("scale1:", scale1)
        #     print("cov2:", cov2)
        #     print("E_eign:", E_eign)
        #     print("E_sqrt_trace:", E_sqrt_trace)

        # 返回 Wasserstein 距离
        # return torch.sqrt(loc_diff2 + CovWasserstein)
        return torch.sqrt(torch.clamp(loc_diff2 + CovWasserstein, min=1e-8))


'''
通过前一帧的高斯分布和当前帧的速度和速度协方差，计算当前帧的高斯分布，卡尔曼滤波结果
'''
class WassersteinExp(nn.Module):
    def __init__(self):
        super(WassersteinExp, self).__init__()

    def forward(self, loc, cov1=None, scale1=None, rot_matrix1=None, velocity=None, velocity_cov=None):
        """
        计算 Wasserstein Exponential of X from A
        支持两种输入方式:
        1. loc, scale1, rot_matrix1, velocity, velocity_cov
        2. loc, cov1, velocity, velocity_cov
        """
        # 初始化默认返回值
        new_loc = loc + velocity
        if cov1 is not None:
            new_cov = cov1  # 初始化为输入的cov1
        elif scale1 is not None and rot_matrix1 is not None:
            new_cov = torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2)))
        else:
            raise ValueError("必须提供 scale1 和 rot_matrix1 或 cov1")

        # 初始化稳定性检查掩码
        stable_mask = torch.ones(loc.shape[0], dtype=torch.bool, device=loc.device)
        
        # 1. 检查输入的cov1是否有效
        if cov1 is not None:
            # 检查cov1的数值范围
            cov_max = torch.abs(cov1).max(dim=-1)[0].max(dim=-1)[0]
            cov_min = torch.abs(cov1).min(dim=-1)[0].min(dim=-1)[0]
            stable_mask &= (cov_max < 1e4) & (cov_min > 1e-8)
            
            # 检查cov1的对称性
            cov_symm_diff = torch.abs(cov1 - cov1.transpose(-1, -2)).max(dim=-1)[0].max(dim=-1)[0]
            stable_mask &= (cov_symm_diff < 1e-5)
            
            # 检查cov1是否为正定矩阵
            try:
                det = torch.linalg.det(cov1)
                stable_mask &= (det > 1e-8)
            except:
                print("Error computing determinant")
                return new_loc, cov1  # 如果计算行列式失败，直接返回原始协方差

            # 检查cov1是否包含nan或inf
            stable_mask &= ~torch.any(torch.isnan(cov1).reshape(cov1.shape[0], -1), dim=1)
            stable_mask &= ~torch.any(torch.isinf(cov1).reshape(cov1.shape[0], -1), dim=1)

            # 只对稳定的样本进行特征值分解
            try:
                scale1_all = torch.zeros((cov1.shape[0], cov1.shape[1]), device=cov1.device)
                rot_matrix1_all = torch.zeros_like(cov1)
                
                if stable_mask.any():
                    scale1_stable, rot_matrix1_stable = torch.linalg.eigh(cov1[stable_mask])
                    scale1_all[stable_mask] = scale1_stable
                    rot_matrix1_all[stable_mask] = rot_matrix1_stable
                
                scale1 = scale1_all
                rot_matrix1 = rot_matrix1_all
            except Exception as e:
                print(f"Error in eigendecomposition: {e}")
                return new_loc, cov1  # 如果特征值分解失败，直接返回原始协方差

        assert scale1 is not None and rot_matrix1 is not None, "必须提供 scale1 和 rot_matrix1 或 cov1"

        # 2. 检查数值范围
        max_value = 1e4
        min_value = 1e-8
        
        # 检查scale1的范围
        stable_mask &= (scale1.abs().max(dim=-1)[0] < max_value)
        stable_mask &= (scale1.abs().min(dim=-1)[0] > min_value)
        
        # 检查velocity_cov的范围
        vel_cov_max = torch.abs(velocity_cov).max(dim=-1)[0].max(dim=-1)[0]
        stable_mask &= (vel_cov_max < max_value)
        
        # 3. 检查对称性
        vel_cov_symm_diff = torch.abs(velocity_cov - velocity_cov.transpose(-1, -2)).max(dim=-1)[0].max(dim=-1)[0]
        symm_threshold = 1e-5
        stable_mask &= (vel_cov_symm_diff < symm_threshold)
        
        # 4. 检查nan和inf
        stable_mask &= ~torch.any(torch.isnan(rot_matrix1).reshape(rot_matrix1.shape[0], -1), dim=1)
        stable_mask &= ~torch.any(torch.isnan(velocity_cov).reshape(velocity_cov.shape[0], -1), dim=1)
        stable_mask &= ~torch.any(torch.isinf(rot_matrix1).reshape(rot_matrix1.shape[0], -1), dim=1)
        stable_mask &= ~torch.any(torch.isinf(velocity_cov).reshape(velocity_cov.shape[0], -1), dim=1)

        # 如果没有稳定的点，直接返回原始协方差
        if not stable_mask.any():
            if cov1 is not None:
                return new_loc, cov1
            else:
                return new_loc, (torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2))))

        try:
            # 只对稳定的点计算新的协方差
            C_ij = rot_matrix1[stable_mask].transpose(1, 2).bmm(velocity_cov[stable_mask]).bmm(rot_matrix1[stable_mask])
            
            E_ij = scale1[stable_mask].unsqueeze(-1) + scale1[stable_mask].unsqueeze(-2)  # Bx3x3
            E_ij = C_ij/(E_ij + 1e-8)  # Bx3x3
            
            gamma = torch.bmm(rot_matrix1[stable_mask], torch.bmm(E_ij, rot_matrix1[stable_mask].transpose(1, 2)))
            
            cov = torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2)))
            
            # 初始化new_cov为原始协方差
            if cov1 is not None:
                new_cov = cov1.clone()  # 使用输入的cov1而不是计算的cov
            elif scale1 is not None and rot_matrix1 is not None:
                new_cov = (torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2)))).clone()
            else:
                raise ValueError("必须提供 scale1 和 rot_matrix1, 或 cov1")
            
            # 只更新稳定点的协方差
            new_cov[stable_mask] = (cov[stable_mask] + 
                                  velocity_cov[stable_mask] + 
                                  gamma.bmm(cov[stable_mask]).bmm(gamma.transpose(1, 2)))
            
        except Exception as e:
            print(f"Error in computation: {e}")
            return new_loc, cov1  # 如果计算失败，直接返回原始协方差
            
        return new_loc, new_cov




def matrix_sqrt(A, eps=1e-5):
    """
    计算矩阵的平方根
    """
    U, S, V = torch.svd(A)

    # # 检查是否有异常值
    # if torch.isnan(U).any() or torch.isinf(U).any() or torch.isnan(S).any() or torch.isinf(S).any() or torch.isnan(V).any() or torch.isinf(V).any():
    #     print("Found NaNs or Infs in U, S, or V")
    #     import pdb; pdb.set_trace()

    sqrt_A = torch.bmm(U, torch.bmm(S.sqrt().diag_embed(), V.transpose(-1, -2)))

    # 检查是否有异常值
    # if torch.isnan(sqrt_A).any() or torch.isinf(sqrt_A).any():
    #     print("Found NaNs or Infs in sqrt_A")
    #     import pdb; pdb.set_trace()

    return sqrt_A



def matrix_sqrt_H(A):
    """
    计算矩阵的平方根, Hermitian 矩阵
    """
    L, Q = torch.linalg.eigh(A)
    # sqrt_A = torch.bmm(Q, torch.bmm(L.sqrt().diag_embed(), Q.transpose(-1, -2)))

    #平方再开四次方
    sqrt_A = torch.bmm(Q, torch.bmm(((L**2)**0.25).diag_embed(), Q.transpose(-1, -2)))
    return sqrt_A


def symmetrize(A):
    return (A + A.transpose(-1, -2))/2

class WassersteinLog(nn.Module):
    def __init__(self):
        super(WassersteinLog, self).__init__()

    def forward(self, miu_1, miu_2, cov_1, cov_2):
        """
        Compute Wasserstein Log with stricter filtering
        """
        # 初始化零速度
        zero_miu_vel = torch.zeros_like(miu_2 - miu_1)
        zero_cov_vel = torch.zeros_like(cov_2 - cov_1)
        
        # 更严格的筛选条件
        stable_mask = torch.ones(miu_1.shape[0], dtype=torch.bool, device=miu_1.device)
        
        # 1. 更严格的值范围检查
        max_cov_value = 1e4  # 降低阈值
        min_cov_value = 1e-4  # 添加最小值阈值
        
        # 检查协方差矩阵的值范围
        cov1_max = torch.abs(cov_1).max(dim=-1)[0].max(dim=-1)[0]
        cov2_max = torch.abs(cov_2).max(dim=-1)[0].max(dim=-1)[0]
        cov1_min = torch.abs(cov_1).min(dim=-1)[0].min(dim=-1)[0]
        cov2_min = torch.abs(cov_2).min(dim=-1)[0].min(dim=-1)[0]
        
        stable_mask &= (cov1_max < max_cov_value) & (cov2_max < max_cov_value)
        stable_mask &= (cov1_min > min_cov_value) & (cov2_min > min_cov_value)
        
        # 2. 检查对称性
        cov1_symm_diff = torch.abs(cov_1 - cov_1.transpose(-1, -2)).max(dim=-1)[0].max(dim=-1)[0]
        cov2_symm_diff = torch.abs(cov_2 - cov_2.transpose(-1, -2)).max(dim=-1)[0].max(dim=-1)[0]
        symm_threshold = 1e-5
        stable_mask &= (cov1_symm_diff < symm_threshold) & (cov2_symm_diff < symm_threshold)
        
        # 3. 检查行列式，确保矩阵不是奇异的
        det_threshold = 1e-6
        try:
            det1 = torch.linalg.det(cov_1)
            det2 = torch.linalg.det(cov_2)
            stable_mask &= (det1.abs() > det_threshold) & (det2.abs() > det_threshold)
        except:
            pass
        
        # 4. 检查nan和inf
        stable_mask &= ~torch.any(torch.isnan(cov_1).view(cov_1.shape[0], -1), dim=1)
        stable_mask &= ~torch.any(torch.isnan(cov_2).view(cov_2.shape[0], -1), dim=1)
        stable_mask &= ~torch.any(torch.isinf(cov_1).view(cov_1.shape[0], -1), dim=1)
        stable_mask &= ~torch.any(torch.isinf(cov_2).view(cov_2.shape[0], -1), dim=1)
        
        # 5. 检查位置向量的范围
        max_pos_value = 1e3
        stable_mask &= (torch.abs(miu_1).max(dim=-1)[0] < max_pos_value)
        stable_mask &= (torch.abs(miu_2).max(dim=-1)[0] < max_pos_value)
        
        # 如果没有稳定的点，直接返回零速度
        if not stable_mask.any():
            return zero_miu_vel, zero_cov_vel
        
        try:
            # 只对稳定的点计算速度
            miu_velocity = miu_2[stable_mask] - miu_1[stable_mask]
            
            # 计算协方差速度
            L_A, Q_A = torch.linalg.eigh(symmetrize(cov_1[stable_mask]))
            L_A = torch.clamp(L_A, min=1e-4)  # 提高最小特征值阈值
            
            A_sqrt = torch.bmm(Q_A, torch.bmm(L_A.sqrt().diag_embed(), Q_A.transpose(-1, -2)))
            A_sqrt_inv = torch.bmm(Q_A, torch.bmm((1e-4+L_A.sqrt()).reciprocal().diag_embed(), Q_A.transpose(-1, -2)))
            
            C = A_sqrt.bmm(cov_2[stable_mask]).bmm(A_sqrt.transpose(-1, -2))
            C = symmetrize(C)
            C_sqrt = matrix_sqrt_H(C)
            
            cov_velocity = A_sqrt.bmm(C_sqrt).bmm(A_sqrt_inv.transpose(-1, -2))
            cov_velocity = cov_velocity + cov_velocity.transpose(-1, -2) - cov_1[stable_mask] - cov_1[stable_mask].transpose(-1, -2)
            
            # 将计算结果放回完整的张量中
            zero_miu_vel[stable_mask] = miu_velocity
            zero_cov_vel[stable_mask] = cov_velocity
            
        except Exception as e:
            print(f"Error in computation: {e}")
            # 发生错误时返回零速度
            pass
        
        return zero_miu_vel, zero_cov_vel




def matrix_log(A):
    """
    计算对称正定矩阵的矩阵对数
    """
    # 进行 Cholesky 分解
    L = torch.linalg.cholesky(A)
    # 计算 L 的对数
    log_L = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))
    # 重构对数矩阵
    log_A = L @ torch.diag_embed(log_L) @ L.transpose(-1, -2)
    return log_A



class WassersteinLog_stable(nn.Module):
    def __init__(self):
        super(WassersteinLog_stable, self).__init__()

    def forward(self, miu_1, miu_2, cov_1, cov_2):
        """
        计算 Wasserstein 对数映射 Log_{cov_1}(cov_2)
        """
        # 计算 cov_1 和 cov_2 的矩阵对数
        log_cov_1 = matrix_log(cov_1)
        log_cov_2 = matrix_log(cov_2)
        # 计算协方差的对数映射
        cov_velocity = log_cov_2 - log_cov_1
        return miu_2 - miu_1, cov_velocity




'''
融合两个高斯分布，返回新的高斯分布
如：
用卡尔曼滤波预测出来的新的高斯分布和当前帧的高斯分布融合

将输入改为协方差矩阵，而不是尺度和旋转矩阵 modified by Junli
'''
class GaussianMerge(nn.Module):
    def __init__(self,device="cuda"):
        super(GaussianMerge, self).__init__()

        # 使用神经网络预测K
        # self.nn_K = nn.Sequential(
        #     nn.Linear(18, 64),  # 输入维度为6x3=18 (cov1和cov2各3x3)
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 9)  # 输出维度为3x3=9
        # ).to(device)

    def forward(self, loc1, cov1, loc2, cov2):
        """
        merge two Gaussians
        loc1, loc2: Bx3
        scale1, scale2: Bx3
        rot_matrix1, rot_matrix2: Bx3x3
        """

        # 防止矩阵求逆出现奇异矩阵 modified by Junli
        epsilon = 1e-6  # 或更小的值，视情况而定

        # 通过计算得到K
        try:
            K = cov1.matmul((cov1 + cov2 + torch.eye(cov1.shape[-1]).to(cov1.device) * epsilon).inverse())
        except:
            print("cov1 contains None:", cov1 is None)
            print("cov2 contains None:", cov2 is None)
            # print("cov1:", cov1)
            # print("cov2:", cov2)
            import pdb; pdb.set_trace()

        # # 将协方差矩阵展平并拼接
        # cov_input = torch.cat([cov1.view(cov1.shape[0], -1), cov2.view(cov2.shape[0], -1)], dim=1)
        
        # # 使用神经网络预测K
        # K = self.nn_K(cov_input).view(cov1.shape)

        loc_new = loc1.unsqueeze(1) + (loc2.unsqueeze(1) - loc1.unsqueeze(1)).bmm(K.transpose(1, 2))
        loc_new = loc_new.squeeze(1)
        cov_new = cov1 + K.matmul(cov1)

        return loc_new, cov_new


def test_wasser(B, loc, scale, rot, wasserstein_distance=True):
    # B = 10  # 使用较小的batch size便于测试和验证
    # loc = torch.randn(B, 3)  # 位置数据 Bx3
    # scale = torch.exp(torch.randn(B, 3))  # 尺度数据，确保为正
    # rot = torch.randn(B, 4)  # 四元数表示的旋转 Bx4
    # rot = F.normalize(rot, p=2, dim=1)  # 规范化四元数

    # 转换四元数到旋转矩阵
    rot_matrix = quaternion_to_matrix(rot)  # 旋转矩阵 Bx3x3

    # Wasserstein Gaussian 测试
    if wasserstein_distance:
        wasserstein_gaussian = WassersteinGaussian()
        w_loss = wasserstein_gaussian(loc[:B//2], scale[:B//2], rot_matrix[:B//2], loc[B//2:], scale[B//2:], rot_matrix[B//2:])
        print("Wasserstein Gaussian Loss:", w_loss)

    # # Wasserstein Exp 测试
    # velocity = torch.randn(B, 3)  # 随机生成速度
    # velocity_cov = torch.eye(3).repeat(B, 1, 1)  # 使用单位矩阵作为速度协方差的简化表示
    # wasserstein_exp = WassersteinExp()
    
    # print("loc: ", loc.shape)
    # print("scale: ", scale.shape)
    # print("rot_matrix: ", rot_matrix.shape)
    # print("velocity: ", velocity.shape)
    # print("velocity_cov: ", velocity_cov.shape)


    # new_loc, new_cov = wasserstein_exp(loc[:B//2], scale[:B//2], rot_matrix[:B//2], velocity[:B//2], velocity_cov[:B//2])
    # print("New Locations after Wasserstein Exp:", new_loc)
    # print("New Covariance Matrices after Wasserstein Exp:", new_cov)

    # # Gaussian Merge 测试
    # # gaussian_merge = GaussianMerge()
    # # merged_loc, merged_cov = gaussian_merge(loc[:B//2], scale[:B//2], rot_matrix[:B//2], loc[B//2:], scale[B//2:], rot_matrix[B//2:])
    # ## 修改了 gaussian_merge 的输入，改为 loc 和 cov
    
    # print("Merged Locations:", merged_loc)
    # print("Merged Covariances:", merged_cov)


def kalman_filter_update(stage, prev_gaussian_params, wasserstein_exp, max_history):
    '''
    卡尔曼滤波求得当前帧的位置和协方差矩阵预测值

    参数:
    - stage: 当前处理的阶段，通常为 "fine"。
    - prev_gaussian_params: 前两帧的高斯参数，包含位置、尺度、旋转矩阵。
    - wasserstein_exp: 用于执行卡尔曼滤波更新的位置和协方差的函数。

    返回:
    - kalman_predicted_loc: 预测的当前帧位置。
    - kalman_predicted_cov: 预测的当前帧协方差矩阵。
    '''
    kalman_predicted_loc = None
    kalman_predicted_cov = None

    if stage == "fine" and \
        len(prev_gaussian_params[0]) == 3 and \
        len(prev_gaussian_params[1]) == 3 and \
        len(prev_gaussian_params) == max_history:

        # 倒数第二帧的参数（t-2）
        loc_t_minus_2 = prev_gaussian_params[0][0]
        scale_t_minus_2 = prev_gaussian_params[0][1]
        rot_quat_t_minus_2 = prev_gaussian_params[0][2]
        rot_matrix_t_minus_2 = quaternion_to_matrix(rot_quat_t_minus_2)  # 旋转矩阵 Bx3x3

        # 倒数第一帧的参数（t-1）
        loc_t_minus_1 = prev_gaussian_params[1][0]
        scale_t_minus_1 = prev_gaussian_params[1][1]
        rot_quat_t_minus_1 = prev_gaussian_params[1][2]
        rot_matrix_t_minus_1 = quaternion_to_matrix(rot_quat_t_minus_1)  # 旋转矩阵 Bx3x3

        # 注意：这里的输出将是当前帧（t）的预参数
        cov_t_minus_2 = torch.bmm(rot_matrix_t_minus_2, torch.bmm(torch.diag_embed(scale_t_minus_2), rot_matrix_t_minus_2.transpose(1, 2)))
        cov_t_minus_1 = torch.bmm(rot_matrix_t_minus_1, torch.bmm(torch.diag_embed(scale_t_minus_1), rot_matrix_t_minus_1.transpose(1, 2)))

        # 计算速度和速度协方差
        velocity = loc_t_minus_1 - loc_t_minus_2
        velocity_cov = cov_t_minus_1 - cov_t_minus_2  # 根据两帧的协方差差异计算速度协方差

        # 使用 WassersteinExp ，卡尔曼滤波预测当前帧的位置和协方差
        kalman_predicted_loc, kalman_predicted_cov = wasserstein_exp(loc_t_minus_1,             
                                                                    scale_t_minus_1, 
                                                                    rot_matrix_t_minus_1, 
                                                                    velocity, 
                                                                    velocity_cov) 

    return kalman_predicted_loc, kalman_predicted_cov




'''
3dgs 源码中的函数：通过输入的高斯分布的scaling和rotation，构建出协方差矩阵
注意：所有的 S 和 R 必须经���规范化，包括：
    S = torch.exp(S)
    R = F.normalize(R, p=2, dim=1)
'''
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    if r.shape[-1] == 3:  # 如果是 3x3 旋转矩阵 modified by Junli
        return r
    elif r.shape[-1] == 4:  # 如果是四元数
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
        q = r / norm[:, None]
        R = torch.zeros((q.size(0), 3, 3), device=r.device)
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R
    else:
        raise ValueError("Rotation input must be either 3x3 matrix or 4D quaternion")

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    # print("actual_covariance: ", actual_covariance.shape)
    # print("actual_covariance: ", actual_covariance)
    symm = strip_symmetric(actual_covariance)
    return symm,actual_covariance


'''
测试计算协方差矩阵的方式是否结果一致
'''
def test_cov(R,S):

    print("采用 3dgs 源码中的方式计算协方差矩阵")
    print("R:", R.shape)
    print("S:", S.shape)
    _, conv_3dgs = build_covariance_from_scaling_rotation(S, 1, R) # == get_covariance(
    # print("cov in 3dgs: ", conv_3dgs.shape )
    # print("cov in 3dgs: ", conv_3dgs )

    
    print("采用yihao的方式计算协方差矩阵")
    # rot = F.normalize(R, p=2, dim=1)  # 规范化四元数
    rot_matrix = quaternion_to_matrix(R)  # 旋转矩阵 Bx3x3
    scale = torch.square(S) # 尺度数据，确保为正
    # scale = torch.exp(S) # 尺度数据，确保为正
    print("rot_matrix: ", rot_matrix.shape)
    # print("scale: ", scale.shape)
    cov = torch.bmm(rot_matrix, torch.bmm(torch.diag_embed(scale), rot_matrix.transpose(1, 2))) # covariance matrix Bx3x3
    print("cov in my way: ", cov.shape)
    # print("cov in my way: ", cov)

    # 减法
    print("cov diff: ", torch.abs(cov.cuda() - conv_3dgs).max())

    if torch.abs(cov.cuda() - conv_3dgs).max() < 1e-5:
        print("cov matrix is the same.")
    else:
        print("cov matrix is different.")


def scipy_wasserstein(mean1, cov1, mean2, cov2):
    """使用SciPy计算Wasserstein距离"""
    diff = np.sum((mean1 - mean2)**2)
    covmean = sqrtm(np.dot(cov1, cov2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sqrt(diff + np.trace(cov1 + cov2 - 2*covmean))


def test_wasserstein_gaussian():
    wg = WassersteinGaussian()
    
    # 测试1：相同分布
    loc1 = torch.tensor([[0., 0., 0.]])
    scale1 = torch.tensor([[1., 1., 1.]])
    rot1 = torch.eye(3).unsqueeze(0)
    
    dist = wg(loc1, scale1, rot1, loc1, scale1, rot1)
    assert torch.allclose(dist, torch.tensor([0.]), atol=1e-6), "相同分布测试失败"
    
    # 测试2：只有均值不同
    loc2 = torch.tensor([[1., 1., 1.]])
    dist = wg(loc1, scale1, rot1, loc2, scale1, rot1)
    expected_dist = torch.sqrt(torch.tensor([3.]))
    assert torch.allclose(dist, expected_dist, atol=1e-6), "只有均值不同测试失败"
    
    # 测试3：与SciPy实现比较
    loc1_np = loc1.numpy()
    loc2_np = loc2.numpy()
    cov1_np = np.eye(3)
    cov2_np = np.eye(3)
    
    scipy_dist = scipy_wasserstein(loc1_np[0], cov1_np, loc2_np[0], cov2_np)
    assert np.allclose(dist.numpy(), scipy_dist, atol=1e-6), "与SciPy实现比较失败"
    
    # 测试4：对称性
    dist_ab = wg(loc1, scale1, rot1, loc2, scale1, rot1)
    dist_ba = wg(loc2, scale1, rot1, loc1, scale1, rot1)
    assert torch.allclose(dist_ab, dist_ba, atol=1e-6), "对称性测试失败"
    
    print("所有测试通过！")



'''
检查SVD分解后是否可以恢复原始协方差矩阵
'''
def check_svd_recovery(cov_p, tolerance=1e-5):
    """
    检查SVD分解后是否可以恢复原始协方差矩阵。
    
    参数:
    cov_p (torch.Tensor): 原始协方差矩阵,形状为 (B, 3, 3)
    tolerance (float): 允许的误差范围
    
    返回:
    bool: 如果可以恢复则返回True,否则返回False
    """
    # 执行特征值分解
    scale, rot_matrix = torch.linalg.eigh(cov_p)


    # diag_scale = torch.diag_embed(scale)

    # recon_cov_test = torch.bmm(rot_matrix, torch.bmm(diag_scale, rot_matrix.transpose(1, 2)))
    
    # 重建协方差矩阵
    _, recovered_cov = build_covariance_from_scaling_rotation(scale.sqrt(), 1, rot_matrix)
    # build_covariance_from_scaling_rotation 就是 get_covariance
    # get_covariance 需要rotation为四元数，故修改 build_rotation 函数，如果输入为旋转矩阵，则不进行四元数到旋转矩阵的转换
    
    # 检查是否可以恢复
    print("recovered_cov: ", recovered_cov)
    diff = torch.abs(recovered_cov - cov_p)
    print("diff: ", diff)
    max_diff = torch.max(diff)
    
    if max_diff > tolerance:
        print(f"最大差异: {max_diff.item()}, 超过容差 {tolerance}, cov_p 无法恢复")
        return False
    else:
        print(f"最大差异: {max_diff.item()}, 在容差 {tolerance} 范围内, cov_p 可以恢复")
        return True


    

if __name__ == "__main__":
    device = "cuda"
    B = 6  # 使用较小的batch size便于测试和验证
    loc = torch.randn(B, 3, device=device)  # 位置数据 Bx3
    R = torch.randn(B, 4, device=device)  # 四元数表示的旋转 Bx4
    R = F.normalize(R, p=2, dim=1)  # 规范化四元数
    S = torch.randn(B, 3, device=device)  # 尺度数据 Bx3
    S = torch.exp(S)  # 尺度数据，确保为正

    test_wasserstein_distance = True
    test_wasserstein_exp = False
    test_cov_SVD = False

    if test_wasserstein_distance:
        test_wasserstein_gaussian()

    '''
    测试wasserstein距离，wasserstein卡尔曼滤波，高斯分布融合
    '''
    if test_wasserstein_exp:
        test_wasser(B, loc, S, R)

    '''
    测试 yihao 方法和 3dgs源码中的方法计算协方差矩阵是否一致
    '''
    # test_cov(R,S)

    '''
    测试 SVD 分解后是否可以恢复原始协方差矩阵
    '''
    if test_cov_SVD:
        cov_p = torch.randn(B, 3, 3, device=device) ### 
        cov_p = cov_p.transpose(-1, -2).matmul(cov_p)
        check_svd_recovery(cov_p)
