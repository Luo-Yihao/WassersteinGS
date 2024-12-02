# Required Imports
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
import matplotlib.pyplot as plt

from YiHao_utils import WassersteinGaussian, WassersteinExp, GaussianMerge, kalman_filter_update, WassersteinLog
from Junli_utils import create_consecutive_groups, get_random_group
from pytorch3d.transforms import quaternion_to_matrix
from debug_utils import debug_print
import torch.nn.functional as F

from einops import rearrange




def kalman_filter_training_step(gaussians, 
                                viewpoint_cams, 
                                scene, 
                                pipe, 
                                background, 
                                stage, 
                                iteration,
                                wasserstein_exp: WassersteinExp,
                                wasserstein_distance: WassersteinGaussian,
                                gaussian_merge: GaussianMerge):
    """
    卡尔曼滤波迭代
    gaussians: 高斯模型
    viewpoint_cams: 相机视角以及时间信息，以连续的三帧为一组 viewpoint_cams
    scene: 场景
    pipe: 参数
    background: 背景
    stage: 阶段
    gaussian_merge: 高斯合并函数
    wasserstein_exp: 卡尔曼预测函数
    wasserstein_distance: wasserstein距离函数
    """
    if stage == "fine":
        pc = gaussians
        means3D = pc._xyz
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity
        shs = pc.get_features

        screenspace_points = torch.zeros_like(means3D, 
                                              dtype=means3D.dtype, 
                                              requires_grad=True, 
                                              device="cuda") + 0

        time_list = []
        gt_image_list = []
        for viewpoint_cam in viewpoint_cams:
            timestamp = torch.tensor(viewpoint_cam.time).to(means3D.device).repeat(means3D.shape[0],1)
            time_list.append(timestamp)
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam['image'].cuda()
            gt_image_list.append(gt_image.unsqueeze(0))

        timestamp_batch = torch.cat(time_list, dim=0)  # 2000*3 , 1
        gt_image = torch.cat(gt_image_list, dim=0)

        assert torch.isfinite(means3D).all(), "means3D 包含 Inf 或 NaN"
        assert torch.isfinite(scales).all(), "scales 包含 Inf 或 NaN"
        assert torch.isfinite(rotations).all(), "rotations 包含 Inf 或 NaN"
        assert torch.isfinite(opacity).all(), "opacity 包含 Inf 或 NaN"
        assert torch.isfinite(shs).all(), "shs 包含 Inf 或 NaN"
        assert torch.isfinite(timestamp_batch).all(), "timestamp_batch 包含 Inf 或 NaN"

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
            means3D.repeat(3,1), scales.repeat(3,1), rotations.repeat(3,1), 
            opacity.repeat(3,1), shs.repeat(3,1,1), timestamp_batch
        ) 


        ## 只测试第三个 time 做 deformation
        # means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
        #     means3D, scales, rotations, opacity, shs, time_list[2]
        # ) 
        # if iteration % 100 == 0:
        #     print("time:", time_list[2])

        scales_final = pc.scaling_activation(scales_final)
        
        rotations_final = pc.rotation_activation(rotations_final)
        # print("rotations_final after pc.rotation_activation:", rotations_final)
        # # import pdb ; pdb.set_trace()

        # inf_mask = torch.isinf(rotations_final)
        # if inf_mask.any():
        #     print("rotations_final 包含 inf 值")
        #     inf_indices = torch.nonzero(inf_mask)
        #     print("inf 值的位置：", inf_indices)
        # else:
        #     print("rotations_final 不包含 inf 值")

        # nan_mask = torch.isnan(rotations_final)
        # if nan_mask.any():
        #     print("rotations_final 包含 nan 值")
        #     nan_indices = torch.nonzero(nan_mask)
        #     print("nan 值的位置：", nan_indices)
        # else:
        #     print("rotations_final 不包含 nan 值")

        # print("rotations_final 统计信息:")
        # print("最小值:", rotations_final.min())
        # print("最大值:", rotations_final.max())
        # print("平均值:", rotations_final.mean())
        # print("标准差:", rotations_final.std())
        # print("包含 inf 的元素数量:", torch.isinf(rotations_final).sum().item())
        # print("包含 NaN 的元素数量:", torch.isnan(rotations_final).sum().item())

        rot_matrix_final = quaternion_to_matrix(rotations_final)

        assert torch.isfinite(rotations_final).all(), "rotations_final 包含 Inf 或 NaN"
        assert torch.isfinite(rot_matrix_final).all(), "rot_matrix_final 包含 Inf 或 NaN"
        assert torch.isfinite(scales_final).all(), "scales_final 包含 Inf 或 NaN"
        assert (scales_final > 0).all(), "scales_final 包含非正值"
        assert torch.isfinite(means3D_final).all(), "means3D_final 包含 Inf 或 NaN"
        assert torch.isfinite(opacity_final).all(), "opacity_final 包含 Inf 或 NaN"
        assert torch.isfinite(shs_final).all(), "shs_final 包含 Inf 或 NaN"

        # print("scales_final.shape:", scales_final.shape)
        # print("rotations_final.shape:", rotations_final.shape)

        assert torch.isfinite(scales_final).all(), "scales_final 包含 Inf 或 NaN"
        assert (scales_final > 0).all(), "scales_final 包含非正值"

        _, cov3D_precomp = pc.get_covariance(1.0, scales_final, rotations_final)


        # Reshape tensors
        means3D_final = rearrange(means3D_final, '(t b) c -> t b c', t=3)
        scales_final = rearrange(scales_final, '(t b) c -> t b c', t=3)
        rotations_final = rearrange(rotations_final, '(t b) c -> t b c', t=3)
        opacity_final = rearrange(opacity_final, '(t b) c -> t b c', t=3)
        shs_final = rearrange(shs_final, '(t b) s c -> t b s c', t=3)
        cov3D_precomp = rearrange(cov3D_precomp, '(t b) h w -> t b h w', t=3)
        timestamp_batch = rearrange(timestamp_batch, '(t b) c -> t b c', t=3)
        rot_matrix_final = rearrange(rot_matrix_final, '(t b) h w -> t b h w', t=3)

        # print("timestamp_batch:", timestamp_batch.shape)
        # print("means3D_final:", means3D_final.shape)
        # print("scales_final:", scales_final.shape)
        # print("rot_matrix_final:", rot_matrix_final.shape)
        # print("cov3D_precomp:", cov3D_precomp.shape)
        # print("opacity_final:", opacity_final.shape)
        # print("shs_final:", shs_final.shape)    

        '''
        torch.bmm(rot_matrix_final[2], torch.bmm(torch.diag_embed(scales_final[2])**2, rot_matrix_final[2].transpose(1, 2)))
        
        '''
        # 平方

        # print("train kalman delta_t1:", timestamp_batch[0])
        # print("train kalman delta_t2:", timestamp_batch[1])
        # print("train kalman delta_t3:", timestamp_batch[2])


        # print("time_ratio:", (timestamp_batch[2]-timestamp_batch[1])/(timestamp_batch[1]-timestamp_batch[0]+1e-8))


        # velocity = (timestamp_batch[2]-timestamp_batch[1])/(timestamp_batch[1]-timestamp_batch[0]+1e-8)*(means3D_final[1]-means3D_final[0])
        # velocity_cov = ((timestamp_batch[2]-timestamp_batch[1])/(timestamp_batch[1]-timestamp_batch[0]+1e-8)).unsqueeze(2)*(cov3D_precomp[1]-cov3D_precomp[0])

        ## log_velocity
        velocity, velocity_cov = WassersteinLog()(means3D_final[1], means3D_final[0], cov3D_precomp[1], cov3D_precomp[0])
        velocity = -velocity    
        velocity_cov = -velocity_cov

        time_diff1 = timestamp_batch[1] - timestamp_batch[0]
        time_diff2 = timestamp_batch[2] - timestamp_batch[1]
        assert time_diff1.abs().sum() > 1e-8, "时间差过小，可能导致除以零"
        assert time_diff2.abs().sum() > 1e-8, "时间差过小，可能导致除以零"

        # print("train kalman velocity:", velocity)
        # print("train kalman velocity_cov:", velocity_cov)
        if torch.isnan(velocity).any():
            print("velocity is nan")
            import pdb ; pdb.set_trace()
        if torch.isnan(velocity_cov).any():
            print("velocity_cov is nan")
            import pdb ; pdb.set_trace()
            print("cov3D_precomp[1]:", cov3D_precomp[1])
            print("cov3D_precomp[0]:", cov3D_precomp[0])

        predict_mean3D_2, predict_cov3D_2 = wasserstein_exp(loc=means3D_final[1], 
                                                            scale1=scales_final[1]**2, 
                                                            rot_matrix1=rot_matrix_final[1], 
                                                            velocity=velocity, 
                                                            velocity_cov=velocity_cov)
        
        assert torch.isfinite(predict_mean3D_2).all(), "predict_mean3D_2 包含 Inf 或 NaN"
        assert torch.isfinite(predict_cov3D_2).all(), "predict_cov3D_2 包含 Inf 或 NaN"


        # print("_____________________________________________________    ")
        # diff_means = means3D_final[2] - predict_mean3D_2
        # diff_covs = cov3D_precomp[2] - predict_cov3D_2
        # print("diff_means between observed and predicted:", diff_means[:2,:])
        # print("diff_covs between observed and predicted:", diff_covs[:2,:,:])

        loc_merge_gaussian_2, cov_merge_gaussian_2 = gaussian_merge(predict_mean3D_2, predict_cov3D_2, means3D_final[2], cov3D_precomp[2])

        # print("predict_mean3D_2:", predict_mean3D_2.shape)
        # print("predict_cov3D_2:", predict_cov3D_2.shape)
        # print("loc_merge_gaussian_2:", loc_merge_gaussian_2.shape)
        # print("cov_merge_gaussian_2:", cov_merge_gaussian_2.shape)

        # import pdb ; pdb.set_trace()

        # 想办法把梯度传到train.py 
        loss_cross = wasserstein_distance(means3D_final[2], scales_final[2]**2, rot_matrix_final[2], predict_mean3D_2, cov2=predict_cov3D_2).mean()

        # 计算帧间的 Wasserstein distance loss
        loss_inter_frame_1_2 = wasserstein_distance(means3D_final[1], scales_final[1]**2, rot_matrix_final[1], 
                                                    means3D_final[2], scale2=scales_final[2]**2, rot_matrix2=rot_matrix_final[2]).mean()
        
        loss_inter_frame_0_1 = wasserstein_distance(means3D_final[0], scales_final[0]**2, rot_matrix_final[0], 
                                                    means3D_final[1], scale2=scales_final[1]**2, rot_matrix2=rot_matrix_final[1]).mean()
        
        # 将这两个loss加入到总的loss中
        loss_cross += loss_inter_frame_1_2 + loss_inter_frame_0_1
        
        # 确保计算结果不包含 Inf 或 NaN
        assert torch.isfinite(loss_inter_frame_1_2).all(), "loss_inter_frame_1_2 包含 Inf 或 NaN"
        assert torch.isfinite(loss_inter_frame_0_1).all(), "loss_inter_frame_0_1 包含 Inf 或 NaN"

        # print("means3D_final[2]:", means3D_final[2].shape)
        # print("scales_final[2]:", scales_final[2].shape)
        # print("rot_matrix_final[2]:", rot_matrix_final[2].shape)
        # print("predict_mean3D_2:", predict_mean3D_2.shape)
        # print("predict_cov3D_2:", predict_cov3D_2.shape)
        
        assert torch.isfinite(wasserstein_distance(means3D_final[2], scales_final[2]**2, rot_matrix_final[2], predict_mean3D_2, cov2=predict_cov3D_2)).all(), "loss_cross 包含 Inf 或 NaN"

        # Render images
        render_image_obv_0 = render(viewpoint_cams[0], pc, pipe, background, stage=stage, 
                                    cam_type=scene.dataset_type,
                                    screenspace_points_from_outside=screenspace_points,
                                    means3D_canonical=means3D,
                                    shs_canonical=shs,
                                    predicted_loc=means3D_final[0], predicted_cov=cov3D_precomp[0],
                                    opacity_final=opacity_final[0], shs_final=shs_final[0])["render"]

        render_image_obv_1 = render(viewpoint_cams[1], pc, pipe, background, stage=stage, 
                                    cam_type=scene.dataset_type,
                                    screenspace_points_from_outside=screenspace_points,
                                    means3D_canonical=means3D,
                                    shs_canonical=shs,
                                    predicted_loc=means3D_final[1], predicted_cov=cov3D_precomp[1],
                                    opacity_final=opacity_final[1], shs_final=shs_final[1])["render"]

        # 测试第二个时间点
        render_image_obv_2_pkg = render(viewpoint_cams[2], pc, pipe, background, stage=stage, 
                                        cam_type=scene.dataset_type,
                                        screenspace_points_from_outside=screenspace_points,
                                        means3D_canonical=means3D,
                                        shs_canonical=shs,
                                        predicted_loc=means3D_final[2], predicted_cov=cov3D_precomp[2],
                                        opacity_final=opacity_final[2], shs_final=shs_final[2])

        render_image_obv_2 = render_image_obv_2_pkg["render"]
        viewspace_point_tensor = render_image_obv_2_pkg["viewspace_points"]
        visibility_filter = render_image_obv_2_pkg["visibility_filter"]
        radii = render_image_obv_2_pkg["radii"]

        render_image_pred_2 = render(viewpoint_cams[2], pc, pipe, background, stage=stage, 
                                     cam_type=scene.dataset_type,
                                     screenspace_points_from_outside=screenspace_points,
                                     means3D_canonical=means3D,
                                     shs_canonical=shs,
                                     predicted_loc=predict_mean3D_2, predicted_cov=predict_cov3D_2,
                                     opacity_final=opacity_final[2], shs_final=shs_final[2])["render"]

        render_image_merge_2 = render(viewpoint_cams[2], pc, pipe, background, stage=stage, 
                                      cam_type=scene.dataset_type,
                                      screenspace_points_from_outside=screenspace_points,
                                      means3D_canonical=means3D,
                                      shs_canonical=shs,
                                      predicted_loc=loc_merge_gaussian_2, predicted_cov=cov_merge_gaussian_2,
                                      opacity_final=opacity_final[2], shs_final=shs_final[2])["render"]

        # Calculate losses
        loss_render_obv_0 = l1_loss(render_image_obv_0.unsqueeze(0), gt_image[0].unsqueeze(0)) 
        loss_render_obv_1 = l1_loss(render_image_obv_1.unsqueeze(0), gt_image[1].unsqueeze(0)) 
        loss_render_obv_2 = l1_loss(render_image_obv_2.unsqueeze(0), gt_image[2].unsqueeze(0)) 
        loss_render_pred_2 = l1_loss(render_image_pred_2.unsqueeze(0), gt_image[2].unsqueeze(0)) 
        loss_render_merge_2 = l1_loss(render_image_merge_2.unsqueeze(0), gt_image[2].unsqueeze(0)) 

        # loss = loss_render_obv_0 + loss_render_obv_1 + loss_render_obv_2 + loss_render_pred_2 + loss_render_merge_2 + loss_cross

        # if iteration % 100 == 0:
        #     plt.clf()
        #     visualize_image(gt_image, 
        #                     render_image_obv_0, 
        #                     render_image_obv_1, 
        #                     render_image_obv_2, 
        #                     render_image_pred_2, 
        #                     render_image_merge_2)

        # 返回所有render image 和 gt image
        image_pkg = {
            "render_image_obv_zero": render_image_obv_0, # torch.zeros_like(render_image_obv_2), # render_image_obv_0,
            "render_image_obv_one": render_image_obv_1, # torch.zeros_like(render_image_obv_2), # render_image_obv_1,
            "render_image_obv_two": render_image_obv_2,
            "render_image_pred_two": render_image_pred_2,
            "render_image_merge_two": render_image_merge_2,
            "gt_image": gt_image
        }

        loss_pkg = {    
            "loss_render_obv_zero": loss_render_obv_0,
            "loss_render_obv_one": loss_render_obv_1,
            "loss_render_obv_two": loss_render_obv_2,
            "loss_render_pred_two": loss_render_pred_2,
            "loss_render_merge_two": loss_render_merge_2,
            "loss_cross": loss_cross
        }

        # 保存用于计算loss_cross的高斯参数
        # gaussian_params = {
        #     "means3D": means3D,
        #     "scales": scales,
        #     "rotations": rotations,
        #     "opacity": opacity,
        #     "shs": shs,
        #     "means3D_final": means3D_final,
        #     "scales_final": scales_final,
        #     "rotations_final": rotations_final,
        #     "rot_matrix_final": rot_matrix_final,
        #     "opacity_final": opacity_final,
        #     "shs_final": shs_final,
        #     "predict_mean3D_2": predict_mean3D_2,
        #     "predict_cov3D_2": predict_cov3D_2,
        #     "loc_merge_gaussian_2": loc_merge_gaussian_2,
        #     "cov_merge_gaussian_2": cov_merge_gaussian_2
        # }

        # 修改返回值，添加gaussian_params
        return viewspace_point_tensor, visibility_filter, radii, image_pkg, loss_pkg #gaussian_params
        

        return viewspace_point_tensor, visibility_filter, radii, image_pkg # , loss_pkg

    return None, None, None, None, None


if __name__ == "__main__":
    pass