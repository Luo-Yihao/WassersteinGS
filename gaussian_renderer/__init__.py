#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
from debug_utils import debug_print
from YiHao_utils import GaussianMerge
from utils.general_utils import strip_symmetric

def render(viewpoint_camera, 
           pc : GaussianModel, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None, 
           stage="fine", 
           cam_type=None,
           screenspace_points_from_outside = None,
           means3D_canonical = None,
           shs_canonical = None,
           predicted_loc = None,
           predicted_cov = None,
           opacity_final = None,
           shs_final = None,
           view_time = None
           ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = None
    if screenspace_points_from_outside is None:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    else:
        screenspace_points = screenspace_points_from_outside
    
    try:
        screenspace_points.retain_grad()
        # print("viewspace_points(screenspace_points) retain_grad success!!")
    except:
        # print("viewspace_points(screenspace_points) retain_grad failed~~")
        # print("viewpoint_camera.time",viewpoint_camera.time)
        pass

    # Set up rasterization configuration
    # if stage == "fine":
    #     import pdb ; pdb.set_trace()
    #     debug_print("check render:")
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )

        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    scales_final = None
    rotations_final = None

    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc._scaling
    #     rotations = pc._rotation

    scales = pc._scaling
    rotations = pc._rotation

    deformation_point = pc._deformation_table


    # import pdb ; pdb.set_trace()

    if "coarse" in stage \
        and predicted_loc is  None \
        and predicted_cov is  None \
        and opacity_final is None \
        and shs_final is None :
        # coarse训练阶段

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs

        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)
    
    elif "fine" in stage \
        and predicted_loc is  None \
        and predicted_cov is  None \
        and opacity_final is None \
        and shs_final is None :
        # 4dgs origin fine训练阶段

        if view_time is not None:
            time = torch.tensor(view_time).to(means3D.device).repeat(means3D.shape[0],1)

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time) 
        # print("time:", time.shape, time)               
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)

    elif "fine" in stage and \
        predicted_loc is not None and \
        predicted_cov is not None and \
        opacity_final is None and \
        shs_final is None:
        # yihao 3 batch kalman 推理阶段
        
        if view_time is not None:
            time = torch.tensor(view_time).to(means3D.device).repeat(means3D.shape[0],1)
        
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time) 
        
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)
       

    else: # yihao 3 batch kalman 滤波时选用，在外部使用 deformation

        # 试用预先计算好的 位置，协方差矩阵，opacity，shs
        means3D_final = predicted_loc
        opacity = pc.opacity_activation(opacity_final)
        # scales_final = pc.scaling_activation(scales_final)
        # rotations_final = pc.rotation_activation(rotations_final)
        # opacity = opacity_final
        shs_final = shs_final


    if pipe.compute_cov3D_python:
        # 根据 offset 形变后的 rotation 和 scaling 计算最终协方差矩阵
        _, cov3D_precomp_matrix = pc.get_covariance(scaling_modifier, scales_final, rotations_final)
        scale_final_store = scales_final
        rotation_final_store = rotations_final

        scales_final = None
        rotations_final = None

        # 如果送入了localtion和cov，就使用传入的
        if stage == "fine" and \
           predicted_loc is not None and \
           predicted_cov is not None:

            # print("kalman_updated_loc:",kalman_updated_loc)
            # print("kalman_updated_cov:",kalman_updated_cov)
            
            means3D_final, cov3D_precomp_matrix = predicted_loc, predicted_cov


        elif stage == "coarse" and predicted_loc is not None and predicted_cov is not None:
            # 在 coarse 阶段不使用卡尔曼滤波更新参数
            pass
        elif stage == "fine":
            # 不使用卡尔曼滤波更新参数
            # 在这种情况下,直接使用当前帧的高斯参数,不进行融合
            pass
        # 将协方差矩阵转换为（batch_size, 6）的形状
        # print("cov3D_precomp:", cov3D_precomp.shape)
        try:
            cov3D_precomp = strip_symmetric(cov3D_precomp_matrix)
        except:
            print("cov3D_precomp:", cov3D_precomp.shape)
            import pdb ; pdb.set_trace()

    else:
        conv3D_precomp = None

    # time2 = get_time()
    # print("asset value:",time2-time1)
    
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            if stage == "fine" and means3D_canonical is not None and shs_canonical is not None:
                shs_view = shs_canonical.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (means3D_canonical - viewpoint_camera.camera_center.cuda().repeat(shs_canonical.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else: # 4dgs origin
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    try:    
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)
    except:
        print("rasterizer error")
        print("means3D_final:", means3D_final.shape)
        print("means2D:", means2D.shape)
        print("shs_final:", shs_final.shape)
        # print("colors_precomp:", colors_precomp.shape)
        print("opacity:", opacity.shape)
        # print("scales_final:", scales_final.shape)
        # print("rotations_final:", rotations_final.shape)
        print("cov3D_precomp:", cov3D_precomp.shape)
        import pdb ; pdb.set_trace()
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "means3D_final":means3D_final,
            "scales_final":scale_final_store,
            "rotations_final":rotation_final_store,
            "cov3D_precomp":cov3D_precomp_matrix
            }

