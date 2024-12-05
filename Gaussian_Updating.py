import torch
from einops import rearrange
from pytorch3d.transforms import quaternion_to_matrix

from utils.loss_utils import l1_loss
from gaussian_renderer import render
from Wasserstein_utils import WassersteinGaussian, WassersteinExp, GaussianMerge, WassersteinLog, WassersteinLog_stable


def wass_training_step(gaussians, 
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
    Kalman filter iteration
    Args:
        gaussians: Gaussian model
        viewpoint_cams: Camera viewpoints and time information, grouped in consecutive three frames
        scene: Scene object
        pipe: Parameters
        background: Background
        stage: Current stage
        gaussian_merge: Gaussian merge function
        wasserstein_exp: Kalman prediction function
        wasserstein_distance: Wasserstein distance function
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

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
            means3D.repeat(3,1), scales.repeat(3,1), rotations.repeat(3,1), 
            opacity.repeat(3,1), shs.repeat(3,1,1), timestamp_batch
        ) 

        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        rot_matrix_final = quaternion_to_matrix(rotations_final)

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


        ### log && exp wasserstein geodesic ，compute velocity and covariance velocity ------------------
        ## log_velocity
        # try:
        #     velocity, velocity_cov = WassersteinLog()(means3D_final[1], means3D_final[0], cov3D_precomp[1], cov3D_precomp[0])
        # except:
        #     velocity, velocity_cov = WassersteinLog_stable()(means3D_final[1], means3D_final[0], cov3D_precomp[1], cov3D_precomp[0])


        # velocity, velocity_cov = WassersteinLog_stable()(means3D_final[1], means3D_final[0], cov3D_precomp[1], cov3D_precomp[0])
        velocity, velocity_cov = WassersteinLog()(means3D_final[1], means3D_final[0], cov3D_precomp[1], cov3D_precomp[0])        
        
        velocity = -velocity    
        velocity_cov = -velocity_cov

        # predict_mean3D_2, predict_cov3D_2 = wasserstein_exp(loc=means3D_final[1], 
        #                                                     scale1=scales_final[1]**2, 
        #                                                     rot_matrix1=rot_matrix_final[1], 
        #                                                     velocity=velocity, 
        #                                                     velocity_cov=velocity_cov)
        
        ### log && exp wasserstein geodesic ，compute velocity and covariance velocity ------------------

        ### Linear acceleration and prediction --------------------------------
        # velocity = (timestamp_batch[2]-timestamp_batch[1])/(timestamp_batch[1]-timestamp_batch[0]+1e-8)*(means3D_final[1]-means3D_final[0])
        # velocity_cov = ((timestamp_batch[2]-timestamp_batch[1])/(timestamp_batch[1]-timestamp_batch[0]+1e-8)).unsqueeze(2)*(cov3D_precomp[1]-cov3D_precomp[0])

        predict_mean3D_2, predict_cov3D_2 = wasserstein_exp(loc=means3D_final[1], 
                                                    scale1=scales_final[1]**2, 
                                                    rot_matrix1=rot_matrix_final[1], 
                                                    velocity=velocity, 
                                                    velocity_cov=velocity_cov)
        ### Linear acceleration and prediction ---------------------------------------------------
        

        loc_merge_gaussian_2, cov_merge_gaussian_2 = gaussian_merge(predict_mean3D_2, predict_cov3D_2, means3D_final[2], cov3D_precomp[2])

        loss_cross = wasserstein_distance(means3D_final[2], scales_final[2]**2, rot_matrix_final[2], predict_mean3D_2, cov2=predict_cov3D_2).mean()

        # calculate Wasserstein distance loss
        loss_inter_frame_1_2 = wasserstein_distance(means3D_final[1], scales_final[1]**2, rot_matrix_final[1], 
                                                    means3D_final[2], scale2=scales_final[2]**2, rot_matrix2=rot_matrix_final[2]).mean()
        
        loss_inter_frame_0_1 = wasserstein_distance(means3D_final[0], scales_final[0]**2, rot_matrix_final[0], 
                                                    means3D_final[1], scale2=scales_final[1]**2, rot_matrix2=rot_matrix_final[1]).mean()
        
        # add these two loss to the total loss
        loss_cross += loss_inter_frame_1_2 + loss_inter_frame_0_1
        
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


        return viewspace_point_tensor, visibility_filter, radii, image_pkg, loss_pkg 

    return None, None, None, None, None # if stage is not fine


if __name__ == "__main__":
    pass