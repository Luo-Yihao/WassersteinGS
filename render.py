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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
import torch.nn.functional as F
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix

from Wasserstein_utils import WassersteinExp, GaussianMerge, kalman_filter_update
from Gaussian_Updating import  wass_training_step

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    '''
    初始化WassersteinExp,GaussianMerge,prev_gaussian_params
    '''
    wasserstein_exp = WassersteinExp()
    gaussian_merge = GaussianMerge()
    prev_gaussian_params = [] # 上两帧的高斯参数,用于卡尔曼滤波
    max_history = 2

    print("point nums:",gaussians._xyz.shape[0])

    time_division = 1  # 默认为1，处以10是为了保持训练同步，小时间间隔


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        time1 = time()
        if pipeline.kalman_filter == True:
            if idx < 2:
                # For the first two frames, render normally and store the Gaussian parameters
                print("----------original------------------------------")
                print("view.time:", view.time)
                print("view.time/time_division:", view.time/time_division)
                render_pkg = render(view, gaussians, pipeline, 
                                    background, cam_type=cam_type, 
                                    view_time=(view.time/time_division))
                rendering = render_pkg["render"]
                # Extract the Gaussian parameters from the render_pkg
                observed_means = render_pkg["means3D_final"]
                observed_covs = render_pkg["cov3D_precomp"]

                # Store the observed parameters along with the timestamp
                prev_gaussian_params.append({
                    "means": observed_means,
                    "covs": observed_covs,
                    "timestamp": torch.tensor(view.time/time_division).to(observed_means.device).repeat(observed_means.shape[0],1)
                })

            else:
                # For frames from idx >= 2, perform Kalman filtering
                prev_param_1 = prev_gaussian_params[-2]
                prev_param_2 = prev_gaussian_params[-1]

                delta_t1 = prev_param_1["timestamp"]
                delta_t2 = prev_param_2["timestamp"]
                delta_t3 = torch.tensor(view.time/time_division).to(prev_param_1["timestamp"].device).repeat(prev_param_1["timestamp"].shape[0],1) 

                time_ratio = (delta_t3 - delta_t2) / (delta_t2 - delta_t1 + 1e-8)

                velocity = time_ratio * (prev_param_2["means"] - prev_param_1["means"])
                velocity_cov = time_ratio.unsqueeze(2) * (prev_param_2["covs"] - prev_param_1["covs"])
                predicted_means, predicted_covs = wasserstein_exp(
                    loc=prev_param_2["means"],
                    cov1=prev_param_2["covs"],
                    velocity=velocity,
                    velocity_cov=velocity_cov
                )

                render_pkg_observed = render(view, gaussians, pipeline, 
                                             background, cam_type=cam_type, 
                                             view_time=(view.time/time_division))
                observed_means_current = render_pkg_observed["means3D_final"]
                observed_covs_current = render_pkg_observed["cov3D_precomp"]

                ## observed 和 predicted 的 diff
                diff_means = observed_means_current - predicted_means
                diff_covs = observed_covs_current - predicted_covs

                # Merge the predicted parameters with the observed parameters
                merged_means, merged_covs = gaussian_merge(
                    predicted_means, predicted_covs,
                    observed_means_current, observed_covs_current
                )

                # Render using the [merged] parameters
                render_pkg = render(
                    view, gaussians, pipeline, background, cam_type=cam_type,
                    predicted_loc=merged_means, predicted_cov=merged_covs, 
                    view_time=(view.time/time_division)
                )
                rendering = render_pkg["render"] # 用kalman滤波后，再融合的参数渲染！！

                # Update prev_gaussian_params with the merged parameters
                prev_gaussian_params.append({
                    "means": observed_means_current.clone().detach(),
                    "covs": observed_covs_current.clone().detach(),
                    "timestamp": delta_t3.clone().detach()
                })

                # Keep only the last two entries to maintain the history
                if len(prev_gaussian_params) > 2:
                    prev_gaussian_params.pop(0)

        else:
            # Original rendering without Kalman filtering
            render_pkg = render(view, gaussians, pipeline, background, cam_type=cam_type)
            rendering = render_pkg["render"]

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        # 保存render
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:03d}'.format(idx) + ".png"))
        print("save render to ", os.path.join(render_path, '{0:03d}'.format(idx) + ".png"))
        # 保存gt    
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    '''
    保存视频
    '''
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
    print("output video to ", os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'))

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)

        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
            
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)