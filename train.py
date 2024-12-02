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
import numpy as np
import random
import os, sys
import torch
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
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

from YiHao_utils import WassersteinGaussian, \
                        WassersteinExp, \
                        GaussianMerge, \
                        kalman_filter_update

from YiHao_kalman_iteration import kalman_filter_training_step

from Junli_utils import create_consecutive_groups, get_random_group, create_random_ordered_groups



from pytorch3d.transforms import quaternion_to_matrix
from debug_utils import debug_print, training_report_wandb
import torch.nn.functional as F
import wandb

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def process_image(img_tensor):
    # 约束范围
    # img_tensor = torch.clamp(img_tensor, 0, 1)

    # 将张量转换为numpy数组,并确保值在0-255范围内
    img_np = (torch.clamp(img_tensor.clone().detach(), 0, 1).cpu().numpy() * 255).astype(np.uint8)
    # 确保通道在最后一维
    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)
    return img_np

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer,
                         wasserstein_loss
                         ):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
        debug_print("copy viewpoint_stack into temp_list.")
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        debug_print("viewpoint_stack = scene.getTrainCameras()")
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
            debug_print("FineSampler is used.")
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
            debug_print("reset dataloader into random dataloader.")
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
        print("stage coarse, zerostamp_init")
    else:
        load_in_memory = False 
                            # 
    count = 0
    prev_gaussian_params = []
    max_history = 2
    wasserstein_exp = WassersteinExp()
    gaussian_merge = GaussianMerge()
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    group_size = 3
    groups = create_consecutive_groups(viewpoint_stack, group_size)
    group = []
    viewpoint_cam = None


    for iteration in range(first_iter, final_iter+1):       

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    # viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=False,num_workers=32,collate_fn=list)
                    # debug_print("reset dataloader into NOT random dataloader.")
                    random_loader = True
                    debug_print("reset dataloader into random dataloder in dynerf's branch.")
                loader = iter(viewpoint_stack_loader)
        else:
            idx = 0
            viewpoint_cams = []

            # *********************************************
            # 在训练循环中使用
            while idx < batch_size :    
                # import pdb; pdb.set_trace()
                # print("idx = ", idx)
                if not groups:
                    groups = create_random_ordered_groups(viewpoint_stack, group_size)  # 每次重新创建组

                if len(group) == 0:
                    group = get_random_group(groups)
                    groups.remove(group)
                    viewpoint_cam = group.pop(0)
                else:
                    viewpoint_cam = group.pop(0)
                # print("len(group) = ", len(group))
                
                viewpoint_cams.append(viewpoint_cam)
                # print("len(viewpoint_cams) = ", len(viewpoint_cams))
                idx += 1
            # *********************************************
            if len(viewpoint_cams) == 0:
                continue

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        viewspace_point_tensor = None
        image = None

        # loss权重
        lambda_loss_render_obv_zero = torch.tensor(1.0)  # 以 tensor 形式新建
        lambda_loss_render_obv_one = torch.tensor(1.0)
        lambda_loss_render_obv_two = torch.tensor(1.0)
        lambda_loss_render_pred_two = torch.tensor(0.1) # 高一点，考虑另一个网络代表predict
        lambda_loss_render_merge_two = torch.tensor(0.1) # 暂时不用，不是真的，只是为了不报错
        lambda_loss_cross = torch.tensor(0.1) # 暂时还拿不出来


        for i, viewpoint_cam in enumerate(viewpoint_cams):
            loss_kal = 0
            # 此时 opt.batchsize 为 3，是为了装三个 view_cam 用于 kalman 滤波   
            # 此时 kalman_filter_training_step 中，viewpoint_cams 为 3 个 view_cam
            # 如果重复执行三次循环，则是重复三次，不要重复执行三次循环，只执行一次 kalman_filter_training_step
            if pipe.kalman_filter == True and stage == "fine":
                '''
                卡尔曼滤波求得下一帧的位置，协方差矩阵
                '''
                if i == 0:
                    viewspace_point_tensor, \
                    visibility_filter, \
                    radii, \
                    image_pkg,\
                    loss_pkg = kalman_filter_training_step(gaussians, 
                                                            viewpoint_cams, # 注意！不是 viewpoint_cam
                                                            scene, 
                                                            pipe, 
                                                            background, 
                                                            stage, 
                                                            iteration,
                                                            wasserstein_exp,
                                                            wasserstein_loss,
                                                            gaussian_merge
                                                            )

                    if iteration % 100 == 0:  # 当 iteration 为 100 的倍数时，记录图像
                        wandb.log({
                            "render_image_obv_zero": wandb.Image(process_image(image_pkg["render_image_obv_zero"]), caption="Render Image Obv Zero"),
                            "render_image_obv_one": wandb.Image(process_image(image_pkg["render_image_obv_one"]), caption="Render Image Obv One"),
                            "render_image_obv_two": wandb.Image(process_image(image_pkg["render_image_obv_two"]), caption="Render Image Obv Two"),
                            "render_image_pred_two": wandb.Image(process_image(image_pkg["render_image_pred_two"]), caption="Render Image Pred Two"),
                            "render_image_merge_two": wandb.Image(process_image(image_pkg["render_image_merge_two"]), caption="Render Image Merge Two"),
                            # "gt_image_zero": wandb.Image(process_image(image_pkg["gt_image"][0].unsqueeze(0)), caption="Gt Image zero"), # 有bug，有待解决
                            # "gt_image_one": wandb.Image(process_image(image_pkg["gt_image"][1].unsqueeze(0)), caption="Gt Image one"),
                            # "gt_image_two": wandb.Image(process_image(image_pkg["gt_image"][2].unsqueeze(0)), caption="Gt Image two"),
                        })
                    
                    image = image_pkg["render_image_obv_two"]
                    
                else:
                    continue


                ###########################################################################################

            # 不使用kalman滤波，按照 4dgs 原过程进行渲染
            else:
                if i == 0:
                    render_pkg = render(viewpoint_cam, 
                                        gaussians, 
                                        pipe, 
                                        background, 
                                        stage=stage,
                                        cam_type=scene.dataset_type,
                                        )
                else:
                    continue
                    

                # 保存当前帧的参数
                # xyz = render_pkg["means3D_final"].detach().clone()
                # rot = render_pkg["rotations_final"].detach().clone()
                # scale = render_pkg["scales_final"].detach().clone()
                # norm_rot = F.normalize(rot.detach().clone(), p=2, dim=1)
                # positive_scale = torch.exp(scale.detach().clone())
                # positive_scale = torch.square(positive_scale.detach().clone())
        
                image, \
                viewspace_point_tensor, \
                visibility_filter, \
                radii = render_pkg["render"], \
                        render_pkg["viewspace_points"], \
                        render_pkg["visibility_filter"], \
                        render_pkg["radii"]
                # print("viewspace_point_tensor:", viewspace_point_tensor)
                # print("viewspace_point_tensor.grad:", viewspace_point_tensor.grad)
            ########################################################################################### 
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()

        

        if pipe.kalman_filter == True and stage == "fine":  
            # loss = loss_kal  # 如不使用kalman滤波，loss_kal 为0
            # Ll1_of_obv_zero = l1_loss(image_pkg["render_image_obv_zero"].unsqueeze(0), image_pkg["gt_image"][0].unsqueeze(0)) * lambda_loss_render_obv_zero
            # Ll1_of_obv_one = l1_loss(image_pkg["render_image_obv_one"].unsqueeze(0), image_pkg["gt_image"][1].unsqueeze(0)) * lambda_loss_render_obv_one
            # Ll1_of_obv_two = l1_loss(image_pkg["render_image_obv_two"].unsqueeze(0), image_pkg["gt_image"][2].unsqueeze(0)) * lambda_loss_render_obv_two
            # Ll1_of_pred_two = l1_loss(image_pkg["render_image_pred_two"].unsqueeze(0), image_pkg["gt_image"][2].unsqueeze(0)) * lambda_loss_render_pred_two
            # Ll1_of_merge_two = l1_loss(image_pkg["render_image_merge_two"].unsqueeze(0), image_pkg["gt_image"][2].unsqueeze(0)) * lambda_loss_render_merge_two  
            # loss_cross = torch.tensor(0.0)
            # if iteration > 0 and gaussian_params["means3D_final"][2].shape[0] == gaussian_params["predict_mean3D_2"].shape[0]:
            #     import pdb; pdb.set_trace()
            #     print("iteration:", iteration)
            #     loss_cross = wasserstein_loss(gaussian_params["means3D_final"][2], gaussian_params["scales_final"][2]**2, \
            #                                     gaussian_params["rot_matrix_final"][2], gaussian_params["predict_mean3D_2"], \
            #                                     cov2=gaussian_params["predict_cov3D_2"]).mean() * lambda_loss_cross
            # print("iteration:", iteration)
            Ll1_of_obv_zero = loss_pkg["loss_render_obv_zero"] * lambda_loss_render_obv_zero
            Ll1_of_obv_one = loss_pkg["loss_render_obv_one"] * lambda_loss_render_obv_one
            Ll1_of_obv_two = loss_pkg["loss_render_obv_two"] * lambda_loss_render_obv_two
            Ll1_of_pred_two = loss_pkg["loss_render_pred_two"] * lambda_loss_render_pred_two
            Ll1_of_merge_two = loss_pkg["loss_render_merge_two"] * lambda_loss_render_merge_two
            loss_cross =  loss_pkg["loss_cross"] * lambda_loss_cross if loss_pkg["loss_cross"] > 2e-4 else 0
            # print("wasserstein_loss:", loss_cross)


            Ll1 = Ll1_of_obv_zero + Ll1_of_obv_one + Ll1_of_obv_two + Ll1_of_pred_two + Ll1_of_merge_two + loss_cross

            if iteration % 1 == 0:
                wandb.log({
                    f"{lambda_loss_render_obv_zero}: Ll1_of_obv_zero": Ll1_of_obv_zero.item(),
                    f"{lambda_loss_render_obv_one}: Ll1_of_obv_one": Ll1_of_obv_one.item(),
                    f"{lambda_loss_render_obv_two}: Ll1_of_obv_two": Ll1_of_obv_two.item(),
                    f"{lambda_loss_render_pred_two}: Ll1_of_pred_two": Ll1_of_pred_two.item(),
                    f"{lambda_loss_render_merge_two}: Ll1_of_merge_two": Ll1_of_merge_two.item(),
                    f"{lambda_loss_cross}: loss_cross": (loss_pkg["loss_cross"] * lambda_loss_cross).item(),
                })

            psnr_of_obv_two = psnr(image_pkg["render_image_obv_two"].unsqueeze(0), image_pkg["gt_image"][2].unsqueeze(0)).mean().double()
            psnr_ = psnr_of_obv_two
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        
        loss = Ll1

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            # 如果 tv_loss 是 nan，不添加 tv_loss
            if not torch.isnan(tv_loss).any():
                loss += tv_loss
            else:
                print("tv_loss is nan, not add to loss.")
            
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        
        with torch.autograd.set_detect_anomaly(True):
            # 对每个参数进行梯度裁剪
            max_norm = 1.0
            for param in [
                gaussians._xyz,
                gaussians._scaling,
                gaussians._rotation,
                # gaussians._features_dc,
                # gaussians._features_rest,
                # gaussians._opacity,
            ]:
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_([param], max_norm)
            # debug_print("因为 gaussian 参数出现 nan，添加梯度裁剪")
            try:
                loss.backward()
            except Exception as e:
                print(e)
                print("loss_cross:", loss_cross)
                import pdb; pdb.set_trace()
        
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            # 重启程序
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # import pdb; pdb.set_trace()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            try:
                # print("viewspace_point_tensor_list[idx].grad:", viewspace_point_tensor_list[idx].grad)
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
         
        # 计算经过的时间（毫秒）
        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            # 在训练循环中
            training_report_wandb(iteration, Ll1, loss, elapsed, testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0

    wandb.init(project=f"4DGS_Wasserstein_{expname.split('/')[-1]}", config={
        # "learning_rate": opt.learning_rate,
        "iteration": opt.iterations,
        # "batch_size": opt.batch_size,
        "model": expname,
        "opt": opt.__dict__,
        "hyper": hyper.__dict__,
        "pipe": pipe.__dict__,
    })

    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)

    wasserstein_loss = WassersteinGaussian()
    # debug_print("add wasserstein loss")

    timer.start()
    print("开始训练 coarse")
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer,
                             wasserstein_loss
                             )
    print("开始训练 fine")
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer,
                         wasserstein_loss
                         )

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        print("Config file saved to {}".format(os.path.join(args.model_path, "cfg_args")))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()

    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000]) # origin 3000, 7000, 14000
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    wandb.finish()
    print("\nTraining complete.")
