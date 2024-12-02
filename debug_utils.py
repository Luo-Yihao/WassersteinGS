import inspect
import datetime
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, clips_array
from PIL import Image, ImageDraw, ImageFont
import wandb
import torch
import numpy as np


def debug_print(message):
    caller = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"[DEBUG] {datetime.datetime.now()} - {caller.filename}:{caller.lineno} - {message}")

def training_report_wandb(iteration, 
                          Ll1, 
                          loss, 
                          elapsed, 
                          testing_iterations, 
                          scene, 
                          renderFunc, 
                          renderArgs, 
                          stage, 
                          dataset_type):
    # 训练数据区域
    wandb.log({
        'metric_train/l1_loss': Ll1.item(),
        'metric_train/total_loss': loss.item(),
        'metric_train/iter_time': elapsed,
        'iteration': iteration
    })

    # 测试数据区域
    if iteration % 500 == 0:
        torch.cuda.empty_cache()
        validation_configs = \
        ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
         {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras']:
                l1_test = 0.0
                psnr_test = 0.0
                
                render_images = []
                gt_images = []
                
                for idx, viewpoint in enumerate(config['cameras'][:5]):  # 只处理前5个视角
                    render_output = renderFunc(viewpoint, scene.gaussians, stage=stage, cam_type=dataset_type, *renderArgs)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint["image"].to("cuda") if dataset_type == "PanopticSports" else viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    render_images.append(image.cpu().numpy().transpose(1, 2, 0))
                    gt_images.append(gt_image.cpu().numpy().transpose(1, 2, 0))
                    
                    l1_test += torch.abs(image - gt_image).mean().item()
                    psnr_test += -10 * torch.log10(torch.mean((image - gt_image) ** 2)).item()
                
                # 拼接图像
                render_concat = np.concatenate(render_images, axis=1)
                gt_concat = np.concatenate(gt_images, axis=1)
                
                # 转换为PIL图像
                render_pil = Image.fromarray((render_concat * 255).astype(np.uint8))
                gt_pil = Image.fromarray((gt_concat * 255).astype(np.uint8))
                
                l1_test /= len(config['cameras'][:5])
                psnr_test /= len(config['cameras'][:5])
                
                wandb.log({
                    f"metric_{config['name']}/avg_l1_loss": l1_test,
                    f"metric_{config['name']}/avg_psnr": psnr_test,
                    f"visualization_{config['name']}/render": wandb.Image(render_pil, caption=f"Iteration {iteration} - {stage} - {config['name']} Render"),
                    f"visualization_{config['name']}/ground_truth": wandb.Image(gt_pil, caption=f"Iteration {iteration} - {stage} - {config['name']} Ground Truth"),
                })
                
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.2f}")

        # 场景统计信息区域
        opacity = scene.gaussians.get_opacity.cpu().numpy()
        deformation = scene.gaussians._deformation_accum.mean(dim=-1).cpu().numpy() / 100

        wandb.log({
            "metric_train/opacity_histogram": wandb.Histogram(opacity),
            "metric_train/motion_histogram": wandb.Histogram(deformation),
            "metric_train/total_points": scene.gaussians.get_xyz.shape[0],
            "metric_train/deformation_rate": scene.gaussians._deformation_table.sum().item() / scene.gaussians.get_xyz.shape[0],
        })

        # 点云可视化区域
        if hasattr(scene.gaussians, 'get_xyz'):
            point_cloud = scene.gaussians.get_xyz.detach().cpu().numpy()
            point_colors = scene.gaussians.get_features[:, :3].detach().cpu().numpy()
            wandb.log({
                "visualization_train/3D_point_cloud": wandb.Object3D({
                    "type": "lidar/beta",
                    "points": point_cloud,
                    "colors": point_colors,
                }, caption=f"{stage} - 3D point cloud"),
            })

        torch.cuda.empty_cache()

    # 学习率记录区域
    if iteration % 500 == 0:
        current_lr = scene.gaussians.optimizer.param_groups[0]['lr']
        wandb.log({'metric_train/learning_rate': current_lr})
        
'''
拼接对比视频
'''
def create_text_image(text, size=(1280, 720), color='yellow', bg_color=None):
    # 创建一个空白图像（透明背景）
    img = Image.new('RGBA', size, (0, 0, 0, 0) if bg_color is None else bg_color)
    draw = ImageDraw.Draw(img)

    # 使用默认字体
    font = ImageFont.load_default()

    # 获取文本的宽高
    text_size = draw.textsize(text, font=font)
    
    # 将文本绘制到图像上，左上角对齐
    text_position = (20, 20)  # 稍微偏移一点，避免靠得太近
    draw.text(text_position, text, font=font, fill=color)

    return img

def add_text_to_video(video, text, position, scale=1.0):
    # 创建一个文本图像
    text_img = create_text_image(text, size=(video.w, video.h))
    
    # 将PIL图像转换为NumPy数组
    text_img_array = np.array(text_img)
    
    # 将NumPy数组转换为MoviePy的ImageClip，设置透明度
    text_clip = (ImageClip(text_img_array, ismask=False)
                 .set_duration(video.duration)
                 .set_position(position)
                 .resize(scale))  # 使用 resize 方法调整大小
    
    # 将文本叠加到视频上（透明背景）
    video = CompositeVideoClip([video, text_clip])
    return video

def concatenate_videos_with_text(video_path1, video_path2, text1, text2, output_path, text_scale=3):
    # 加载两个视频文件
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)
    
    # 在每个视频的左上方添加文字说明，并调整大小
    clip1 = add_text_to_video(clip1, text1, position=("left", "top"), scale=text_scale)
    clip2 = add_text_to_video(clip2, text2, position=("left", "top"), scale=text_scale)

    # 确保两个视频的高度相同
    if clip1.h != clip2.h:
        clip2 = clip2.resize(height=clip1.h)

    # 将两个视频并排拼接在一起
    final_clip = clips_array([[clip1, clip2]])
    
    # 写入到输出文件
    final_clip.write_videofile(output_path, codec='libx264')

def main():
    # 可编辑的视频文件路径
    video1_base_path = "./output/dnerf/视频文件夹1/video/ours_20000/video_rgb.mp4"
    video2_base_path = "./output/dnerf/视频文件夹2/video/ours_20000/video_rgb.mp4"

    # 可编辑的部分，修改"其他名字"和说明文字
    # other_name_1 = "jumpingjacks_origin"
    other_name_1 = "jumpingjacks_kalman_wasserstein_between_frame"
    other_name_2 = "jumpingjacks_kalman_wasserstein_log"
    # text1 = "4dgs"
    text1 = "4dgs_kalman_wasserstein_between_frame"
    text2 = "4dgs_kalman_wasserstein_log"

    # 更新路径
    video1_path = video1_base_path.replace("视频文件夹1", other_name_1)
    video2_path = video2_base_path.replace("视频文件夹2", other_name_2)

    # 输出文件路径
    output_path = f"./output/对比_{other_name_1}_and_{other_name_2}.mp4"

    # 拼接视频
    concatenate_videos_with_text(video1_path, video2_path, text1, text2, output_path)

if __name__ == "__main__":
    main()
