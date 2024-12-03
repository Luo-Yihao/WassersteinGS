## WGS: Wasserstein Gaussian Splatting for Dynamic Scene Rendering

[![arXiv](https://img.shields.io/badge/arXiv-2412.00333-b31b1b.svg)](https://arxiv.org/abs/2412.00333)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://cucdengjunli.github.io/wgs/)

Official repository for the paper [Gaussians on their Way: Wasserstein-Constrained 4D Gaussian Splatting with State-Space Modeling](https://arxiv.org/abs/2412.00333) by Junli Deng and Yihao Luo. This repository implements a novel approach that combines state-space modeling with Wasserstein geometry to achieve more natural and fluid motion in 4D Gaussian Splatting for dynamic scene rendering. 

Our method addresses the fundamental challenge of making 3D Gaussians move through time as naturally as they would in the real world while maintaining smooth and consistent motion.

### Overview

![WGS Overview](figures/Demo_WGS.png)

Dynamic scene rendering with 4D Gaussian Splatting faces challenges in modeling precise scene dynamics. Our approach combines control theory with three key innovations:

- **State Consistency Filter**: Models Gaussian deformation as a dynamic system state, merging predictions with observations for better temporal consistency.

- **Wasserstein Distance Regularization**: Uses Wasserstein distance to ensure smooth parameter updates and reduce motion artifacts.

- **Wasserstein Geometry Modeling**: Captures motion and deformation using Wasserstein geometry's logarithmic and exponential mappings for more natural dynamics.

Our method achieves smoother, more realistic motion by guiding Gaussians through Wasserstein space.

### TODO List

#### 🚧 In Progress
- [x] Core differentiable Wasserstein geometry implementation in `WassersteinGeom_Yihao.py`
  - Wasserstein distance calculation
  - Wasserstein logarithmic mapping
  - Wasserstein exponential mapping
  - Gaussian distribution merging
- [x] Provide plug-and-play code samples
- [x] Add video demonstrations
- [ ] Release inference code
- [ ] Upload pre-trained models
- [ ] Create documentation for model usage



### Quick Start
We assume that flickering artifacts in videos arise from the variations in the shape or position of Gaussians between adjacent frames. By applying Wasserstein distance constraints between the Gaussian spheres of consecutive frames, we can effectively mitigate the occurrence of these artifacts. 

Here's a minimal example showing how to add Wasserstein distance constraints between consecutive frames:

```python
from WassersteinGeom_Yihao import WassersteinDistGS

# Initialize Wasserstein distance calculator
wasserstein_distance = WassersteinDistGS()

# # In your training loop, make sure the iterations are processed in chronological order
for iteration in range(iterations):
    # Process current and next frame's Gaussians
    means3D_curr, scales_curr, rot_matrix_curr = process_frame(frame_t)
    means3D_next, scales_next, rot_matrix_next = process_frame(frame_t + 1)
    
    # Calculate Wasserstein distance loss between consecutive frames
    loss_wasserstein = wasserstein_distance(
        means3D_curr, scales_curr**2, rot_matrix_curr,
        means3D_next, scales_next**2, rot_matrix_next
    ).mean()
    
    # Add to total loss
    loss = render_loss + 0.1 * loss_wasserstein
```
This constraint helps ensure smooth transitions between frames by penalizing large changes in Gaussian distributions. The plug-and-play regularization method is applicable to other Gaussian-based video tasks, and we recommend using it for your task.



### How to Use

#### Wasserstein Geometry on Gaussians
One of the key innovations in our approach is the integration of Wasserstein geometry into the Gaussian dynamics modeling. We provide a set of differentiable PyTorch classes that implement explicit computing of Wasserstein distance, logarithmic and exponential mappings, and Gaussian distribution merging. These classes are essential for modeling the dynamics of Gaussian distributions in a consistent and physically plausible manner. 

Only pure [PyTorch](https://pytorch.org/) is required to run the code.

For quick start, run the following command 
```bash
cd WassersteinGS
python WassersteinGeom_Yihao.py
```

Sample code snippets for using these classes are provided below:

1. Prepare the data for two batches of Gaussian distributions:

```python
import torch
from WassersteinGeom_Yihao import WassersteinDistGS, WassersteinLog, WassersteinExp

device = torch.device("cuda")
B = 6 # batch size
# Example data
loc0 = torch.randn(B, 3).to(device)
cov0 = torch.randn(B, 3, 3).to(device)
cov0 = cov0.bmm(cov0.transpose(1, 2))+1e-8*torch.eye(3).to(device).unsqueeze(0) 
# make it positive definite
loc1 = torch.randn(B, 3).to(device)
cov1 = torch.randn(B, 3, 3).to(device)
cov1 = cov1.bmm(cov1.transpose(1, 2))+1e-8*torch.eye(3).to(device).unsqueeze(0) 
# Eigenvalue decomposition (Optional)
scale0, rot0 = torch.linalg.eigh(cov0) # R@diag(S)@R^T = cov
scale1, rot1 = torch.linalg.eigh(cov1)
```

2. Compute the Wasserstein distance between two Gaussian distributions:

```python
# Wasserstein distance
WG_dist = WassersteinDistGS()
dist = WassersteinDistGS()(loc0, scale0, rot0, loc1, scale1, rot1)
# Alternatively, you can provide covariance matrices directly
# dist = WG_dist(loc0=loc0, scale0=None, rot_matrix0=None, loc1=loc1, scale1=None, 
#                rot_matrix1=None, cov0=cov0, cov1=cov1) 
print("Wasserstein distance from GS0 to GS1:", dist)
```
3. Compute the logarithmic mapping between two Gaussians (velocity and velocity covariance):

```python
# Wasserstein logarithmic mapping
miu_velocity, cov_velocity = WassersteinLogGS()(loc_1, loc_0, cov_1, cov_0)
miu_velocity = -miu_velocity
cov_velocity = -cov_velocity
```
*Remark. Note that the velocity from 'GS0' to 'GS1' is not generally the inverse of the velocity from 'GS1' to 'GS0' in Wasserstein space. The velocity is computed in the tangent space of the Wasserstein manifold.*

4. Compute the exponential mapping to predict the new Gaussian distribution based on the previous frame's distribution and the current frame's velocity and velocity covariance:

```python
# Wasserstein exponential mapping
new_loc, new_cov = WassersteinExpGS()(loc_0, cov_0, miu_velocity, cov_velocity)
## Check the distance between the GS0 and the new GS
dist = WG_dist(loc0, scale0, rot0, new_loc, scale1=None, rot1=None, cov1=new_cov)
print("Wasserstein distance from GS0 to GS2:", dist)
```

The results may look like this:
```
Wasserstein distance from GS0 to GS1:  
tensor([1.8248, 3.8078, 2.0212, 2.9639, 3.5641, 2.4605], device='cuda:0')
Wasserstein distance from GS0 to GS2:  
tensor([3.6496, 7.4741, 4.0424, 4.4655, 7.1282, 4.6840], device='cuda:0')
```
It is easy to see that the Wasserstein distance between GS0 and GS2 is approximately twice the distance between GS0 and GS1. In other words, GS2 is a natural extension of GS1 from GS0.


### Contributions

Some source code of ours is borrowed from the following repositories:
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [K-planes](https://github.com/Giodiro/kplanes_nerfstudio)
- [HexPlane](https://github.com/Caoang327/HexPlane)
- [TiNeuVox](https://github.com/hustvl/TiNeuVox)
- [Depth-Rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization)
- [4DGS](https://github.com/hustvl/4DGaussians)

We sincerely appreciate the excellent works of these authors, which have greatly influenced our project.

### Citation
If you find this repository helpful in your research or project, please consider citing our work:

```
@misc{deng2024gaussianswaywassersteinconstrained4d,
      title={Gaussians on their Way: Wasserstein-Constrained 4D Gaussian Splatting with State-Space Modeling}, 
      author={Junli Deng and Yihao Luo},
      year={2024},
      eprint={2412.00333},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00333}, 
}

@article{luo2021geometric,
  title={Geometric Characteristics of the Wasserstein Metric on SPD(n) 
  and Its Applications on Data Processing},
  author={Luo, Yihao and Zhang, Shiqiang and Cao, Yueqi and Sun, Huafei},
  journal={Entropy},
  volume={23},
  number={9},
  pages={1214},
  year={2021},
  publisher={MDPI}
}
```
We appreciate any feedback, suggestions, or potential collaborations. Please feel free to reach out to us at [y.luo23@imperial.ac.uk](mailto:y.luo23@imperial.ac.uk).

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

