## WGS: Wasserstein Gaussian Splatting for Dynamic Scene Rendering

This repository implements a novel approach that combines state-space modeling with Wasserstein geometry to achieve more natural and fluid motion in 4D Gaussian Splatting for dynamic scene rendering. Our method addresses the fundamental challenge of making 3D Gaussians move through time as naturally as they would in the real world while maintaining smooth and consistent motion.

## Overview

Dynamic scene rendering has seen significant advances with 4D Gaussian Splatting, but accurately modeling scene dynamics remains challenging due to limitations in estimating precise Gaussian transformations. Our approach draws inspiration from control theory and introduces three key innovations:

- **State Consistency Filter**: By modeling each Gaussian's deformation as a state in a dynamic system, we merge prior predictions with current observations to estimate transformations more accurately, enabling Gaussians to maintain temporal consistency.

- **Wasserstein Distance Regularization**: We employ Wasserstein distance as a key metric between Gaussian distributions to ensure smooth and consistent parameter updates, effectively reducing motion artifacts while preserving the underlying Gaussian structure.

- **Wasserstein Geometry Modeling**: Our framework leverages Wasserstein geometry to capture both translational motion and shape deformations in a unified way, resulting in more physically plausible Gaussian dynamics and improved motion trajectories.

Our method guides Gaussians along their natural way in the Wasserstein space, achieving smoother, more realistic motion and stronger temporal coherence. Experimental results demonstrate significant improvements in both rendering quality and efficiency compared to current state-of-the-art techniques.

## TODO List

### ✅ Completed
- [x] Core Wasserstein geometry implementation in `WassersteinGeom_Yihao.py`
  - Wasserstein distance calculation
  - Wasserstein logarithmic mapping
  - Wasserstein exponential mapping
  - Gaussian distribution merging

### 🚧 In Progress
- [ ] Release inference code
- [ ] Upload pre-trained models
- [ ] Add video demonstrations
- [ ] Create documentation for model usage
- [ ] Add evaluation scripts
- [ ] Provide example training configurations
- [ ] Create installation guide
- [ ] Add benchmarking results
- [ ] Provide data preprocessing scripts

### 📝 Future Plans
- [ ] Support for more dynamic scene types
- [ ] Real-time rendering optimization
- [ ] Multi-GPU training support
- [ ] Interactive visualization tools
- [ ] Integration with popular rendering frameworks




## Core Innovations

The core innovations of our method are implemented in the following four classes:

1. [`WassersteinGaussian`](#1-wassersteingaussian)
2.  [`WassersteinLog`](#2-wassersteinlog)
3. [`WassersteinExp`](#3-wassersteinexp)
4. [`GaussianMerge`](#4-gaussianmerge)

Each class corresponds to a key component of our approach, as detailed below.

### 1. `WassersteinGaussian`

This class computes the **Wasserstein distance** between two Gaussian distributions, capturing both positional and covariance differences. It is crucial for measuring the distance in a way that aligns with the geometry of the space of Gaussian distributions.

**Code:**

```python
class WassersteinGaussian(nn.Module):
    def __init__(self):
        super(WassersteinGaussian, self).__init__()

    def forward(self, loc1, scale1, rot_matrix1, loc2, scale2=None, rot_matrix2=None, cov2=None):
        """
        Calculate Wasserstein distance between two Gaussian distributions.

        Args:
            loc1, loc2: Mean vectors of the Gaussians (Bx3)
            scale1, scale2: Scale (standard deviations) of the Gaussians (Bx3)
            rot_matrix1, rot_matrix2: Rotation matrices of the Gaussians (Bx3x3)
            cov2: Covariance matrix of the second Gaussian (optional)

        Returns:
            Wasserstein distance between the two Gaussians.
        """
        # ensure scale parameters are non-negative
        scale1 = torch.clamp(scale1, min=1e-8)
        if scale2 is not None:
            scale2 = torch.clamp(scale2, min=1e-8)

        # calculate location difference
        loc_diff2 = torch.sum((loc1 - loc2)**2, dim=-1)

        # Wasserstein distance Tr(C1 + C2 - 2(C1^0.5 * C2 * C1^0.5)^0.5)
        cov1_sqrt_diag = torch.sqrt(scale1).diag_embed()  # Bx3x3

        if cov2 is None:
            assert rot_matrix2 is not None and scale2 is not None
            cov2 = torch.bmm(rot_matrix2, torch.bmm(torch.diag_embed(scale2), rot_matrix2.transpose(1, 2)))  # covariance matrix Bx3x3

        cov2_R1 = torch.bmm(rot_matrix1.transpose(1, 2), cov2).matmul(rot_matrix1)  # Bx3x3
        E = torch.bmm(torch.bmm(cov1_sqrt_diag, cov2_R1), cov1_sqrt_diag)  # Bx3x3

        # ensure E is symmetric
        E = (E + E.transpose(1, 2)) / 2

        # add numerical stability processing
        epsilon = 1e-8
        batch_size, dim, _ = E.size()
        identity = torch.eye(dim, device=E.device).expand(batch_size, dim, dim)
        E = E + epsilon * identity

        # calculate eigenvalues and clip them
        E_eign = torch.linalg.eigvalsh(E)
        E_eign = torch.clamp(E_eign, min=1e-8)

        # calculate the trace of the square root of E
        E_sqrt_trace = torch.sqrt(E_eign).sum(dim=-1)


        # calculate Wasserstein distance of covariance
        CovWasserstein = scale1.sum(dim=-1) + cov2.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 2 * E_sqrt_trace
        
        # ensure CovWasserstein is non-negative
        CovWasserstein = torch.clamp(CovWasserstein, min=0)

        # return Wasserstein distance
        return torch.sqrt(torch.clamp(loc_diff2 + CovWasserstein, min=1e-8))

```

**Explanation:**

- **Purpose:** Computes the squared 2-Wasserstein distance between two 3D Gaussian distributions, considering both the means and covariances.
- **Usage:** Essential for regularizing the motion of Gaussians and ensuring smooth transitions between frames.

### 2. `WassersteinLog`

This class computes the **logarithmic map** in Wasserstein space to obtain the velocity between two Gaussian distributions, essential for modeling Gaussian dynamics.

**Code:**

```python
class WassersteinLog(nn.Module):
    def __init__(self):
        super(WassersteinLog, self).__init__()

    def forward(self, miu_1, miu_2, cov_1, cov_2):
        """
        Calculate Wasserstein logarithm log_cov1(cov2) to get the velocity between two Gaussian distributions.
        This function computes the velocity in the tangent space of the Wasserstein manifold.

        Args:
            miu_1, miu_2: Mean vectors of two Gaussian distributions (Bx3)
            cov_1, cov_2: Covariance matrices of two Gaussian distributions (Bx3x3)

        Returns:
            Tuple of:
            - Mean velocity (miu_2 - miu_1)
            - Covariance velocity in the tangent space
        """
        # Implementation details...
        A_sqrt = matrix_sqrt_H(cov_1 + torch.eye(cov_1.shape[-1]).to(cov_1.device) * 1e-8)
        
        A_sqrt_inv = torch.inverse(A_sqrt + torch.eye(cov_1.shape[-1]).to(cov_1.device) * 1e-8)

        C = A_sqrt.bmm(cov_2).bmm(A_sqrt.transpose(-1, -2))

        C_sqrt = matrix_sqrt_H(C + torch.eye(cov_1.shape[-1]).to(cov_1.device) * 1e-8)

        cov_velocity = A_sqrt.bmm(C_sqrt).bmm(A_sqrt_inv.transpose(-1, -2)) 

        cov_velocity = cov_velocity + cov_velocity.transpose(-1, -2) - 2*cov_1

        return miu_2 - miu_1, cov_velocity

```

**Explanation:**

- **Purpose:** Calculates the velocity required to transport one Gaussian distribution to another in Wasserstein space.
- **Usage:** Essential for computing the updates in the state consistency filter.

### 3. `WassersteinExp`

This class calculates the **exponential map** in Wasserstein space to predict the new Gaussian distribution based on the previous frame's distribution and the current frame's velocity and velocity covariance.

**Code:**

```python
class WassersteinExp(nn.Module):
    """
    Calculate new Gaussian distribution based on previous frame's distribution 
    and current frame's velocity and velocity covariance (Kalman filter result)
    """
    def __init__(self):
        super(WassersteinExp, self).__init__()

    def forward(self, loc, cov1=None, scale1=None, rot_matrix1=None, velocity=None, velocity_cov=None):
        """
        Calculate Wasserstein Exponential of X from A.
        Supports two input methods:
        1. loc, scale1, rot_matrix1, velocity, velocity_cov
        2. loc, cov1, velocity, velocity_cov

        Args:
            loc: Mean vector of the Gaussian (Bx3)
            cov1: Covariance matrix of the Gaussian (Bx3x3, optional)
            scale1: Scale (standard deviations) of the Gaussian (Bx3, optional)
            rot_matrix1: Rotation matrix of the Gaussian (Bx3x3, optional)
            velocity: Velocity vector (Bx3)
            velocity_cov: Velocity covariance matrix (Bx3x3) 

        Returns:
            Tuple of new mean and covariance matrix.
        """
        if cov1 is not None:
            # if input is cov1, then decompose it
            scale1, rot_matrix1 = torch.linalg.eigh(cov1)
            # the scale1 decomposed from cov1 is already squared

        assert scale1 is not None and rot_matrix1 is not None, "scale1 and rot_matrix1 must be provided"

        new_loc = loc + velocity

        C_ij = rot_matrix1.transpose(1, 2).bmm(velocity_cov).bmm(rot_matrix1)
       
        E_ij = scale1.unsqueeze(-1) + scale1.unsqueeze(-2) # Bx3x3
        E_ij = C_ij/(E_ij+1e-8) # Bx3x3

        gamma = torch.bmm(rot_matrix1, torch.bmm(E_ij, rot_matrix1.transpose(1, 2)))

        cov = torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2))) # covariance matrix Bx3x3

        new_cov = cov + velocity_cov + gamma.bmm(cov).bmm(gamma.transpose(1, 2))

        return new_loc, new_cov

```

**Explanation:**

- **Purpose:** Performs the exponential mapping in Wasserstein space, predicting the Gaussian distribution at the next time step.
- **Usage:** Used for state prediction in the Gaussian dynamics, considering both mean and covariance.



### 4. `GaussianMerge`

This class merges two Gaussian distributions and returns a new Gaussian distribution, analogous to the update step in a Kalman filter.

**Code:**

```python
class GaussianMerge(nn.Module):
    def __init__(self, device="cuda"):
        super(GaussianMerge, self).__init__()

        # Optional neural network to predict K (Kalman Gain)
        # self.nn_K = nn.Sequential(
        #     nn.Linear(18, 64),  
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 9)  
        # ).to(device)

    def forward(self, loc1, cov1, loc2, cov2):
        """
        Merge two Gaussian distributions.

        Args:
            loc1: Mean vector of first Gaussian distribution (Bx3)
            cov1: Covariance matrix of first Gaussian distribution (Bx3x3) 
            loc2: Mean vector of second Gaussian distribution (Bx3)
            cov2: Covariance matrix of second Gaussian distribution (Bx3x3)

        Returns:
            Tuple of:
            - Merged mean vector (Bx3)
            - Merged covariance matrix (Bx3x3)
        """

        # prevent singular matrix when inverting covariance matrix
        epsilon = 1e-6  # or smaller value, depending on the situation

        # get K by calculating
        K = cov1.matmul((cov1 + cov2 + torch.eye(cov1.shape[-1]).to(cov1.device) * epsilon).inverse())

        # flatten and concatenate covariance matrices, optional
        # cov_input = torch.cat([cov1.view(cov1.shape[0], -1), cov2.view(cov2.shape[0], -1)], dim=1)
        
        # # use neural network to predict K, optional
        # K = self.nn_K(cov_input).view(cov1.shape)

        loc_new = loc1.unsqueeze(1) + (loc2.unsqueeze(1) - loc1.unsqueeze(1)).bmm(K.transpose(1, 2))
        loc_new = loc_new.squeeze(1)
        cov_new = cov1 + K.matmul(cov1)

        return loc_new, cov_new
```

**Explanation:**

- **Purpose:** Implements the Kalman-like state updating mechanism by merging predictions with observations.
- **Usage:** Used to optimally combine prior predictions and observed data, accounting for uncertainties.

### Example Usage

Below is an example of how to use the core classes in your project.

```python
import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix

from wasserstein_gaussian import WassersteinGaussian
from wasserstein_exp import WassersteinExp
from wasserstein_log import WassersteinLog
from gaussian_merge import GaussianMerge

# Initialize the classes
wg = WassersteinGaussian()
we = WassersteinExp()
wl = WassersteinLog()
gm = GaussianMerge()

# Example data
B = 10  # Batch size
loc1 = torch.randn(B, 3)
scale1 = torch.exp(torch.randn(B, 3))  # Ensure positive scale
rot1 = F.normalize(torch.randn(B, 4), p=2, dim=1)  # Quaternion
rot_matrix1 = quaternion_to_matrix(rot1)

loc2 = torch.randn(B, 3)
scale2 = torch.exp(torch.randn(B, 3))
rot2 = F.normalize(torch.randn(B, 4), p=2, dim=1)
rot_matrix2 = quaternion_to_matrix(rot2)

# Compute Wasserstein distance between two Gaussians
wasserstein_distance = wg(loc1, scale1, rot_matrix1, loc2, scale2, rot_matrix2)
print("Wasserstein Distance:", wasserstein_distance)

# Compute Wasserstein logarithmic mapping
mean_velocity, cov_velocity = wl(loc1, loc2, new_cov, new_cov)
print("Mean Velocity:", mean_velocity)
print("Covariance Velocity:", cov_velocity)

# Perform Wasserstein exponential mapping
velocity = torch.randn(B, 3)
velocity_cov = torch.eye(3).repeat(B, 1, 1)
new_loc, new_cov = we(loc1, scale1=scale1, rot_matrix1=rot_matrix1, velocity=velocity, velocity_cov=velocity_cov)
print("New Location:", new_loc)
print("New Covariance:", new_cov)

# Merge two Gaussian distributions
merged_loc, merged_cov = gm(loc1, new_cov, loc2, new_cov)
print("Merged Location:", merged_loc)
print("Merged Covariance:", merged_cov)
```

## Results

Our method demonstrates improved temporal consistency and rendering quality in dynamic scenes. By incorporating the Wasserstein geometry, we achieve smoother Gaussian transitions and reduce artifacts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

We hope this repository helps you in your research or project. If you have any questions or suggestions, feel free to open an issue or contact us.