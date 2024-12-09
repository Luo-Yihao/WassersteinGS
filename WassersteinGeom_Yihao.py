import torch
import torch.nn as nn

### Utils
def matrix_sqrt_H(A):
    """
    Calculate the square root of a Hermitian matrix
    """
    A = symmetrize(A)
    L, Q = torch.linalg.eigh(A)
    # sqrt_A = torch.bmm(Q, torch.bmm(L.sqrt().diag_embed(), Q.transpose(-1, -2)))

    # Square and then take the fourth root, to ensure the positive definiteness
    sqrt_A = torch.bmm(Q, torch.bmm(((L**2)**0.25).diag_embed(), Q.transpose(-1, -2)))
    return sqrt_A


def symmetrize(A):
    return (A + A.transpose(-1, -2))/2

## Wasserstein Geodesic Distance 


class WassersteinDistGS(nn.Module):
    def __init__(self):
        super(WassersteinDistGS, self).__init__()

    def forward(self, loc0, scale0, rot_matrix0, loc1, scale1, rot_matrix1, cov0=None, cov1=None):
        """
        compute the Wasserstein distance between two Gaussians
        # Cov = rot_matrix @ diag(scale) @ rot_matrix^T
        loc0, loc1: Bx3
        scale0, scale1: Bx3
        rot_matrix0, rot_matrix1: Bx3x3
        cov0, cov1: Bx3x3, optional
        """
        if cov0 is not None:
            cov0 = symmetrize(cov0)
            scale0, rot_matrix0 = torch.linalg.eigh(cov0)
            scale0 = torch.clamp(scale0, min=1e-8)

        if cov1 is None:
            cov1 = torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2))) # covariance matrix Bx3x3
        
        loc_diff2 = torch.sum((loc0 - loc1)**2, dim=-1)

        ## Wasserstein distance Tr(C1 + C2 - 2(C1^0.5 * C2 * C1^0.5)^0.5)

        cov0_sqrt_diag = torch.sqrt(scale0).diag_embed() # Bx3x3

        
        cov1_R1 = torch.bmm(rot_matrix0.transpose(1, 2), cov1).matmul(rot_matrix0) # Bx3x3
        # E = cv1^0.5*cv2*cv1^0.5

        E = torch.bmm(torch.bmm(cov0_sqrt_diag, cov1_R1), cov0_sqrt_diag) # Bx3x3

        E = (E + E.transpose(1, 2))/2
        E_eign = torch.linalg.eigvalsh(E)


        E_sqrt_trace = (E_eign.pow(2).pow(1/4)).sum(dim=-1)

        if cov0 is None:
            cov0_trace = scale0.sum(dim=-1)
        else:
            cov0_trace = cov0.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        CovWasserstein = cov0_trace + cov1.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 2*E_sqrt_trace
        
        CovWasserstein = torch.clamp(CovWasserstein, min=1e-8) # numerical stability for small negative values

        return torch.sqrt(loc_diff2 + CovWasserstein)
    


class WassersteinExpGS(nn.Module):
    def __init__(self):
        super(WassersteinExpGS, self).__init__()

    def forward(self, loc, scale1, rot_matrix1, velocity, velocity_cov):
        """
        compute the Wasserstein Exponential of X from A
        loc: Bx3
        scale1: Bx3
        rot_matrix1: Bx3x3
        velocity: Bx3
        velocity_cov: Bx3x3 
        """
        new_loc = loc + velocity

        # new_cov = exp_A(X)
        C_ij = rot_matrix1.transpose(1, 2).bmm(velocity_cov).bmm(rot_matrix1)

       
        E_ij = scale1.unsqueeze(-1) + scale1.unsqueeze(-2) # Bx3x3
        E_ij = C_ij/(E_ij+1e-8) # Bx3x3

        gamma = torch.bmm(rot_matrix1, torch.bmm(E_ij, rot_matrix1.transpose(1, 2)))

        cov = torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2))) # covariance matrix Bx3x3

        new_cov = cov + velocity_cov + gamma.bmm(cov).bmm(gamma.transpose(1, 2))

        return new_loc, new_cov




class WassersteinLogGS(nn.Module):
    def __init__(self):
        super(WassersteinLogGS, self).__init__()

    def forward(self, miu_1, miu_2, cov_1, cov_2):
        """
        Compute Wasserstein Log: 
        Input: miu_1, miu_2: Bx3
                cov_1, cov_2: Bx3x3
        Output: miu_velocity = miu_2 - miu_1, 
                cov_velocity = log_cov1(cov2), Bx3x3
        """
        #########

        L_A, Q_A = torch.linalg.eigh(symmetrize(cov_1))

        L_A = torch.clamp(L_A, min=1e-8)

        A_sqrt = torch.bmm(Q_A, torch.bmm(L_A.sqrt().diag_embed(), Q_A.transpose(-1, -2)))

        
        A_sqrt_inv = torch.bmm(Q_A, torch.bmm((1e-8+L_A.sqrt()).reciprocal().diag_embed(), Q_A.transpose(-1, -2)))

        C = A_sqrt.bmm(cov_2).bmm(A_sqrt.transpose(-1, -2))

        C = symmetrize(C)

        C_sqrt = matrix_sqrt_H(C + torch.eye(cov_1.shape[-1]).to(cov_1.device) * 1e-8)

        cov_velocity = A_sqrt.bmm(C_sqrt).bmm(A_sqrt_inv.transpose(-1, -2)) 

        cov_velocity = cov_velocity + cov_velocity.transpose(-1, -2) - cov_1 - cov_1.transpose(-1, -2)
        #########

        return miu_2 - miu_1, cov_velocity



## Gaussian Merge
class GaussianMerge(nn.Module):
    def __init__(self):
        super(GaussianMerge, self).__init__()

    def forward(self, loc1, scale1, rot_matrix1, loc2, scale2, rot_matrix2, cov1=None, cov2=None):
        """
        merge two Gaussians
        loc1, loc2: Bx3
        scale1, scale2: Bx3
        rot_matrix1, rot_matrix2: Bx3x3
        cov1, cov2: Bx3x3, optional
        """
        if cov1 is not None:
            cov1 = torch.bmm(rot_matrix1, torch.bmm(torch.diag_embed(scale1), rot_matrix1.transpose(1, 2))) # covariance matrix Bx3x3
            
        if cov2 is not None:    
            cov2 = torch.bmm(rot_matrix2, torch.bmm(torch.diag_embed(scale2), rot_matrix2.transpose(1, 2)))

        K = cov1.matmul((cov1 + cov2 + torch.eye(3).to(cov1.device)*1e-8).inverse())
        loc_new = loc1.unsqueeze(1) + (loc2.unsqueeze(1) - loc1.unsqueeze(1)).bmm(K.transpose(1, 2))
        loc_new = loc_new.squeeze(1)
        cov_new = cov1 + K.matmul(cov1)

        return loc_new, cov_new

# example

if __name__ == "__main__":
    device = torch.device("cuda")
    B = 6 # batch size

    # loc_0 = torch.randn(B, 3).to(device) # location Bx3
    # rot_0 = torch.randn(B, 4).to(device) # quaternion Bx4
    # rot_0 = F.normalize(rot_0, p=2, dim=1) # normalize quaternion
    # scale_0 = torch.randn(B, 3).to(device) # scale Bx3
    # scale_0 = torch.exp(scale_0) # make sure scale is positive

    # loc_1 = torch.randn(B, 3).to(device) # location Bx3
    # rot_1 = torch.randn(B, 4).to(device) # quaternion Bx4
    # rot_1 = F.normalize(rot_1, p=2, dim=1) # normalize quaternion
    # scale_1 = torch.randn(B, 3).to(device) # scale Bx3
    # scale_1 = torch.exp(scale_1) # make sure scale is positive


    loc_0 = torch.randn(B, 3).to(device)
    cov_0 = torch.randn(B, 3, 3).to(device)
    cov_0 = cov_0.bmm(cov_0.transpose(1, 2))+1e-8*torch.eye(3).to(device).unsqueeze(0) # make it positive definite

    loc_1 = torch.randn(B, 3).to(device)
    cov_1 = torch.randn(B, 3, 3).to(device)
    cov_1 = cov_1.bmm(cov_1.transpose(1, 2))+1e-8*torch.eye(3).to(device).unsqueeze(0) # make it positive definite

    # Eigenvalue decomposition (Optional)
    scale_0, rot_matrix_0 = torch.linalg.eigh(symmetrize(cov_0)) # R@diag(S)@R^T = cov0
    scale_1, rot_matrix_1 = torch.linalg.eigh(symmetrize(cov_1)) # R@diag(S)@R^T = cov1

    wasserstein_dist = WassersteinDistGS()(loc_0, scale_0, rot_matrix_0, loc_1, scale_1, rot_matrix_1)
    assert (wasserstein_dist >= 0).all(), "Wasserstein distance should be non-negative"


    miu_velocity, cov_velocity = WassersteinLogGS()(loc_1, loc_0, cov_1, cov_0)
    miu_velocity = -miu_velocity
    cov_velocity = -cov_velocity
    
    assert not torch.isnan(cov_velocity).any(), "Covariance matrix should not be NaN"

    loc_2, cov_2 = WassersteinExpGS()(loc_1, scale_1, rot_matrix_1, miu_velocity, cov_velocity)

    assert not torch.isnan(loc_2).any(), "Location should not be NaN"
    assert not torch.isnan(cov_2).any(), "Covariance matrix should not be NaN"

    wasserstein_dist_2 = WassersteinDistGS().forward(loc0=loc_0, scale0=scale_0, rot_matrix0=rot_matrix_0,
                                                     loc1=loc_2, scale1=None, rot_matrix1=None, cov1=cov_2)
                                                    

    print("GS2 = Exp(GS1, -Log(GS1, GS0))")

    print("Wasserstein distance from GS0 to GS1: ", wasserstein_dist)
    print("Wasserstein distance from GS0 to GS2: ", wasserstein_dist_2)



