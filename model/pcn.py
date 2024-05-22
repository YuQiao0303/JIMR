import torch
import torch.nn as nn

from lib.chamfer_distance.chamfer_distance import ChamferDistance
CD = ChamferDistance()
# EMD = EarthMoverDistance()

class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_coarse=1024, latent_dim=1024, grid_size=4):
        super().__init__()

        self.num_coarse = num_coarse
        self.latent_dim = latent_dim
        self.grid_size = grid_size


        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )


    def forward(self, xyz):
        B, N, _ = xyz.shape
        # print('xyz.shape',xyz.shape)
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))
        # print('feature.shape',feature.shape)# (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1) # this line
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud

        return coarse.contiguous()

    # added by Qiao
    def encoder(self,xyz):
        B, N, _ = xyz.shape
        # print('xyz.shape',xyz.shape)

        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))
        # print('feature.shape',feature.shape)# (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1) # this line
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)
        return feature_global

    # added by Qiao
    def decoder(self,feature_global):
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)  # (B, num_coarse, 3), coarse point cloud

        return coarse.contiguous()


def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    # print(pcs2)
    # print(dist1,dist2)
    return torch.mean(dist1) + torch.mean(dist2)