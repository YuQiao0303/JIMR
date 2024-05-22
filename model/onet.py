# Occupancy Networks
import torch
import torch.nn as nn

import torch.distributions as dist

from torch.nn import functional as F
from model.onet_modules import DecoderCBatchNorm,make_3d_grid,cosinematrix,Generator3D,ResnetPointnet


from visualization_utils import vis_actors_vtk, get_pc_actor_vtk,  \
    vis_occ_hat_voxel_vtk # vis_np_histogram,# visualization


class ONet(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, cfg, optim_spec=None):
        super(ONet, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Parameter Configs'''
        decoder_kwargs = {}


        dim = 3

        self.use_cls_for_completion = cfg.use_cls_for_completion
        # c_dim = self.use_cls_for_completion * cfg.dataset_config.num_class + cfg.c_dim
        c_dim = cfg.c_dim
        self.threshold = cfg.threshold

        '''Module Configs'''
        self.teacher_encoder = torch.nn.Sequential(
            ResnetPointnet(c_dim=cfg.c_dim,
                           dim=3,  # self.input_feature_dim + 3 + 128,
                           hidden_dim=cfg.hidden_dim),
            # nn.BatchNorm1d(c_dim)
        )
        self.teacher_encoder.eval()
        self.student_encoder = torch.nn.Sequential(
            ResnetPointnet(c_dim=cfg.c_dim,
                           dim=3,  # self.input_feature_dim + 3 + 128,
                           hidden_dim=cfg.hidden_dim),
            # nn.BatchNorm1d(c_dim)
        )

        self.decoder = DecoderCBatchNorm(dim=dim,  c_dim=c_dim, **decoder_kwargs)
        # self.decoder.eval()

        self.use_mask = cfg.use_mask
        if self.use_mask:
            self.mask_loss_weight = cfg.mask_loss_weight
            self.relu = nn.ReLU()
       

        '''Mount mesh generator'''
     
        self.generator = Generator3D(self,
                                     threshold=cfg.threshold,
                                     resolution0=cfg.resolution_0,
                                     upsampling_steps=cfg.upsampling_steps,
                                     sample=cfg.use_sampling,
                                     refinement_step=cfg.refinement_step,
                                     simplify_nfaces=cfg.simplify_nfaces,
                                     preprocessor=None)

    def point_occ_2voxel(self, points, logits, vsize=16):
        radius = 1
        vol = torch.zeros((vsize, vsize, vsize)).to(points.device)
        voxel = 2 * radius / float(vsize)
        in_points = points[logits > 0]
        in_logits = logits[logits > 0]  # .clamp(max=1)

        locations = (in_points + radius) / voxel  # point_num, 3 # location 
        locations = locations.long()  # point_num, 3
      
        for i in range(locations.shape[0]):
            vol[locations[i, 0], locations[i, 1], locations[i, 2]] += torch.sigmoid(in_logits[i])

    
        return vol

    def classify_loss(self, logits, target):
        if self.classify_loss_type == 'bce':
            loss_i = F.binary_cross_entropy_with_logits(
                logits, target, reduction='none')
            classify_loss = loss_i.sum(-1).mean()

        elif self.classify_loss_type == 'sign_bce':
            pred_occ = torch.clamp(torch.sign(logits), min=0)
            loss_i = F.binary_cross_entropy(
                pred_occ, target, reduction='none')
            classify_loss = loss_i.sum(-1).mean()

        elif self.classify_loss_type == "sigmoid_l1":
            classify_loss = nn.L1Loss(reduction='mean')(torch.sigmoid(logits), target)

        elif self.classify_loss_type == "linear_l1":
            input = torch.max(torch.min(logits, logits * 0.002 + 0.998), logits * 0.002)
            classify_loss = nn.L1Loss(reduction='mean')(input, target)

        else:
            print("wrong classify_loss_type:", self.classify_loss_type)
            return
        return classify_loss

    def decode(self, input_points_for_completion, z, features, teacher=False):
        ''' Returns occupancy probabilities for the sampled points.
        :param input_points_for_completion: points
        :param z: latent code z
        :param features: latent conditioned features
        :return:
        '''

        return_values = self.decoder(input_points_for_completion, z, features)


        logits = return_values[-1]
        p_r = dist.Bernoulli(logits=logits)
        return p_r, return_values[:-1]

    def forward(self, input_points_for_completion, input_features_for_completion, cls_codes_for_completion=None, sample=False):
        '''
        Performs a forward pass through the network.
        :param input_points_for_completion (tensor): sampled points
        :param input_features_for_completion (tensor): conditioning input
        :param cls_codes_for_completion: class codes for input shapes.
        :param sample (bool): whether to sample for z
        :param kwargs:
        :return:
        '''
        device = input_features_for_completion.device
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.to(device).float()
            input_features_for_completion = torch.cat([input_features_for_completion, cls_codes_for_completion], dim=-1)
        '''Encode the inputs.'''
        batch_size = input_points_for_completion.size(0)
        z = None # self.get_z_from_prior((batch_size,), device, sample=sample)
        p_r,_ = self.decode(input_points_for_completion, z, input_features_for_completion)[0] #.logits .probs

    
        return p_r









