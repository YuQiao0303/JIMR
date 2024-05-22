# must import open3d before pytorch.
import logging
import os
import sys
import trimesh
from util.icp import icp

import numpy as np
import pandas as pd
import pyquaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
spconv2 = True
try:
    import spconv.pytorch as spconv
    from spconv.pytorch.modules import SparseModule
except:
    spconv2 = False
    import spconv
    from spconv.modules import SparseModule
import functools
from collections import OrderedDict

from lib.ext_PG_OP.pytorch import ext_pointgroup_ops
from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from util.utils import mesh_fit_scan,pc_fit_scan,pc_fit2standard,pc_fit2standard_torch,pc_cano2pcn_torch,pc_pcn2cano
from util.utils import standard_7_dim_bbox_to_bbox_surface_dist, _bbox_pred_to_bbox
# from model.bsp import CompNet, PolyMesh
from model.onet import *
from model.pcn import *
from lib.chamfer_distance.chamfer_distance import ChamferDistance


from util.config import cfg

from visualization_utils import vis_actors_vtk, get_pc_actor_vtk,get_mesh_actor_vtk, \
                        vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
# from lib.iou3d_nms import iou3d_nms_utils
from lib.rotated_iou import cal_giou_3d, cal_iou_3d
# from lib.softgroup import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
#                    get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
#                    voxelization_idx)
from util.bbox import BBoxUtils
BBox = BBoxUtils()

from util.consts import RFS2CAD_arr, CAD_weights





class AttributeDict(dict):
    def __init__(self, d=None):
        if d is not None:
            for k, v in d.items():
                self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

# current 1x1 conv in spconv2x has a bug. It will be removed after the bug is fixed
class Custom1x1Subm3d(spconv.SparseConv3d):
    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape, input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            if not spconv2:
                self.i_branch = spconv.SparseSequential(
                    spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False))
            else:

                self.i_branch = spconv.SparseSequential(
                    Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False)
                )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.LeakyReLU(0.01),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.LeakyReLU(0.01),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        to_add = self.i_branch(identity)
        if not spconv2:
            output.features += to_add.features
        else:
            output = output.replace_feature(output.features + to_add.features)

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.LeakyReLU(0.01),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
                  for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.LeakyReLU(0.01),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False,
                                    indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.LeakyReLU(0.01),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False,
                                           indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                if i == 0:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[0] * 2, nPlanes[0], norm_fn,
                                                             indice_key='subm{}'.format(indice_key_id))
                else:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[0], nPlanes[0], norm_fn,
                                                             indice_key='subm{}'.format(indice_key_id))

            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            if not spconv2:
                output.features = torch.cat((identity.features, output_decoder.features), dim=1)
            else:
                output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))

            output = self.blocks_tail(output)

        return output


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        m = cfg.m  # 16 or 32
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.semantic_classes = cfg.classes
        self.ignore_label = cfg.non_cad_labels

        self.group_method = cfg.group_method
        self.score_thr = cfg.score_thr
        self.TEST_NPOINT_THRESH = cfg.TEST_NPOINT_THRESH
        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre
        self.class_numpoint_mean = cfg.class_numpoint_mean

        self.merge_mode = cfg.merge_mode  # # mean, weighted_mean
        self.merge_sample_mode = cfg.merge_sample_mode
        self.score_type = cfg.score_type # cls_centerness, cls

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs
        self.prepare_epochs_2 = cfg.prepare_epochs_2

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        self.angle_parameter = cfg.angle_parameter
        self.fix_angle_distance = cfg.fix_angle_distance

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        input_c = 0
        if cfg.use_coords:
            input_c += 3
        if cfg.use_rgb:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, 2 * m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([2 * m, 2 * m, 4 * m, 4 * m, 6 * m, 6 * m, 8 * m], norm_fn, block_reps, block,
                           indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(2 * m),
            nn.LeakyReLU(0.01)
        )

        #### semantic segmentation branch
        self.linear = nn.Linear(2 * m, classes)  # bias(default): True

        #### offset branch
        self.offset = nn.Sequential(
            nn.Linear(2 * m, m, bias=True), # bias default is true anyway
            norm_fn(m),
            nn.LeakyReLU(0.01),
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### bbox branch 
        self.bbox = nn.Sequential(
            nn.Linear(2 * m, 2* m), # m = 16
            norm_fn(2* m),
            nn.LeakyReLU(0.01),
        )
        self.bbox_linear = nn.Linear(2*m, 6, bias=True)



        if self.angle_parameter == 'bin':
            self.angle_linear = nn.Linear(2*m, 12*2)
        else:
            self.angle_linear = nn.Linear(2*m, 3, bias=True)



        # # phase2 bbox score branch
        # self.bbox_score_phase2 = nn.Sequential(
        #     nn.Linear(1024, 512, bias=True), # input: output features of PCN encoder
        #     norm_fn(512),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(512, 1),
        # )
        ### phase 2
        self.cls_score = nn.Sequential(
            nn.Linear(1024, 8, bias=True),  # input: output features of PCN encoder
        )
        self.seg_score = nn.Sequential(
            nn.Linear(1024, 1, bias=True), # input: output features of PCN encoder
        )

        self.bbox_score2 = nn.Sequential(
            nn.Linear(1024, 1, bias=True), # input: output features of PCN encoder
        )

        self.mesh_score = nn.Sequential(
            nn.Linear(1024, 1, bias=True), # input: output features of PCN encoder
        )


        #### init

        self.apply(self.set_bn_init)

        self.onet = ONet(cfg)
        self.pcn = PCN()
        
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                        'bbox':self.bbox,

                          'bbox_linear':self.bbox_linear,
                      'angle_linear':self.angle_linear,

                      'onet':self.onet, 'pcn':self.pcn, # added by Qiao
                      }

        #### fix weights
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:

            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Loaded pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict,
                                                                                    prefix=m))




        ### load pretrained occupancy net
        if cfg.use_pretrained_onet:
            onet_pretrain_dict = torch.load(os.path.join('datasets/onet', 'ONet.pth'))
            self.onet.load_state_dict(onet_pretrain_dict, strict=True)
            print(f"Loaded pretrained O-Net: #params = {sum([p.numel() for p in self.onet.parameters()])}")

        ### load pretrained pcn
        if cfg.use_pretrained_pcn:
            pcn_pretrain_dict = torch.load(os.path.join('datasets/pcn', 'pcn.pth'))
            self.pcn.load_state_dict(pcn_pretrain_dict, strict=False)
            print(f"Loaded pretrained PCN: #params = {sum([p.numel() for p in self.onet.parameters()])}")

        ### fix ONet

        if cfg.fix_onet_decoder:
            for param in self.onet.decoder.parameters():
                param.requires_grad = False
            self.onet.decoder.eval()
        for param in self.onet.teacher_encoder.parameters(): 
            param.requires_grad = False
        self.onet.teacher_encoder.eval()

        if cfg.fix_pcn:
            for param in self.pcn.parameters():
                param.requires_grad = False
            self.pcn.eval()

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)



    def merge_cluster_values(self, clusters_idx, clusters_offset, feats, semantics,semantic_scores, pred_7_dim_bbox,
                             pred_bbox_param,pred_angle_param, coords, fullscale, scale,
                              mode,

                             K = 2048):
        '''

        :param clusters_idx:
        :param clusters_offset:
        :param feats:
        :param semantics:
        :param semantic_scores:
        :param pred_7_dim_bbox:
        :param pred_angle_param:
        :param coords:
        :param fullscale:
        :param scale:
        :param mode:
        :param merge_mode: str: mean, weighted_mean
        :param merge_sample_mode:  'all',
        :param score_type: str: cls_centerness, cls
        :param K:
        :return:
        '''
        merge_mode = self.merge_mode  # # mean, weighted_mean
        merge_sample_mode = self.merge_sample_mode
        score_type = self.score_type  # cls_centerness, cls

        clusters_angle_flip = None

        c_idxs = clusters_idx[:, 1].cuda().long()

        clusters_points_feats = feats[c_idxs]  # [sumNPoint, C]
       
        clusters_points_coords = coords[c_idxs]  # [sumNPoint, 3]
        clusters_points_semantics = semantics[c_idxs]  # [sumNPoint, 8]
        cluster_points_semantic_scores = torch.softmax(semantic_scores[c_idxs],dim=1 ) # [sumNPoint, 8]
        cluster_points_best_semantic_scores = torch.max(cluster_points_semantic_scores,dim=1)[0]  # [sumNPoint, 1]


        # get the semantic label of each proposal
        clusters_semantics = pointgroup_ops.sec_mean(clusters_points_semantics, clusters_offset.cuda())  # [nCluster, 8]

        if 'weighted' in merge_mode:
            # get cluster centers: geometrical center (max - min) and barycenter (mean)
            if 'centerness' in score_type:
                clusters_coords_min_ori = pointgroup_ops.sec_min(clusters_points_coords,
                                                                 clusters_offset.cuda())  # (nCluster, 3), float
                clusters_coords_max_ori = pointgroup_ops.sec_max(clusters_points_coords,
                                                                 clusters_offset.cuda())  # (nCluster, 3), float
                clusters_coords_mean = pointgroup_ops.sec_mean(clusters_points_coords, clusters_offset.cuda())  # (nCluster, 3), float # barycenter
                clusters_centroid = (clusters_coords_max_ori + clusters_coords_min_ori) / 2  # (nCluster, 3), float

                clusters_points_centroid = torch.index_select(clusters_centroid, 0,
                                                              clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
                # clusters_points_barycenter = torch.index_select(clusters_coords_mean, 0,
                #                                               clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float

                # get centerness:
                clusters_coords_min = pointgroup_ops.sec_min(clusters_points_coords,
                                                             clusters_offset.cuda())  # (nCluster, 3), float
                clusters_coords_max = pointgroup_ops.sec_max(clusters_points_coords,
                                                             clusters_offset.cuda())  # (nCluster, 3), float
                clusters_bbox_size = clusters_coords_max - clusters_coords_min


                clusters_points_coords_min =  torch.index_select(clusters_coords_min, 0,clusters_idx[:, 0].cuda().long())
                clusters_points_coords_max =  torch.index_select(clusters_coords_max, 0,clusters_idx[:, 0].cuda().long())

                cluster_points_distance2bbox_faces = torch.stack([
                    torch.abs(clusters_points_coords[:, 0] - clusters_points_coords_min[:, 0]),
                    torch.abs(clusters_points_coords[:, 0] - clusters_points_coords_max[:, 0]),
                    torch.abs(clusters_points_coords[:, 1] - clusters_points_coords_min[:, 1]),
                    torch.abs(clusters_points_coords[:, 1] - clusters_points_coords_max[:, 1]),
                    torch.abs(clusters_points_coords[:, 2] - clusters_points_coords_min[:, 2]),
                    torch.abs(clusters_points_coords[:, 2] - clusters_points_coords_max[:, 2]),
                ],1) # [nPoints, 6]
                centerness = compute_centerness(cluster_points_distance2bbox_faces) # [nPoints]



            scores = torch.ones_like(cluster_points_best_semantic_scores)
            if 'cls' in score_type:
                print('cls')
                scores *= cluster_points_best_semantic_scores #.unsqueeze(1)
            if 'centerness' in score_type:
                print('centerness')
                scores *= centerness


        # bbox: top N or weighted mean

        if merge_sample_mode == 'all':


            merge_dist_then_decode = False
            if merge_dist_then_decode:
                if merge_mode == 'mean':

                    clusters_bboxes = pointgroup_ops.sec_mean(pred_7_dim_bbox[c_idxs],
                                                              clusters_offset.cuda())  # [nCluster, 7]
                else:

                    clusters_bboxes = ext_pointgroup_ops.sec_weighted_mean(pred_7_dim_bbox[c_idxs], scores.cuda(),
                                                                           clusters_offset.cuda())  # [nCluster, 7]


            else:
                if merge_mode == 'mean':
                    clusters_bboxes = pointgroup_ops.sec_mean(pred_7_dim_bbox[c_idxs], clusters_offset.cuda())  # [nCluster, 7]
                else:
                    clusters_bboxes = ext_pointgroup_ops.sec_weighted_mean(pred_7_dim_bbox[c_idxs],scores.cuda(),
                                                              clusters_offset.cuda())  # [nCluster, 7]


            if self.angle_parameter == 'Mobius_flip':
                pred_angle_flip = pred_angle_param[:, 2]  # Mobius_flip
                pred_angle_flip_sigmoid = torch.sigmoid(pred_angle_flip[c_idxs]) # between 0-1
                if merge_mode == 'mean':
                    clusters_angle_flip = pointgroup_ops.sec_mean(pred_angle_flip_sigmoid.unsqueeze(1), clusters_offset.cuda())
                else:
                    clusters_angle_flip = ext_pointgroup_ops.sec_weighted_mean(pred_angle_flip_sigmoid.unsqueeze(1),scores,
                                                                  clusters_offset.cuda())


                clusters_bboxes[:,6]  += torch.round(clusters_angle_flip.view(-1)) * torch.tensor(3.14159)
            elif self.angle_parameter == 'bin':
                clusters_points_angles_label = pred_angle_param[c_idxs, :12]  # [sumNPoint, 12]
                clusters_points_angles_residual = pred_angle_param[c_idxs, 12:]  # [sumNPoint, 12]
                # get mean angle as the bbox angle
                clusters_points_angles_label = torch.softmax(clusters_points_angles_label, dim=1)

                if merge_mode == 'mean':
                    clusters_angles_label_mean = pointgroup_ops.sec_mean(clusters_points_angles_label,
                                                                         clusters_offset.cuda())  # (nCluster, 12), float
                    clusters_angles_residual_mean = pointgroup_ops.sec_mean(clusters_points_angles_residual,
                                                                            clusters_offset.cuda())  # (nCluster, 12), float
                else:

                    clusters_angles_label_mean = ext_pointgroup_ops.sec_weighted_mean(clusters_points_angles_label,scores,
                                                                         clusters_offset.cuda())  # (nCluster, 12), float
                    clusters_angles_residual_mean = ext_pointgroup_ops.sec_weighted_mean(clusters_points_angles_residual,scores,
                                                                            clusters_offset.cuda())  # (nCluster, 12), float

                # decode angles
                clusters_angles_label_mean = torch.argmax(clusters_angles_label_mean, dim=1)  # [nCluster, ] long
                clusters_angles_residual_mean = torch.gather(clusters_angles_residual_mean * np.pi / 12, 1,
                                                             clusters_angles_label_mean.unsqueeze(1)).squeeze(1)
                # detach !!!
                clusters_angles = BBox.class2angle_cuda(clusters_angles_label_mean,
                                                        clusters_angles_residual_mean)#.detach()
                clusters_bboxes[:, 6] = clusters_angles


     

        ######### canonical transformation


        '''move'''


        cluster_points_bbox_branch_center = torch.index_select(clusters_bboxes[:, 0:3], 0,
                                                        clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_points_coords -= cluster_points_bbox_branch_center  # clusters_points_centroid


        # rotate
        clusters_points_angles = clusters_bboxes[:,6][clusters_idx[:, 0].long()]
        cos_ = torch.cos(-clusters_points_angles)
        sin_ = torch.sin(-clusters_points_angles)

        clusters_points_coords[:, 0], clusters_points_coords[:, 1] = clusters_points_coords[:,
                                                                     0] * cos_ - clusters_points_coords[:,
                                                                                 1] * sin_, clusters_points_coords[:,
                                                                                            0] * sin_ + clusters_points_coords[
                                                                                                        :, 1] * cos_

        # # scale
        soft_scale=False
        soft_center = False
        if soft_scale:
            clusters_coords_min = pointgroup_ops.sec_min(clusters_points_coords,
                                                         clusters_offset.cuda())  # (nCluster, 3), float, xyz
            clusters_coords_max = pointgroup_ops.sec_max(clusters_points_coords,
                                                         clusters_offset.cuda())  # (nCluster, 3), float, xyz

            if soft_center:
                clusters_bbox_size = clusters_coords_max - clusters_coords_min
                clusters_points_coords =  clusters_points_coords + cluster_points_bbox_branch_center - clusters_points_centroid
            else:
                clusters_bbox_size = 2*torch.maximum(torch.abs(clusters_coords_min), torch.abs(clusters_coords_max))  # (nCluster, 3), float, xyz

            bbox_branch_residual_scale = clusters_bboxes[:, 3:6] - clusters_bbox_size  #  # (nCluster, 3), float, xyz
            residual_scale = torch.maximum(bbox_branch_residual_scale, - 0.0 * clusters_bbox_size) # 0.1
            cluster_scale = clusters_bbox_size + residual_scale
            clusters_bboxes[:, 3:6] = cluster_scale


        cluster_scale_max = torch.max(clusters_bboxes[:, 3:6], dim=1)[0]  # [0] values, [1] indices # sumNPoint

        cluster_points_bbox_scale = torch.index_select(cluster_scale_max, 0,
                                                        clusters_idx[:, 0].cuda().long())  # (sumNPoint)
        clusters_points_coords[:,0] /= cluster_points_bbox_scale
        clusters_points_coords[:,1] /= cluster_points_bbox_scale
        clusters_points_coords[:,2] /= cluster_points_bbox_scale
        #
        # swap axes : xyz  ->  -z,-x,y.  so: x = -y, y = z, z = -x
        cano_clusters_points_coords = torch.zeros_like(clusters_points_coords)
        cano_clusters_points_coords[:, 0] = -clusters_points_coords[:, 1]
        cano_clusters_points_coords[:, 1] = clusters_points_coords[:, 2]
        cano_clusters_points_coords[:, 2] = -clusters_points_coords[:, 0]

        vis_cano = False
        if vis_cano:
            from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
            ### match GT by max IoU (point cloud) # added by Qiao

            for i in range(clusters_bboxes.shape[0]):
                print(i)
                if i not in [15,18,19]:
                    continue
                # cluster_pc = cano_clusters_points_coords[clusters_offset[i].long():clusters_offset[i + 1].long()].detach().cpu().numpy()
                cluster_pc = clusters_points_coords[clusters_offset[i].long():clusters_offset[i + 1].long()].detach().cpu().numpy()
                cano_reference_pc = np.array(
                        [
                            [0.5,0.5,0.5],
                            [0.5,0.5,-0.5],
                            [0.5,-0.5,0.5],
                            [0.5,-0.5,-0.5],
                            [-0.5,0.5,0.5],
                            [-0.5,0.5,-0.5],
                            [-0.5,-0.5,0.5],
                            [-0.5,-0.5,-0.5],
                        ]
                    )

         
                print('pred_scale',clusters_bboxes[i,3:6])
                print('cluster_bbox_size',clusters_bbox_size[i])
                print('residual_scale',residual_scale[i])
                print('max min xyz:',
                      '(', np.min(cluster_pc[:, 0]), np.max(cluster_pc[:, 0]), ')',
                      '(', np.min(cluster_pc[:, 1]), np.max(cluster_pc[:, 1]), ')',
                      '(', np.min(cluster_pc[:, 2]), np.max(cluster_pc[:, 2]), ')',

                      )
                vis_actors_vtk([
                    # get_pc_actor_vtk(coords[coords_id].detach().cpu().numpy())
                    # get_pc_actor_vtk(clusters_points_coords[:,1:][clusters_offset[i].long():clusters_offset[i+1].long()]
                    #                  .detach().cpu().numpy())
                    get_pc_actor_vtk(cluster_pc),
                    get_pc_actor_vtk(cano_reference_pc,color = (1,0,0)),
                    # get_pc_actor_vtk(clusters_points_coords[clusters_offset[i].long():clusters_offset[i + 1].long()]
                    #                  .detach().cpu().numpy(),color=(1,0,0))

                ])

        return clusters_bboxes, clusters_angle_flip, clusters_semantics,clusters_points_coords,cano_clusters_points_coords

    def forward(self, data, training_mode='train'):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda

        phase1: semantic segmentation, point-wise offset, point-wise angle.
        phase2: instance clustering, confidence score, bbox params.
        '''
        input = data['input']
        input_map = data['input_map']
        coords = data['coords'] # from coords_float in model_fn
        batch_idxs = data['batch_idxs']
        # batch_offsets = data['batch_offsets']
        epoch = data['epoch']



        ret = {}
      
        output = self.input_conv(input)
       
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)  # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]  # (N), long

        semantic_preds_CAD = torch.from_numpy(RFS2CAD_arr[semantic_preds.cpu()]).long().cuda()
        semantic_preds_CAD[semantic_preds_CAD == -1] = 8
        semantic_scores_CAD = F.one_hot(semantic_preds_CAD, 9).float()

        ret['semantic_scores'] = semantic_scores
        ret['semantic_scores_CAD'] = semantic_scores_CAD

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)  # (N, 3), float32

        ret['pt_offsets'] = pt_offsets


        #### bbox
        pt_bboxes_feats = self.bbox(output_feats)
        pt_bboxes = self.bbox_linear(pt_bboxes_feats)
        pt_bboxes[:, :6] = torch.exp(pt_bboxes[:, :6])  # the first 6 parameters are distances and should be positive

        # pt_angle_feats = self.angle(output_feats)
        pt_angles = self.angle_linear(pt_bboxes_feats)

        pt_bboxes = torch.cat((pt_bboxes,pt_angles),dim=1)
        if 'Mobius' in self.angle_parameter:
            pred_7_dim_bbox = _bbox_pred_to_bbox(coords,
                                             pt_bboxes[:,:8], 'Mobius')  # convert the 8-dim fcaf mobius bbox into 7-dim standard bbox
        else: # sin_cos , naive
            # print('not mobius')
            pred_7_dim_bbox = _bbox_pred_to_bbox(coords, pt_bboxes[:,], self.angle_parameter,self.fix_angle_distance)

        pred_angle_param = pt_angles
        # print(pred_7_dim_bbox)
        ret['pt_bboxes'] = (pred_7_dim_bbox,pred_angle_param)


        if epoch > self.prepare_epochs: # cluster

            if self.group_method == 'softgroup':
                proposals_idx, proposals_offset,proposals_semantic = self.soft_grouping(semantic_scores,pt_offsets,batch_idxs,coords)
                proposals_semantic_CAD = torch.from_numpy(RFS2CAD_arr[proposals_semantic.cpu()]).long().cuda()
                proposals_semantic_CAD[proposals_semantic_CAD == -1] = 8
                clusters_semantics = proposals_semantic_CAD

            elif self.group_method == 'dimr':
                proposals_idx, proposals_offset = self.dimr_grouping(semantic_preds_CAD,pt_offsets,batch_idxs,coords)


            ret['proposal_info'] = ( proposals_idx, proposals_offset)
           

            clusters_bboxes, clusters_angle_flip, clusters_semantics,clusters_points_coords,cano_clusters_points_coords\
                = self.merge_cluster_values(
                proposals_idx, proposals_offset, output_feats, semantic_scores_CAD, semantic_scores,
                pred_7_dim_bbox, pt_bboxes[:6], pred_angle_param,coords,
                self.score_fullscale, self.score_scale, self.mode)


            ret['proposal_bboxes'] = clusters_bboxes,clusters_angle_flip
            ret['proposal_semantics'] = clusters_semantics # CAD

            ret['proposal_cano_points'] = cano_clusters_points_coords

        return ret


    def get_batch_offsets(self, batch_idxs, bs):
        '''
        this function is for soft grouping
        :param batch_idxs:
        :param bs:
        :return:
        '''
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets
    def soft_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None):
        proposals_idx_list = []
        proposals_offset_list = []
        proposals_semantic_list = []
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        radius = self.cluster_radius
        mean_active = self.cluster_shift_meanActive #300
        npoint_thr = self.cluster_npoint_thre
        class_numpoint_mean = torch.tensor(
            self.class_numpoint_mean, dtype=torch.float32)
        assert class_numpoint_mean.size(0) == self.semantic_classes
        for class_id in range(self.semantic_classes):
            if class_id in self.ignore_label:
                continue
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.score_thr).nonzero(as_tuple=False).view(-1)
            if object_idxs.size(0) < self.TEST_NPOINT_THRESH:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                               radius, mean_active)
            proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
                                                          start_len.cpu(), npoint_thr, class_id)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            proposals_semantic = (torch.zeros(proposals_offset.shape[0]-1) + class_id)

            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
                proposals_semantic_list.append(proposals_semantic)
        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
            proposals_semantic = torch.cat(proposals_semantic_list).int()
        else:
            proposals_idx = torch.zeros((0, 2), dtype=torch.int32)
            proposals_offset = torch.zeros((0, ), dtype=torch.int32)
            proposals_semantic = torch.zeros((0, ), dtype=torch.int32)
        return proposals_idx, proposals_offset,proposals_semantic

    def dimr_grouping(self,semantic_preds_CAD,pt_offsets,batch_idxs,coords):
        object_idxs = torch.nonzero(semantic_preds_CAD != 8).view(
            -1)  # only get CAD objects, mask out floor and wall and non-CADs.
        batch_size = batch_idxs.max() + 1 # added by Qiao
        batch_idxs_ = batch_idxs[object_idxs]
        # batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
        batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)
        coords_ = coords[object_idxs]
        pt_offsets_ = pt_offsets[object_idxs]
        semantic_preds_ = semantic_preds_CAD[object_idxs].int().cpu()
        # single-scale proposal gen (pointgroup)
        if self.training:

            ### BFS clustering on shifted coords
            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                          batch_offsets_, self.cluster_radius,
                                                                          self.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_,
                                                                                     idx_shift.cpu(),
                                                                                     start_len_shift.cpu(),
                                                                                     self.cluster_npoint_thre)
            proposals_idx_shift[:, 1] = object_idxs[
                proposals_idx_shift[:, 1].long()].int()  # remap: sumNPoint --> N

            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int, start/end index for each proposal, e.g., [0, c1, c1+c2, ..., c1+...+c_nprop = sumNPoint], same information as cluster_id, just in the convinience of cuda operators.

            ### BFS clustering on original coords
            idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_,
                                                              self.cluster_radius, self.cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_, idx.cpu(),
                                                                         start_len.cpu(), self.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int

            ### merge two type of clusters
            proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]

            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)  # .long().cuda()
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]), dim=0)  # .cuda()
            # why [1:]: offset is (0, c1, c2), offset_shift is (0, d1, d2) + c2, output is (0, c1, c2, c2+d1, c2+d2)

        # multi-scale proposal gen (naive, maskgroup)
        else:

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                          batch_offsets_, 0.01,
                                                                          self.cluster_shift_meanActive)
            proposals_idx_shift_0, proposals_offset_shift_0 = pointgroup_ops.bfs_cluster(semantic_preds_,
                                                                                         idx_shift.cpu(),
                                                                                         start_len_shift.cpu(),
                                                                                         self.cluster_npoint_thre)
            proposals_idx_shift_0[:, 1] = object_idxs[proposals_idx_shift_0[:, 1].long()].int()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                          batch_offsets_, 0.03,
                                                                          self.cluster_shift_meanActive)
            proposals_idx_shift_1, proposals_offset_shift_1 = pointgroup_ops.bfs_cluster(semantic_preds_,
                                                                                         idx_shift.cpu(),
                                                                                         start_len_shift.cpu(),
                                                                                         self.cluster_npoint_thre)
            proposals_idx_shift_1[:, 1] = object_idxs[proposals_idx_shift_1[:, 1].long()].int()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                          batch_offsets_, 0.05,
                                                                          self.cluster_shift_meanActive)
            proposals_idx_shift_2, proposals_offset_shift_2 = pointgroup_ops.bfs_cluster(semantic_preds_,
                                                                                         idx_shift.cpu(),
                                                                                         start_len_shift.cpu(),
                                                                                         self.cluster_npoint_thre)
            proposals_idx_shift_2[:, 1] = object_idxs[proposals_idx_shift_2[:, 1].long()].int()

            idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, 0.03,
                                                              self.cluster_meanActive)
            proposals_idx_0, proposals_offset_0 = pointgroup_ops.bfs_cluster(semantic_preds_, idx.cpu(),
                                                                             start_len.cpu(),
                                                                             self.cluster_npoint_thre)
            proposals_idx_0[:, 1] = object_idxs[proposals_idx_0[:, 1].long()].int()

            _offset = proposals_offset_0.size(0) - 1
            proposals_idx_shift_0[:, 0] += _offset
            proposals_offset_shift_0 += proposals_offset_0[-1]

            _offset += proposals_offset_shift_0.size(0) - 1
            proposals_idx_shift_1[:, 0] += _offset
            proposals_offset_shift_1 += proposals_offset_shift_0[-1]

            _offset += proposals_offset_shift_1.size(0) - 1
            proposals_idx_shift_2[:, 0] += _offset
            proposals_offset_shift_2 += proposals_offset_shift_1[-1]

            proposals_idx = torch.cat(
                (proposals_idx_0, proposals_idx_shift_0, proposals_idx_shift_1, proposals_idx_shift_2), dim=0)
            proposals_offset = torch.cat((proposals_offset_0, proposals_offset_shift_0[1:],
                                          proposals_offset_shift_1[1:], proposals_offset_shift_2[1:]))
        return proposals_idx,proposals_offset


def compute_centerness(bbox_targets):
    '''

    :param bbox_targets: [num_bbox, 6]. Distances to the 6 faces of the bounding boxes.
    :return:
    '''
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    return torch.sqrt(centerness_targets)

