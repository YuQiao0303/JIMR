# must import open3d before pytorch.

# import os
# import sys
import trimesh
from util.icp import icp

import numpy as np
import pandas as pd
import pyquaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from util.utils import mesh_fit_scan,pc_fit_scan,pc_fit2standard,pc_fit2standard_torch,pc_cano2pcn_torch,pc_pcn2cano,export_pc_xyz

# from model.bsp import CompNet, PolyMesh
from model.onet import *
from model.pcn import *
# from lib.chamfer_distance.chamfer_distance import ChamferDistance

from util.bbox import BBoxUtils
from util.config import cfg

from visualization_utils import vis_actors_vtk, get_pc_actor_vtk,get_mesh_actor_vtk, \
                        vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization


from model.loss import *
import copy



BBox = BBoxUtils()

from util.consts import RFS2CAD_arr, CAD_weights, not_synthetic_mask,CAD_labels

# from lib.iou3d_nms import iou3d_nms_utils # used for bbox nms. If not use bbox nms, no need to import this

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    not used
    Input:`
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """

    # handle (N,C) (M,C)
    if len(pc1.shape) == 2: pc1 = pc1.unsqueeze(0)
    if len(pc2.shape) == 2: pc2 = pc2.unsqueeze(0)

    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    return dist1, idx1, dist2, idx2

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)







def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    def test_model_fn(batch, model, epoch):

        ### assert batch_size == 1

        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        rgbs = batch['feats'].cuda()  # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        # added by Qiao
        instance_pc = batch['instance_pc'].cuda()  # added by Qiao
        instance_meshes = batch['instance_meshes']  # added by Qiao


        if hasattr(model, 'module'):
            onet = model.module.onet
            pcn = model.module.pcn
            cls_score = model.module.cls_score
            seg_score = model.module.seg_score
            bbox_score = model.module.bbox_score2
            mesh_score = model.module.mesh_score
        else:
            onet = model.onet
            pcn = model.pcn
            cls_score = model.cls_score
            seg_score = model.seg_score
            bbox_score = model.bbox_score2
            mesh_score = model.mesh_score

        # instance_zs = onet.teacher_encoder(instance_pc) # added by Qiao
        instance_zs_valid = batch['instance_zs_valid'].cuda()  # (total_nInst), int, cuda
        instance_labels = batch['instance_labels'].cuda()  # (N), long, cuda, 0~total_nInst, -100
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        instance_info = batch['instance_info'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_bboxes = batch['instance_bboxes'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)

        semantic_labels = batch['labels'].cuda()
        instance_shapenet_catids = batch['instance_shapenet_catids']
        instance_semantic_CAD = batch['instance_semantic_CAD'].cuda() #


        if cfg.use_coords:
            feats = coords_float

        if cfg.use_rgb:
            feats = torch.cat((rgbs, feats), 1)

        use_gt_seg_and_bbox = cfg.use_gt_seg_and_bbox
        '''Phase 1'''
        if not use_gt_seg_and_bbox:

            voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)



            model_inp = {
                'input': input_,
                'input_map': p2v_map,
                'coords': coords_float,
                'batch_idxs': coords[:, 0].int(),
                'batch_offsets': batch_offsets,
                'epoch': epoch,
            }

            ret = model(model_inp, 'test')
            if hasattr(cfg,'get_param_num'):
                if cfg.get_param_num:
                    from thop import profile, clever_format
                    flops, params = profile(model, inputs=(model_inp,))
                    print("------- params: %.3fMB ------- flops: %.3fG" % (params / (1000 ** 2), flops / (1000 ** 3)))
                    # print("------- params: %.2fMB ------- flops: %.2fG" % ( params / ((2**10) ** 2), flops / ((2**10) ** 3)))
                    # flops, params = clever_format([flops, params], '%3.3f')
                    # print('flops', flops)
                    # print('params', params)
                    return


            semantic_scores = ret['semantic_scores_CAD']  # (N, nClass) float32, cuda
            semantic_scores_all_cls = ret['semantic_scores'] # all stage-1 classes including walls and floors
            pt_offsets = ret['pt_offsets']  # (N, 3), float32, cuda

            # pt_angles = ret['pt_angles']  # [N, 24]
            pred_7_dim_bbox, pred_angle_param = ret['pt_bboxes']



            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
            semantic_pred_all_cls = semantic_scores_all_cls.max(1)[1]  # (N) long, cuda

        '''Phase 1.5'''
        if epoch > cfg.prepare_epochs and not use_gt_seg_and_bbox:
            # scores, proposals_idx, proposals_offset, proposals_semantics = ret['proposal_scores']
            (proposals_idx, proposals_offset) =ret['proposal_info']
            clusters_bboxes, clusters_angle_flip = ret['proposal_bboxes']
            proposals_semantics = ret['proposal_semantics']
            cano_clusters_points_coords = ret['proposal_cano_points']

        '''Phase 2'''
        if epoch > cfg.prepare_epochs_2 :
            if not use_gt_seg_and_bbox:
                '''get clusters'''
                N = coords.shape[0]
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int,
                                             device=clusters_bboxes.device)  # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

                ori_proposal_ids = torch.arange(0, proposals_offset.shape[0] - 1)  # added by Qiao

                cluster_semantic_score1 = proposals_semantics
                cluster_semantic_id1 = cluster_semantic_score1.max(1)[1]

                '''get proposal scores'''
                rand_k_indices = ext_pointgroup_ops.sec_rand_k_idx(2048, proposals_offset.cuda()).long()
                pred_fix_num_pcs = cano_clusters_points_coords[rand_k_indices]  # [cluster_num, 2048, 3])
                all_cluster_features = pcn.encoder(pc_cano2pcn_torch(pred_fix_num_pcs.detach()))

                pred_seg_score = torch.sigmoid(seg_score(all_cluster_features).view(-1))
                pred_bbox_score = torch.sigmoid(bbox_score(all_cluster_features).view(-1))
                pred_mesh_score = torch.sigmoid(mesh_score(all_cluster_features).view(-1))
                cluster_sem2 = torch.softmax(cls_score(all_cluster_features), dim=1)
                cluster_sem_scores2, cluster_sem_label2 = torch.max(cluster_sem2, dim=1)
                if cfg.cls_label_from == 2:
                    cluster_semantic_id = cluster_sem_label2 # merge phase 1 pred semantic label
                    cluster_sem_scores = torch.gather(cluster_sem2, 1, cluster_semantic_id.unsqueeze(1)).squeeze(1) # a
                else:  # cfg.cls_label_from == 1
                    cluster_semantic_id = cluster_semantic_id1 # phase2 pred
                    cluster_sem_scores = torch.gather(cluster_sem2, 1, cluster_semantic_id.unsqueeze(1)).squeeze(1) # always use phase2 cls score


                # scores_pred = torch.sigmoid(pred_seg_score.view(-1))

                scores_pred = torch.ones_like(pred_seg_score)
                used_score_type_num = 0.0
                if 'seg' in cfg.final_score_type:
                    scores_pred *= pred_seg_score
                    used_score_type_num+=1
                if 'cls' in cfg.final_score_type:
                    scores_pred *= cluster_sem_scores
                    used_score_type_num += 1
                if 'box' in cfg.final_score_type:
                    scores_pred *= pred_bbox_score
                    used_score_type_num += 1
                if 'mesh' in cfg.final_score_type:
                    scores_pred *= pred_mesh_score
                    used_score_type_num += 1
                # scores_pred = torch.pow(scores_pred,1.0/used_score_type_num)

                device = scores_pred.device
                all_clusters_num = pred_seg_score.shape[0]


                ##### some thresholds for filtering / selecting proposals
                ##### score threshold

                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)  # .cpu()

                scores_pred = scores_pred[score_mask]  # the 4 are all cuda
                proposals_pred = proposals_pred[score_mask]
                cluster_semantic_id = cluster_semantic_id[score_mask]
                cluster_semantic_score1 = cluster_semantic_score1[score_mask]
                clusters_bboxes = clusters_bboxes[score_mask]

                # proposals_offset = torch.cat([proposals_offset[0].unsqueeze(0),proposals_offset[1:][score_mask]]  )# added by Qiao
                ori_proposal_ids = ori_proposal_ids[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)  # .cpu() # cpu by Qiao
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                cluster_semantic_id = cluster_semantic_id[npoint_mask]
                cluster_semantic_score1 = cluster_semantic_score1[npoint_mask]
                clusters_bboxes = clusters_bboxes[npoint_mask]
                # proposals_offset = torch.cat([proposals_offset[0].unsqueeze(0),proposals_offset[1:][npoint_mask]] ) # added by Qiao
                ori_proposal_ids = ori_proposal_ids[npoint_mask]
                ##### nms on segmentation
                if cluster_semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(proposals_pred_f,
                                            proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(),
                                                    cfg.TEST_NMS_THRESH)  # int, (nCluster, N)

                clusters = proposals_pred[pick_idxs]  # (nProp, N)
                cluster_scores = scores_pred[pick_idxs]  # (nProp, )
                cluster_semantic_id = cluster_semantic_id[pick_idxs]  # (nProp, ) RFS id
                cluster_semantic_score1 = cluster_semantic_score1[pick_idxs]  # (nProp, C)
                # proposals_offset = torch.cat([proposals_offset[0].unsqueeze(0),proposals_offset[1:][pick_idxs] ]  )# added by Qiao
                ori_proposal_ids = ori_proposal_ids[pick_idxs]
                cluster_bboxes = clusters_bboxes[pick_idxs]

                ### nms on bunding box
                if cfg.TEST_BBOX_NMS:
                    if len(pick_idxs)==0:
                        bbox_nms_pick_idx = np.empty(0)
                    else:
                        # bbox_ious: nProposal, nProposal. # right here
                        # print('cluster_bboxes.shape',cluster_bboxes.shape)
                        bbox_ious = iou3d_nms_utils.boxes_iou3d_gpu(cluster_bboxes, cluster_bboxes) # it does not have a cpu version
                        # print('bbox_ious.shape', bbox_ious.shape)
                        bbox_nms_pick_idx = non_max_suppression(bbox_ious.cpu().numpy(), cluster_scores.cpu().numpy(),
                                                        cfg.TEST_BBOX_NMS_THRESH)  # int, (nCluster, N)
                        # print('bbox_nms_pick_idx.shape', bbox_nms_pick_idx.shape)

                        clusters = clusters[bbox_nms_pick_idx]  # (nProp, N)
                        cluster_scores = cluster_scores[bbox_nms_pick_idx]  # (nProp, )
                        cluster_semantic_id = cluster_semantic_id[bbox_nms_pick_idx]  # (nProp, ) RFS id
                        cluster_semantic_score1 = cluster_semantic_score1[bbox_nms_pick_idx]  # (nProp, C)
                        # proposals_offset = torch.cat([proposals_offset[0].unsqueeze(0),proposals_offset[1:][pick_idxs] ]  )# added by Qiao
                        ori_proposal_ids = ori_proposal_ids[bbox_nms_pick_idx]
                        cluster_bboxes = cluster_bboxes[bbox_nms_pick_idx]


                n_clusters = clusters.shape[0]

                # selected_cluster_ids = torch.arange(all_clusters_num)[score_mask][npoint_mask][pick_idxs][bbox_nms_pick_idx]
                selected_cluster_ids = ori_proposal_ids.long()  # same with the previous line
                # print('selected_cluster_ids',selected_cluster_ids)
                if len(selected_cluster_ids) == 0: #
                    preds = {'has_instance': False}
                    return preds


                selected_cluster_features = all_cluster_features[selected_cluster_ids]

                selected_pred_fix_num_pcs = pred_fix_num_pcs[selected_cluster_ids]
                # cluster_bboxes = clusters_bboxes[selected_cluster_ids]
                fix_num_canonical_cluster_pcs = selected_pred_fix_num_pcs


                # reconstruction
                ### match GT by max IoU (point cloud) # added by Qiao (matching with gt meshes)
                ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels,
                                              instance_pointnum)  # (nProposal, nInstance), float

                gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long

                all_gt_mesh_instance_idxs = torch.cumsum(instance_zs_valid,dim=0).long() -1
                pred_corresponding_gt_mesh_id= all_gt_mesh_instance_idxs[gt_instance_idxs[selected_cluster_ids]]

                cluster_seg_scores = pred_seg_score[selected_cluster_ids]
                cluster_bbox_scores = pred_bbox_score[selected_cluster_ids]
                cluster_mesh_scores = pred_mesh_score[selected_cluster_ids]
                cluster_sem_scores = cluster_sem_scores[selected_cluster_ids]

                cluster_bboxes = cluster_bboxes.detach().cpu().numpy()
                if epoch > cfg.prepare_epochs_3 :
                    ''' get meshes'''

                    complete_pcs = pc_pcn2cano(pcn.decoder(selected_cluster_features))  # complete
                  
                    zs = onet.student_encoder(complete_pcs)
                    cluster_meshes = onet.generator.generate_mesh(zs, cls_codes=None, teacher=True)

                    if cfg.save_cano_mesh:
                        cano_meshes = copy.deepcopy(cluster_meshes)

                    final_instance_num = fix_num_canonical_cluster_pcs.shape[0]

                    ### mesh post-processings.
                    fix_num_canonical_cluster_pcs = fix_num_canonical_cluster_pcs.detach().cpu().numpy()
                    complete_pcs = complete_pcs.detach().cpu().numpy()
                    complete_pcs_world = np.zeros_like(complete_pcs)
                    world_cluster_pcs = []

                    for cid in range(final_instance_num):
                        mesh = cluster_meshes[cid]
                        if mesh is None:
                            continue
                        world_cluster_pc = coords_float[proposals_idx[proposals_offset[selected_cluster_ids[cid]].long():proposals_offset[selected_cluster_ids[cid]+1].long()][:,1].long()]
                        world_cluster_pc = world_cluster_pc.detach().cpu().numpy()
                        world_cluster_pcs.append(world_cluster_pc)
                        complete_pc = complete_pcs[cid]

                        # fit back into world bbox if possible
                        bbox = cluster_bboxes[cid]
                        if bbox is not None:
                            # o3d ICP
                            if len(mesh.vertices) > 0:
                                # if True:
                                visualize_mesh_and_pc = False
                                if visualize_mesh_and_pc:
                                    '''visualize mesh and pc'''

                                    from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                                        get_mesh_actor_vtk  # vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                                    vis_actors_vtk([
                                        # get_pc_actor_vtk(fix_num_canonical_cluster_pc, color = (0,1,0)),
                                        get_pc_actor_vtk(rescale_gt_pc, color=(0, 0, 1)),
                                        get_pc_actor_vtk(complete_pc, color=(1, 0, 0)),
                                        get_mesh_actor_vtk(mesh)
                                    ])

                                '''fit to scan'''
                                mesh = mesh_fit_scan(mesh, bbox)
                                tmesh = mesh  # tmesh = mesh.to_trimesh() if isinstance(mesh, PolyMesh) else mesh

                                visualize_world = False
                                if visualize_world:
                                    from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                                        get_mesh_actor_vtk
                                    vis_actors_vtk([
                                        get_pc_actor_vtk(world_cluster_pc, color=(1, 0, 0)),
                                        get_mesh_actor_vtk(mesh, opacity=0.5)
                                    ])

                                '''ICP'''
                                if hasattr(cfg, 'ICP'):
                                    if cfg.ICP:
                                        source_pc, _ = trimesh.sample.sample_surface_even(tmesh, 2048)
                                        transformed_vertices, fitness = icp(source_pc, world_cluster_pc, mesh.vertices)
                                        mesh.vertices = transformed_vertices
                                else:
                                    source_pc, _ = trimesh.sample.sample_surface_even(tmesh, 2048)
                                    transformed_vertices, fitness = icp(source_pc, world_cluster_pc, mesh.vertices)
                                    mesh.vertices = transformed_vertices
                        cluster_meshes[cid] = mesh

                        # save complete pc
                        complete_pc = pc_fit_scan(complete_pc, bbox)
                        complete_pcs_world[cid] = complete_pc

                    fix_num_world_cluster_pcs = world_cluster_pcs
                if use_gt_seg_and_bbox: # use_gt_seg
                    device = coords_float.device
                    all_instance_num = instance_pc.shape[0]
                    
                    all_point_num = instance_labels.shape[0]
                    valid_instance_id = torch.arange(all_instance_num)[instance_zs_valid.bool() == True].to(device).long()
                    instance_points_idx = torch.arange(all_point_num)[instance_labels != -100].to(device).long() # all instance points idx
                    instance_label_sort = torch.argsort(instance_labels[instance_points_idx], descending=False)

                    proposals_idx = instance_points_idx[instance_label_sort]
                    labels = instance_labels[instance_points_idx][instance_label_sort]
                    proposals_idx = torch.vstack([labels,proposals_idx]).T
                    proposals_offset = torch.cumsum(torch.cat((torch.zeros((1)).to(device),(instance_pointnum)),0), dim=0).long()
                    gt_instance_idxs = torch.arange(all_instance_num).to(device).long()


                    return_dict = \
                        get_selected_cluster_pc(proposals_offset, proposals_idx, coords_float, valid_instance_id,
                                                gt_instance_idxs, proposals_semantics = None, instance_bboxes=instance_bboxes,
                                                pred_bbox=None, instance_pc=instance_pc,use_gt_seg=True )
                    fix_num_canonical_cluster_pcs = return_dict['fix_num_canonical_cluster_pcs']
                    rescale_gt_pcs = return_dict['rescale_gt_pcs']
                    cluster_bboxes = return_dict['cluster_bboxes']
                    fix_num_world_cluster_pcs = return_dict['fix_num_world_cluster_pcs']
                    cluster_valid_mask = return_dict['cluster_valid_mask']

                    temp = torch.sort(valid_instance_id)[1]
                    gt_mesh_instance_idxs = torch.sort(temp)[1]

                    pred_corresponding_gt_mesh_id = gt_mesh_instance_idxs[cluster_valid_mask.bool()].detach().cpu().numpy()


                    # instance semantic label
                    shapenet_cat_id_str2int = {'04379243':0, '03001627':1, '02871439':2, '04256520':3, '02747177':4,
                                               '02933112':5, '03211117':6, '02808440':7}
                    indices = valid_instance_id[cluster_valid_mask.bool()].detach().cpu().numpy()
                    selected_shapenet_catids = [instance_shapenet_catids[i] for i in indices]
                    instance_semantic = torch.tensor(np.array([shapenet_cat_id_str2int[i ] for i in selected_shapenet_catids])).cuda()
                    cluster_semantic_id = instance_semantic





                    complete_pcs = pcn(pc_cano2pcn_torch(fix_num_canonical_cluster_pcs)) # rotate them to fit pretrained PCN
                    complete_pcs = pc_pcn2cano(complete_pcs)# rotate them to fit pretrained PCN
                    zs = onet.student_encoder(complete_pcs)
                    cluster_meshes = onet.generator.generate_mesh(zs, cls_codes=None, teacher=True)
                    final_instance_num = fix_num_world_cluster_pcs.shape[0]


                    ### mesh post-processings.
                    fix_num_canonical_cluster_pcs = fix_num_canonical_cluster_pcs.detach().cpu().numpy()
                    fix_num_world_cluster_pcs = fix_num_world_cluster_pcs.detach().cpu().numpy()
                    rescale_gt_pcs = rescale_gt_pcs.detach().cpu().numpy()
                    complete_pcs = complete_pcs.detach().cpu().numpy()
                    cluster_bboxes = cluster_bboxes.detach().cpu().numpy()
                    complete_pcs_world  = np.zeros_like(complete_pcs)



                    for cid in range(final_instance_num):
                        mesh = cluster_meshes[cid]
                        if mesh is None:
                            continue
                        fix_num_canonical_cluster_pc = fix_num_canonical_cluster_pcs[cid]
                        fix_num_world_cluster_pc = fix_num_world_cluster_pcs[cid]
                        rescale_gt_pc = rescale_gt_pcs[cid]
                        complete_pc = complete_pcs[cid]

                        # fit back into world bbox if possible
                        bbox = cluster_bboxes[cid]
                        if bbox is not None:
                            # o3d ICP
                            if len(mesh.vertices)>0:
                            # if True:
                                visualize_mesh_and_pc = False
                                if visualize_mesh_and_pc:
                                    '''visualize mesh and pc'''
                                    # print(bbox)
                                    from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                                        get_mesh_actor_vtk #vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization

                                    vis_actors_vtk([
                                        # get_pc_actor_vtk(fix_num_canonical_cluster_pc, color = (0,1,0)),
                                        get_pc_actor_vtk(rescale_gt_pc, color = (0,0,1)),
                                        get_pc_actor_vtk(complete_pc,color = (1,0,0)),
                                        get_mesh_actor_vtk(mesh)
                                    ])

                                '''fit to scan'''
                                mesh = mesh_fit_scan(mesh, bbox)
                                tmesh = mesh  # tmesh = mesh.to_trimesh() if isinstance(mesh, PolyMesh) else mesh


                                visualize_world = False
                                if visualize_world:
                                    from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                                        get_mesh_actor_vtk
                                    vis_actors_vtk([
                                        get_pc_actor_vtk(fix_num_world_cluster_pc, color=(1, 0, 0)),
                                        get_mesh_actor_vtk(mesh,opacity=0.5)
                                    ])

                                '''ICP'''
                                source_pc, _ = trimesh.sample.sample_surface_even(tmesh, 2048)
                                transformed_vertices, fitness = icp(source_pc, fix_num_world_cluster_pc, mesh.vertices)
                                mesh.vertices = transformed_vertices
                        cluster_meshes[cid] = mesh


                        # save complete pc
                        complete_pc = pc_fit_scan(complete_pc,bbox)
                        complete_pcs_world[cid] =complete_pc



        ##### preds
        preds = {}
        preds['has_instance'] = True
        # Phase 1
        if not use_gt_seg_and_bbox:
            preds['semantic_pred'] = semantic_pred
            preds['semantic_pred_all_cls'] = semantic_pred_all_cls
            preds['pt_offsets'] = pt_offsets
            preds['pt_bboxes'] = pred_7_dim_bbox
            preds['pt_angle_param'] = pred_angle_param
        # Phase 1.5
        if epoch > cfg.prepare_epochs:

            if epoch > cfg.prepare_epochs_2:
                preds['gt_seg_ious'] = gt_ious[selected_cluster_ids]
                preds['clusters'] = clusters
                preds['cluster_scores'] = cluster_seg_scores,cluster_bbox_scores,cluster_sem_scores,cluster_mesh_scores
                preds['cluster_final_scores'] = cluster_scores


                preds['cluster_semantic_id'] = cluster_semantic_id

                preds['cluster_bboxes'] = cluster_bboxes
                preds['pred_corresponding_gt_mesh_id'] = pred_corresponding_gt_mesh_id

                if epoch > cfg.prepare_epochs_3:
                    preds['cluster_meshes'] = cluster_meshes
                    preds['cluster_completed_pc'] = complete_pcs_world
                    preds['cluster_partial_pc'] = fix_num_world_cluster_pcs
                    if cfg.save_cano_mesh:
                        preds['cluster_cano_meshes'] = cano_meshes
                    if hasattr(cfg, 'save_cano_pc'):
                        if cfg.save_cano_pc:
                            preds['cluster_cano_completed_pc'] = complete_pcs
                            preds['cluster_cano_partial_pc'] = fix_num_canonical_cluster_pcs

                if cfg.retrieval:
                    preds['cluster_alignment'] = cluster_alignment

        return preds


    def model_fn(batch, model, epoch):

        '''prepare data'''
        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        rgbs = batch['feats'].cuda()  # (N, C), float32, cuda
        labels = batch['labels'].cuda()  # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()  # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        instance_pc = batch['instance_pc'].cuda()  # added by Qiao
        instance_points_for_occ = batch['instance_points_for_occ'].cuda()
        instance_occ = batch['instance_occ'].cuda()


        vis_points_and_occ = False
        if vis_points_and_occ:
            for i in range(instance_points_for_occ.shape[0]):
                from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, get_mesh_actor_vtk, \
                    vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                in_points = instance_points_for_occ[i][instance_occ[i] ==1]
                out_points = instance_points_for_occ[i][instance_occ[i] ==0]
                vis_actors_vtk([
                    get_pc_actor_vtk(in_points.detach().cpu().numpy(),color = (1,0,0)),
                    get_pc_actor_vtk(out_points.detach().cpu().numpy()),

                ])



        if hasattr(model, 'module'):
            onet = model.module.onet
            pcn = model.module.pcn
            cls_score = model.module.cls_score
            seg_score = model.module.seg_score
            bbox_score = model.module.bbox_score2
            mesh_score = model.module.mesh_score
        else:
            onet = model.onet
            pcn = model.pcn
            cls_score = model.cls_score
            seg_score = model.seg_score
            bbox_score = model.bbox_score2
            mesh_score = model.mesh_score
 
        instance_zs_valid = batch['instance_zs_valid'].cuda()  # (total_nInst), int, cuda

        instance_bbox_size = batch['instance_bbox_size'].cuda()
        instance_bboxes = batch['instance_bboxes'].cuda()
        instance_semantic_CAD= batch['instance_semantic_CAD'].cuda()
        instance_shapenet_catids = batch['instance_shapenet_catids'] #

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']


        debug_bbox = False
        if debug_bbox:
            from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, get_mesh_actor_vtk, \
                vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
            from util.utils import standard_7_dim_bbox_to_bbox_surface_dist,_bbox_pred_to_bbox

            # print('instance_shapenet_catids',instance_shapenet_catids) #instance_shapenet_catids ['03001627', '03001627', '03001627', '03001627', None, '04379243', None, None, None, None, None, None, None, None, None, '02933112']
            # print('instance_semantic_CAD',instance_semantic_CAD) #([   1,    1,    1,    1, -100,    0, -100, -100, -100, -100, -100, -100,-100, -100, -100,    5],
            for instance_id in [0,1,2,3,5,15]:

                # real data
                pc = coords_float[instance_labels == instance_id]
                point = pc #[[108,0],:] # randomly select a point

                bbox7 = instance_bboxes[instance_id]
                angle = bbox7[6].repeat(point.shape[0],1)
                # print('point,',point,point.shape)
                # print('bbox7,',bbox7,bbox7.shape)



                bbox_dist = standard_7_dim_bbox_to_bbox_surface_dist(point,bbox7)

                # print('bbox_dist,', bbox_dist, bbox_dist.shape)
                # print('angle,', angle, angle.shape)
                bbox_dist_cat_angle = torch.cat([bbox_dist,angle],dim=1)

                # print('bbox_dist_cat_angle',bbox_dist_cat_angle,bbox_dist_cat_angle.shape)
                resume_bbox7 = _bbox_pred_to_bbox(point,bbox_dist_cat_angle,'naive')
                print('bbox7',bbox7)
                print('resume_bbox7',resume_bbox7.mean(0))
                # print('bbox_dist',bbox_dist)
                # print('point',point)


                # in_points = instance_points_for_occ[i][instance_occ[i] == 1]
                # out_points = instance_points_for_occ[i][instance_occ[i] == 0]
                # vis_actors_vtk([
                #     get_pc_actor_vtk(in_points.detach().cpu().numpy(), color=(1, 0, 0)),
                #     get_pc_actor_vtk(out_points.detach().cpu().numpy()),
                #
                # ])
                break

            return

        if cfg.use_coords:
            feats = coords_float

        if cfg.use_rgb:
            feats = torch.cat((rgbs, feats), 1)


        vis_input = False
        if vis_input:
            print('vis_input')

            from visualization_utils import vis_actors_vtk, get_pc_actor_vtk
            print('coords_float', coords_float)
            print('feats', feats)
            vis_actors_vtk([
                get_pc_actor_vtk(coords_float.detach().cpu().numpy())
            ])
        '''Phase 1: get per points' seg and det results'''
        if not cfg.use_gt_seg_and_bbox: 
            voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

           
            ### model call
            model_inp = {
                'input': input_,
                'input_map': p2v_map,
                'coords': coords_float,
                'batch_idxs': coords[:, 0].int(),
                'batch_offsets': batch_offsets,
                'epoch': epoch,
                ##
                # 'instance_info': instance_info,
                # 'instance_labels':instance_labels,
            }

            ret = model(model_inp)

            semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
            semantic_scores_CAD = ret['semantic_scores_CAD']
            pt_offsets = ret['pt_offsets']  # (N, 3), float32, cuda
            # pt_angles = ret['pt_angles']  # (N, 24)
            pt_bboxes, pred_angle_param = ret['pt_bboxes'] # (N,8)  bbox parameters

            device = semantic_scores.device
        ''' Phase 1.5: get grouped/merged cluster results'''
        if epoch > cfg.prepare_epochs and (not cfg.use_gt_seg_and_bbox):
            proposals_idx, proposals_offset = ret['proposal_info']  # = ( )
            # proposals_idx_soft, proposals_offset_soft = ret['proposal_info_soft']  # = ( )

            pred_cluster_bboxes ,clusters_angle_flip= ret['proposal_bboxes'] #= clusters_bboxes
            clusters_semantics = ret['proposal_semantics']
            cano_clusters_points_coords = ret['proposal_cano_points']
            # match GT by max IoU (point cloud)
            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels,
                                          instance_pointnum)  # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long

        

        '''Phase2: get mesh and other scores'''
        if epoch > cfg.prepare_epochs_2:
            pred_cls_score=None
            pred_seg_score=None

            pred_mesh_score=None


            if not cfg.use_gt_seg_and_bbox:
                valid_cluster_num = instance_zs_valid[gt_instance_idxs].long().sum()
                if valid_cluster_num > 0:
                    # select good clusters, randomly sample fix number of points in them
                    gt_ious = gt_ious * instance_zs_valid[gt_instance_idxs]  # keep CAD instances only, added by Qiao
                    rand_k_indices = ext_pointgroup_ops.sec_rand_k_idx(2048, proposals_offset.cuda()).long()
                    pred_fix_num_pcs = cano_clusters_points_coords[rand_k_indices]  # [cluster_num, 2048, 3])
                    selected_cluster_ids = select_good_clusters(gt_ious, gt_instance_idxs, instance_zs_valid)
                    selected_pred_fix_num_pcs = pred_fix_num_pcs[selected_cluster_ids]
                    selected_pred_cluster_bboxes = pred_cluster_bboxes[selected_cluster_ids]  # [10,7]
                    fix_num_canonical_cluster_pcs = selected_pred_fix_num_pcs

                    # rescale gt template
                    selected_gt_pc_wo_rescale = instance_pc[gt_instance_idxs[selected_cluster_ids]]
                    complete_template_pc_scale_xyz = torch.max(selected_gt_pc_wo_rescale, 1)[0] - \
                                                     torch.min(selected_gt_pc_wo_rescale, 1)[0]  # # [10,3]
                    complete_template_pc_scale_xyz = complete_template_pc_scale_xyz.unsqueeze(1)  # [10,1,3]
                    complete_template_pc_scale_xyz = complete_template_pc_scale_xyz.repeat(1, 2048, 1)  # [10,2048,3]
                    complete_template_pc_scale_xyz = complete_template_pc_scale_xyz.view((-1, 3))  # [20480,3]

                    selected_pred_cluster_bboxes = selected_pred_cluster_bboxes.unsqueeze(1)  # 10,1,7
                    selected_pred_cluster_bboxes = selected_pred_cluster_bboxes.repeat(1, 2048, 1)  # 10,2048,7
                    selected_pred_cluster_bboxes = selected_pred_cluster_bboxes.view(-1, 7)  # 20480,7

                    selected_gt_pc_wo_rescale = selected_gt_pc_wo_rescale.view((-1, 3))  # [20480,3]

                    selected_gt_pc = selected_gt_pc_wo_rescale / complete_template_pc_scale_xyz

                    selected_gt_pc *= selected_pred_cluster_bboxes[:, [4, 5, 3]]
                    scale = torch.max(selected_pred_cluster_bboxes[:, [4, 5, 3]], dim=1)[0]
                    selected_gt_pc[:, 0] /= scale
                    selected_gt_pc[:, 1] /= scale
                    selected_gt_pc[:, 2] /= scale
                    selected_gt_pc = selected_gt_pc.view(-1, 2048, 3)
                    rescale_gt_pcs = selected_gt_pc

                    # Encode clusters
                    # torch.autograd.set_detect_anomaly(True)
                    all_cluster_features = pcn.encoder(pc_cano2pcn_torch(pred_fix_num_pcs.detach()))
                    # selected_cluster_features = pcn.encoder(pc_cano2pcn_torch(selected_pred_fix_num_pcs.detach()))
                    selected_cluster_features = all_cluster_features[selected_cluster_ids]

                    # segmentation score branch
                    pred_seg_score = seg_score(all_cluster_features)

                    # cls score branch
                    pred_cls_score = cls_score(selected_cluster_features)  # (N, nClass), float32, cuda.  (nClass = 8)

                    # bbox score branch
                    pred_bbox_score = bbox_score(selected_cluster_features)

                    # mesh score branch
                    pred_mesh_score = mesh_score(selected_cluster_features)

                    # mesh score

                    # mesh branch
                    pred_complete_pcs = pc_pcn2cano(pcn.decoder(selected_cluster_features))  # complete
                    zs = onet.student_encoder(pred_complete_pcs)  # encode completed point clouds, [cluster_num, 512]
                    input_points_for_completion = instance_points_for_occ[gt_instance_idxs[selected_cluster_ids]]
                    pred_occ_logits = onet.decode(input_points_for_completion, z=None, features=zs)[
                        0].logits  # .logits .probs  [cluster_num, 2048]

                    teacher_zs = onet.teacher_encoder(rescale_gt_pcs)
                    gt_occ = instance_occ[gt_instance_idxs[selected_cluster_ids]]
   
                    complete_pcs = pred_complete_pcs

                    vis_pred_gt_pc = False
                    if vis_pred_gt_pc:
                        from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, get_mesh_actor_vtk, \
                            vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                        # selected_gt_pc = selected_gt_pc.view(-1, 2048, 3)
                        wo_scale_teacher_zs = onet.teacher_encoder(selected_gt_pc_wo_rescale.view(-1, 2048, 3))

                        teacher_meshes = onet.generator.generate_mesh(wo_scale_teacher_zs, cls_codes=None, teacher=True)
                        for i in range(selected_gt_pc.shape[0]):
                            in_points = input_points_for_completion[i][pred_occ_logits[i] > 0]
                            out_points = input_points_for_completion[i][pred_occ_logits[i] < 0]
                            vis_actors_vtk([
                                # get_pc_actor_vtk(selected_gt_pc[i].detach().cpu().numpy()),  # gt pc
                                # # get_pc_actor_vtk(selected_gt_pc_wo_rescale.view((-1, 2048, 3))[i].detach().cpu().numpy()), # gt pc w.o. rescale
                                get_pc_actor_vtk(selected_pred_fix_num_pcs[i].detach().cpu().numpy(), color=(1, 0, 0)),  # pred_cluster_pc red
                                get_pc_actor_vtk(pred_complete_pcs[i].detach().cpu().numpy(), color=(0, 1, 0)), # pred complete pc green
                                # # pcn completed pc
                                #
                                # get_pc_actor_vtk(in_points.detach().cpu().numpy(), color=(1, 0, 0)),
                                # onet predict in mesh points
                                # get_pc_actor_vtk(out_points.detach().cpu().numpy()),
                                get_mesh_actor_vtk(teacher_meshes[i])
                            ])
                            # teacher_meshes[i].export(f"L2_teacher_mesh{i}.ply")
                            # print('saved',i)

                else:
                    print("!!!!!!!!!!!!! no CAD cluster !!!!!!!!!", valid_cluster_num)

            else: # use gt_seg_and_bbox
                device = coords_float.device
                all_instance_num = instance_pc.shape[0]
                valid_cluster_num = instance_zs_valid.long().sum()
              
                all_point_num = instance_labels.shape[0]
                valid_instance_id = torch.arange(all_instance_num)[instance_zs_valid.bool() == True].to(
                    device).long()
                selected_cluster_ids = valid_instance_id
                instance_points_idx = torch.arange(all_point_num)[instance_labels != -100].to(
                    device).long()  # all instance points idx
                instance_label_sort = torch.argsort(instance_labels[instance_points_idx], descending=False)

                proposals_idx = instance_points_idx[instance_label_sort]
                labels = instance_labels[instance_points_idx][instance_label_sort]
                proposals_idx = torch.vstack([labels, proposals_idx]).T
                proposals_offset = torch.cumsum(torch.cat((torch.zeros((1)).to(device), (instance_pointnum)), 0),
                                                dim=0).long()
                gt_instance_idxs = torch.arange(all_instance_num).to(device).long()

                return_dict = \
                    get_selected_cluster_pc(proposals_offset, proposals_idx, coords_float, valid_instance_id,
                                            gt_instance_idxs, proposals_semantics=None,
                                            instance_bboxes=instance_bboxes,
                                            pred_bbox=None, instance_pc=instance_pc, use_gt_seg=True)

                fix_num_canonical_cluster_pcs = return_dict['fix_num_canonical_cluster_pcs']

                rescale_gt_pcs = return_dict['rescale_gt_pcs']
                # cluster_bboxes = return_dict['cluster_bboxes']
                # fix_num_world_cluster_pcs = return_dict['fix_num_world_cluster_pcs']
                cluster_valid_mask = return_dict['cluster_valid_mask']
                # print('cluster_valid_mask',cluster_valid_mask)

                input_points_for_completion = instance_points_for_occ[gt_instance_idxs[selected_cluster_ids]][cluster_valid_mask.bool()]

                gt_occ = instance_occ[gt_instance_idxs[selected_cluster_ids]][cluster_valid_mask.bool()]
                teacher_zs = onet.teacher_encoder(rescale_gt_pcs)

                # PCN encoding
                xyz = pc_cano2pcn_torch(fix_num_canonical_cluster_pcs)
                B, N, _ = xyz.shape
                feature = pcn.first_conv(xyz.transpose(2, 1))
                # print('feature.shape',feature.shape)# (B,  256, N)
                feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1) # this line
                feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
                feature = pcn.second_conv(feature)  # (B, 1024, N)
                feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)

                # PCN decoding
                # complete_pcs = pcn(pc_cano2pcn_torch(fix_num_canonical_cluster_pcs))  # rotate them to fit pretrained PCN
                complete_pcs =  pcn.mlp(feature_global).reshape(-1, pcn.num_coarse, 3).contiguous()
                # bbox_scores = self.z_bbox_score(feature_global)



                complete_pcs = pc_pcn2cano(complete_pcs)  # rotate back to fit GT
                zs = onet.student_encoder(complete_pcs)
                pred_occ_logits = onet.decode(input_points_for_completion, z = None, features=zs)[0].logits  # .logits .probs

                # complete_pc = pcn(fix_num_canonical_cluster_pcs)
                vis_onet_results = False
                if vis_onet_results:
                    for i in range(zs.shape[0]):
                        from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, get_mesh_actor_vtk, \
                            vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                        in_points = input_points_for_completion[i][occ[i] >0]
                        out_points = input_points_for_completion[i][occ[i] <0]
                        vis_actors_vtk([
                            get_pc_actor_vtk(in_points.detach().cpu().numpy(), color=(1, 0, 0)),
                            get_pc_actor_vtk(out_points.detach().cpu().numpy()),

                        ])

        ''' Parepare loss inputs: pred and gt'''
        loss_inp = {}
        ### Phase1: Per point results
        if not cfg.use_gt_seg_and_bbox:
            loss_inp['semantic_scores'] = (semantic_scores, labels,  instance_semantic_CAD,instance_shapenet_catids)
            loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
            loss_inp['pt_bboxes'] = (pt_bboxes,pred_angle_param, instance_bboxes, instance_bbox_size ,instance_zs_valid)

        ### Phase1.5: merged clsuter results
        if epoch > cfg.prepare_epochs and not cfg.use_gt_seg_and_bbox:
            loss_inp['proposal_info'] = (proposals_idx, proposals_offset)  # ret['proposal_info']
            loss_inp['gt_info'] = gt_ious,gt_instance_idxs
            loss_inp['proposal_bbox'] = pred_cluster_bboxes ,pred_cluster_bboxes
            loss_inp['instance_pc'] = instance_pc

        ### Phase1.5: merged clsuter results
        if epoch > cfg.prepare_epochs_2:
            if not cfg.use_gt_seg_and_bbox:
                loss_inp['gt_info'] = gt_ious, gt_instance_idxs
                loss_inp['CAD_instance_num'] = valid_cluster_num
                if valid_cluster_num>0:
                    loss_inp['completion_data'] = (fix_num_canonical_cluster_pcs,rescale_gt_pcs,complete_pcs,
                                                   pred_occ_logits,gt_occ, zs,teacher_zs)
                    loss_inp['scores'] = (pred_cls_score,pred_seg_score,pred_bbox_score,pred_mesh_score)
                    loss_inp['selected_cluster_ids'] = selected_cluster_ids
            else:
                loss_inp['completion_data'] = (fix_num_canonical_cluster_pcs, rescale_gt_pcs, complete_pcs,
                                               pred_occ_logits, gt_occ, zs, teacher_zs)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            if not cfg.use_gt_seg_and_bbox:
                preds['semantic'] = semantic_scores
                preds['pt_offsets'] = pt_offsets
                if epoch > cfg.prepare_epochs:
                    # preds['score'] = scores
                    preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

      
        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}
        '''get all losses'''
        # ----------------Phase 1: point-wise loss: semantic, offset, bbox, angle_flip, bbox_score-------------------
        if not cfg.use_gt_seg_and_bbox:
            '''semantic loss'''
            semantic_scores, semantic_labels, instance_semantic_CAD,instance_shapenet_catids = loss_inp['semantic_scores']
            # semantic_scores: (N, nClass), float32, cuda
            # semantic_labels: (N), long, cuda
            semantic_loss = pt_semantic_loss(semantic_scores, semantic_labels, loss_out)

            '''offset loss'''
            pt_offsets,  coords_float, instance_info, instance_labels = loss_inp['pt_offsets']
            offset_norm_loss, offset_dir_loss = pt_offset_loss(pt_offsets,  coords_float, instance_info,loss_out)

            ''' bbox iou loss '''
            pt_bboxes,pred_angle_param, instance_bboxes, instance_bbox_size, instance_zs_valid = loss_inp['pt_bboxes']
            bbox_loss ,iou = pt_bbox_iou_loss(pt_bboxes, instance_bboxes, instance_info,instance_labels,loss_out)
            # print('iou',iou.mean())

            ''' bbox angle flip loss'''
            angle_parameter = cfg.angle_parameter
            angle_strict = cfg.angle_strict
            if angle_parameter!= 'bin':
                angle_flip_loss,angle_flip_accuracy = pt_angle_flip_loss(angle_parameter, angle_strict,instance_info,instance_semantic_CAD, instance_labels, instance_bboxes,pred_angle_param, loss_out)
            else:
                angle_label_loss, angle_residual_loss = pt_angle_flip_loss(angle_parameter, angle_strict,instance_info,instance_semantic_CAD, instance_labels, instance_bboxes,pred_angle_param, loss_out)

            '''bbox reg loss'''

            bbox_reg_loss = pt_bbox_reg_loss(cfg.use_bbox_reg_loss,
                                                 pt_bboxes, instance_bboxes, instance_info, instance_labels, loss_out)




        # ----------Phase 1.5: proposal wise loss:  merged bbox iou, merged bbox point distance ----------
        if epoch > cfg.prepare_epochs and (not cfg.use_gt_seg_and_bbox): # stage 2, proposal-wize
            proposals_idx, proposals_offset = loss_inp['proposal_info']  # ret['proposal_info']
            gt_ious, gt_instance_idxs = loss_inp['gt_info']
            # loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)
            pred_cluster_bboxes, pred_cluster_bboxes = loss_inp['proposal_bbox']



            '''merged bbox iou loss'''
            pred_cluster_bboxes_valid_mask = instance_zs_valid[gt_instance_idxs].bool()
            valid_pred_cluster_bboxes = pred_cluster_bboxes
            valid_gt_clsuter_bboxes = instance_bboxes[gt_instance_idxs]
            # valid_gt_scores = gt_scores

            cluster_bbox_loss, cluster_bbox_iou = giou_3d_loss(valid_pred_cluster_bboxes, valid_gt_clsuter_bboxes)

        # ----------------Phase 3: instance-wise loss: mesh, semantic, segmentation score, bbox score, mesh score-------------------
        if epoch > cfg.prepare_epochs_2: #
            if cfg.use_gt_seg_and_bbox:
                fix_num_canonical_cluster_pcs, rescale_gt_pcs, complete_pcs, \
                pred_occ_logits, gt_occ, pred_zs, teacher_zs = loss_inp['completion_data']
                '''completion CD loss: PCN produces complete point clouds from partial ones'''
                completion_loss = instance_completion_CD_loss(cfg.use_completion_CD_loss, complete_pcs, rescale_gt_pcs,
                                                              loss_out)
                '''binary cross entropy loss of occupancy values indicating whether a point is inside or outside a surface'''
                mesh_loss, mesh_accuracy = instance_mesh_reconstruction_loss(cfg.use_mesh_reconstruction_loss,
                                                                             pred_occ_logits, gt_occ, loss_out)
                '''z loss (mesh embedding compared with the teacher)'''
                z_loss = instance_z_loss(cfg.use_z_loss, pred_zs, teacher_zs, loss_out)


            else:
                CAD_instance_num = loss_inp['CAD_instance_num']
                if CAD_instance_num > 0:
                    selected_cluster_ids = loss_inp['selected_cluster_ids']
                    pred_cls_score, pred_seg_score, pred_bbox_score, pred_mesh_score = loss_inp['scores']
                    fix_num_canonical_cluster_pcs, rescale_gt_pcs, complete_pcs, \
                    pred_occ_logits, gt_occ, pred_zs, teacher_zs = loss_inp['completion_data']
                    gt_seg_score = get_segmented_scores(gt_ious)

                    '''seg score loss'''
                    instance_seg_score_loss = score_criterion(torch.sigmoid(pred_seg_score.view(-1)), gt_seg_score).mean()
                    loss_out['instance_score_loss'] = (instance_seg_score_loss, gt_ious.shape[0])


                    ''' cls score loss '''
                    gt_cls_label = instance_semantic_CAD[gt_instance_idxs][selected_cluster_ids]  # (N), long, cuda  # -100,0,1,2,3,4,5,6,7
                    instance_semantic_loss = semantic_criterion(pred_cls_score, gt_cls_label).mean()

                    loss_out['instance_semantic_loss'] = (instance_semantic_loss, gt_cls_label.shape[0])


                    '''bbox score loss'''
                    gt_bbox_score = cluster_bbox_iou[selected_cluster_ids]
                    instance_bbox_score_loss = score_criterion(torch.sigmoid(pred_bbox_score.view(-1)), gt_bbox_score.detach()).mean()
                    loss_out['instance_bbox_score_loss'] = (instance_bbox_score_loss, gt_bbox_score.shape[0])


                    '''completion CD loss: PCN produces complete point clouds from partial ones'''
                    completion_loss = instance_completion_CD_loss(cfg.use_completion_CD_loss, complete_pcs, rescale_gt_pcs, loss_out)
                    '''binary cross entropy loss of occupancy values indicating whether a point is inside or outside a surface'''
                    mesh_loss,mesh_accuracy = instance_mesh_reconstruction_loss(cfg.use_mesh_reconstruction_loss, pred_occ_logits, gt_occ, loss_out)
                    '''z loss (mesh embedding compared with the teacher)'''
                    z_loss = instance_z_loss(cfg.use_z_loss, pred_zs, teacher_zs, loss_out)


                    '''mesh score loss'''

                    instance_mesh_score_loss = score_criterion(torch.sigmoid(pred_mesh_score.view(-1)), mesh_accuracy.detach()).mean()

                    loss_out['instance_mesh_score_loss'] = (instance_mesh_score_loss, gt_cls_label.shape[0])
                else:
                    completion_loss = torch.tensor(0).cuda()
                    mesh_loss = torch.tensor(0).cuda()
                    z_loss = torch.tensor(0).cuda()
                    instance_seg_score_loss = torch.tensor(0).cuda()
                    instance_bbox_score_loss = torch.tensor(0).cuda()
                    instance_mesh_score_loss = torch.tensor(0).cuda()
                    instance_semantic_loss = torch.tensor(0).cuda()
                    loss_out['completion_loss'] = (completion_loss, 1)
                    loss_out['mesh_loss'] = (mesh_loss, 1)
                    loss_out['z_loss'] = (z_loss, 1)
                    loss_out['instance_score_loss'] = (z_loss, 1)
                    loss_out['instance_bbox_score_loss'] = (z_loss, 1)
                    loss_out['instance_mesh_score_loss'] = (z_loss, 1)
                    loss_out['instance_semantic_loss'] = (z_loss, 1)
                    loss_out['mesh_accuracy'] = (z_loss, 1)



        '''add all losses to the total loss'''
        # Phase 1
        if not cfg.use_gt_seg_and_bbox:
            # print('bbox_score_loss',bbox_score_loss)
            loss = semantic_loss + offset_norm_loss + offset_dir_loss + \
                   bbox_loss
            if angle_parameter != 'bin':
                loss += angle_flip_loss
            else:
                loss += angle_label_loss + angle_residual_loss
            if cfg.use_bbox_reg_loss:
                loss += bbox_reg_loss * cfg.bbox_reg_loss_weight
        else:
            loss = 0

        # Phase 2
        if epoch > cfg.prepare_epochs_2:
            if not cfg.use_gt_seg_and_bbox:
                loss += instance_seg_score_loss + instance_semantic_loss + instance_bbox_score_loss + instance_mesh_score_loss
            if cfg.use_completion_CD_loss:
                loss += completion_loss * cfg.CD_loss_weight
            if cfg.use_mesh_reconstruction_loss:
                # mesh_loss_weight = 1 # 0.0000001 # 0.0000001
                loss += mesh_loss * cfg.mesh_loss_weight
            # print('mesh_loss *mesh_loss_weight', mesh_loss * mesh_loss_weight)  # 500-600p
            if cfg.use_z_loss:
                # z_loss_weight = 0.0001 # proper for z loss
                # print('z_loss *z_loss_weight', z_loss*z_loss_weight)  # 500-600p
                loss += z_loss * cfg.z_loss_weight
            # # print(loss.item(), completion_loss.item(),mesh_loss.item())

        return loss, loss_out, infos


    ### gt_scores,
    def get_segmented_scores(scores, fg_thresh=0.75, bg_thresh=0.25):
        '''
        :param scores: (N), float, 0~1, the max IoU !!!
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh  # fg == positive, valid match
        bg_mask = scores < bg_thresh  # bg == negative, not a valid match
        interval_mask = (fg_mask == 0) & (bg_mask == 0)  # between fg & bg, maybe a valid match

        segmented_scores = (fg_mask > 0).float()  # hard threshold, score --> 1 for valid match

        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b  # soft threshold

        return segmented_scores

    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn


def select_good_clusters(gt_ious,gt_instance_idxs,instance_zs_valid):
    '''

    :param gt_ious: (nProposal) float, long
    :param gt_instance_idxs:  (nProposal)  long
    :param instance_zs_valid:  # (total_nInst), int,
    :return:
    '''
    from util.config import cfg
    nProposal = gt_ious.shape[0]
    object_limit_per_batch = cfg.object_limit_per_scene * cfg.batch_size


    porposals_w_gt_ids = torch.tensor([i for i in range(nProposal) if instance_zs_valid[gt_instance_idxs[i]] == 1]).long()
    # porposals_w_gt_ids = torch.tensor([i for i in range(nProposal) ]).long() # not use instance_zs_valid for debugging
    # gt_ious = gt_ious[porposals_w_gt_ids]
    # gt_instance_idxs = gt_instance_idxs[porposals_w_gt_ids]
    proposal_w_gt_mask = instance_zs_valid[gt_instance_idxs]
    gt_ious = (gt_ious * proposal_w_gt_mask).float() #


    # cluster_point_nums = proposals_offset[1:] - proposals_offset[:-1]
    # objectness_sort = torch.argsort(scores.squeeze(1), descending=True)  # index, not value
    gt_ious_sort = torch.argsort(gt_ious[porposals_w_gt_ids], descending=True)
    my_sort = gt_ious_sort
    # from score high to low, and different gt, get cluster ids. the number is how many gts detected
    gt_ids = np.unique(gt_instance_idxs[porposals_w_gt_ids][my_sort].cpu().numpy(), return_index=True)[1]  # np unique,0 is value, 1 is id
    # print('unique',gt_instance_idxs[porposals_w_gt_ids][my_sort][gt_ids])
    # print('unique ious',gt_ious[porposals_w_gt_ids][my_sort][gt_ids])
    # exclude clusters with too low ious
    # gt_ids = [i for i in gt_ids if(cluster_point_nums[my_sort][i] >= cfg.cfg.cluster_npoint_thre)]#
    gt_ids = [i for i in gt_ids if (gt_ious[porposals_w_gt_ids][my_sort][i] >= cfg.cluster_gt_iou_thre)]  #
    # if not enough, add others from highest scores
    gt_ids = np.hstack([gt_ids, np.setdiff1d(range(len(my_sort)), gt_ids, assume_unique=True)])[
             :object_limit_per_batch]
    selected_cluster_ids = porposals_w_gt_ids[my_sort][gt_ids].long()  # id of selected cluster
    # print(selected_cluster_ids)
    # debut selection
    if False:  # debug
        # print("gt-ious",gt_ious[gt_ious_sort])
        # get predicted cluster pc (partial scan)
        # proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, coords_float.shape[0]), dtype=torch.int,
        #                              device=scores.device)  # (nProposal, N), int, cuda
        # proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        # clusters = proposals_pred #[selected_cluster_ids]  # (nProp, N)
        #
        # ## vis differnet gt_iou
        # for i in range(len(gt_ious_sort)):
        #     print(i, gt_ious[gt_ious_sort][i])
        #     target_pc = coords_float\
        #     [clusters[gt_ious_sort][i].bool()].detach().cpu().numpy()
        #
        #     template_pc = instance_pc[gt_instance_idxs[gt_ious_sort][i]].detach().cpu().numpy()
        #     gt_pc = coords_float[(instance_labels==gt_instance_idxs[gt_ious_sort][i])].detach().cpu().numpy()
        #     print('target_pc.shape', target_pc.shape)  # back here
        #     print('gt_pc.shape',gt_pc.shape)
        #     # target_pc = pc_fit2standard(target_pc, bbox)
        #     from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
        #         vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
        #     vis_actors_vtk([get_pc_actor_vtk(target_pc, color=(1, 0, 0)),
        #                     # get_pc_actor_vtk(template_pc), # template
        #                     get_pc_actor_vtk(gt_pc),
        #                     ])

        ## check selection
        # print('selected_cluster_ids',selected_cluster_ids)
        # print('all scores',scores.squeeze(1)[objectness_sort])
        # print('all gts',gt_instance_idxs[objectness_sort])
        # # print('selected scores',scores.squeeze(1)[selected_cluster_ids])
        print('selected gts', gt_instance_idxs[selected_cluster_ids])
        # # print('selected p_num',cluster_point_nums[selected_cluster_ids])
        # # print('selected p_num',cluster_point_nums[selected_cluster_ids])
        print('all gt_iou', gt_ious[gt_ious_sort])
        print('selected gt_iou', gt_ious[selected_cluster_ids])
        #
        print('-' * 100)
    return selected_cluster_ids.to(instance_zs_valid.device)


def get_selected_cluster_pc(proposals_offset,proposals_idx,coords_float,
                            selected_cluster_ids,gt_instance_idxs,
                            proposals_semantics,instance_bboxes, pred_bbox,
                            instance_pc = None, use_gt_seg = False) :
    device = coords_float.device

    # object_limit_per_batch = cfg.object_limit_per_scene * cfg.batch_size
    selected_cluster_num = selected_cluster_ids.shape[0]

    # if not use_gt_seg:
    #     pred_zs, pred_angle, pred_center, pred_scale, pred_residual_center, pred_residual_scale, pred_residual_angle = \
    #         pred_bbox
    #     # print('pred_residual_scale.shape',pred_residual_scale.shape)  # 120,24
    #     n_clusters = pred_residual_center.shape[0]
    #     pred_residual_center = pred_residual_center.reshape(n_clusters, 8, 3)  # [nProp, 8, 3]
    #     pred_residual_scale = pred_residual_scale.reshape(n_clusters, 8, 3)  # [nProp, 8, 3]
    #     pred_residual_angle = pred_residual_angle.reshape(n_clusters,8)  # [nProp, 8]
    #     cluster_semantic_ids = proposals_semantics.max(1)[1]

    # cluster_semantic_ids = proposals_semantics.max(1)[1]
    # print('proposals_offset',proposals_offset) # ok
    # print('coords_float',coords_float) # ok
    # print('proposals_offset.shape',proposals_offset.shape)
    # print('coords_float.shape',coords_float.shape)
    # print('proposals_offset.shape[0]-1',proposals_offset.shape[0] - 1)
    # prepare masks and shuffle
    proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, coords_float.shape[0]), dtype=torch.int,
                                 device=coords_float.device)  # (nProposal, N), int, cuda # ok


    proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
    clusters = proposals_pred[
        selected_cluster_ids]  # (object_limit_per_batch, instance_point_num) clusters[i,j]: if point j is in cluster i
    shuffle_idx = np.arange(0, coords_float.shape[0]) # instance_point_num
    np.random.shuffle(shuffle_idx)

    sample_points_per_cluster = 2048

    # print(proposals_pred,'proposals_pred')
    # print(clusters,'clusters')
    # print(shuffle_idx,'shuffle_idx')

    # print(device)
    # print(selected_cluster_num)
    # print(sample_points_per_cluster)

    # temp = torch.zeros((selected_cluster_num,sample_points_per_cluster,3)) # this is ok

    fix_num_world_cluster_pcs = torch.zeros((selected_cluster_num,sample_points_per_cluster,3)).to(device) #

    fix_num_canonical_cluster_pcs = torch.zeros((selected_cluster_num,sample_points_per_cluster,3)).to(device)
    rescale_gt_pcs = torch.zeros((selected_cluster_num,sample_points_per_cluster,3)).to(device)
    cluster_bboxes = torch.zeros((selected_cluster_num, 7)).to(device)
    cluster_valid_mask = torch.ones(selected_cluster_num).to(device)


    ##### get bbox iou
    cids = range(selected_cluster_num)
    cid_in_all = selected_cluster_ids[cids]

    gt_bbox_cid = instance_bboxes[gt_instance_idxs[cid_in_all]]

    # print('iou', iou)

    for cid in range(selected_cluster_num):
        cid_in_all = selected_cluster_ids[cid]
        original_target_pc = coords_float[shuffle_idx][clusters[:, shuffle_idx][cid].bool()]
        fix_num_target_pc = original_target_pc[0:sample_points_per_cluster] # the original_target_pc may have less than this num
        # print('original_target_pc.shape',original_target_pc.shape)
        # print('fix_num_target_pc.shape',fix_num_target_pc.shape)

        # print('fix_num_world_cluster_pcs[cid]',fix_num_world_cluster_pcs[cid])

        # from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
        #     vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
        # vis_actors_vtk([
        #     get_pc_actor_vtk(original_target_pc.detach().cpu().numpy(), color=(1, 0, 0)),
        #
        # ])

        # transform to canonical
        ### get pred bboxes
        # semantic label
        if not use_gt_seg:
            pred_bbox_cid = pred_bbox[cid_in_all]
            _, iou = giou_3d_loss(pred_bbox_cid, gt_bbox_cid)
            fix_num_world_cluster_pcs[cid][:fix_num_target_pc.shape[0]] = fix_num_target_pc
            # cluster_bbox = torch.zeros((7)).cuda()
            # cluster_label = cluster_semantic_ids[cid_in_all]


            # # center
            # cluster_bbox[0:3] = pred_center[cid_in_all] +  pred_residual_center[cid_in_all, cluster_label]
            #
            # # scale (soft)
            # residual_scale = torch.max(pred_residual_scale[cid_in_all, cluster_label], - 0.1 * pred_scale[cid_in_all])  #
            # cluster_bbox[3:6] = pred_scale[cid_in_all] + residual_scale
            #
            # # if cid == 3:
            # #     print('center main:', pred_center[cid_in_all])
            # #     print('center res :', pred_residual_center[cid_in_all, cluster_label])
            # #     print('scale main:', pred_scale[cid_in_all])
            # #     print('scale res :', pred_residual_scale[cid_in_all, cluster_label])
            # #     print("----------------------------------------------")
            # # rotation (w/o residual)
            # cluster_bbox[6] = pred_angle[cid_in_all]  # +
            cluster_bbox = pred_bbox[cid_in_all]
            fix_num_target_pc_pred = pc_fit2standard_torch(fix_num_target_pc, cluster_bbox)
            fix_num_canonical_cluster_pcs[cid][:fix_num_target_pc_pred.shape[0]] = fix_num_target_pc_pred
            vis_now = True
            if vis_now: # check semantic
                # from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                #     vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                # for i in range(fix_num_canonical_cluster_pcs.shape[0]):
                #     print(i)
                #     # if i!= 3:
                #     #     continue
                #     vis_actors_vtk([
                #         get_pc_actor_vtk(fix_num_canonical_cluster_pcs[i].detach().cpu().numpy(), color=(1, 0, 0)),
                #         get_pc_actor_vtk(rescale_gt_pcs[i].detach().cpu().numpy(), color=(0, 0, 1)),
                #
                #     ])

                print('cluster_bbox',cluster_bbox)

                from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                    vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                vis_actors_vtk([
                    get_pc_actor_vtk(fix_num_target_pc_pred.detach().cpu().numpy(), color=(1, 0, 0)),
                    # get_pc_actor_vtk(fix_num_canonical_cluster_pcs[cid].detach().cpu().numpy(), color=(0, 0, 1)),

                ])
        else:
            cluster_bbox = instance_bboxes[gt_instance_idxs[cid_in_all]]
            fix_num_target_pc_pred = pc_fit2standard_torch(fix_num_target_pc, cluster_bbox)
            # filter out instances with WDH too bigger than gt (after normalization, biggest WDH for gt will be 1)
            # fix_num_target_pc_pred_scale_xyz = torch.max(fix_num_target_pc_pred, 0)[0] - \
            #                                    torch.min(fix_num_target_pc_pred, 0)[0]
            # fix_num_target_pc_pred_scale_max = torch.max(fix_num_target_pc_pred_scale_xyz)

            fix_num_target_pc_pred_scale_max = torch.max(torch.abs(fix_num_target_pc_pred))
            # print('fix_num_target_pc_pred_scale_xyz', fix_num_target_pc_pred_scale_xyz)
            # print('fix_num_target_pc_pred_scale_max',fix_num_target_pc_pred_scale_max)
            if fix_num_target_pc_pred_scale_max >0.8:
                cluster_valid_mask[cid] = 0

            fix_num_canonical_cluster_pcs[cid][:fix_num_target_pc_pred.shape[0]] = fix_num_target_pc_pred
            fix_num_world_cluster_pcs[cid][:fix_num_target_pc.shape[0]] = fix_num_target_pc




        if instance_bboxes is not None:
            gt_bbox = instance_bboxes[gt_instance_idxs[cid_in_all]]
            fix_num_target_pc_gt = pc_fit2standard_torch(fix_num_target_pc, gt_bbox)
        if instance_pc is not None:
            # print('cid_in_all',cid_in_all) # 456
            # print('gt_instance_idxs.shape',gt_instance_idxs.shape)
            # print('gt_instance_idxs',gt_instance_idxs[cid_in_all])
            # print('instance_pc[',instance_pc.shape)
            complete_template_pc = instance_pc[gt_instance_idxs[cid_in_all]]  # canonical but needs to be scaled
            complete_template_pc_scale_xyz = torch.max(complete_template_pc, 0)[0] - torch.min(complete_template_pc, 0)[0]
            complete_template_pc = complete_template_pc / complete_template_pc_scale_xyz * cluster_bbox[[4, 5, 3]] / torch.max(
                cluster_bbox[[4, 5, 3]])



            rescale_gt_pcs[cid] = complete_template_pc
        # print(complete_template_pc_scale_xyz)
        # print(complete_template_pc)
        # template_pc_to_world =  pc_fit_scan(complete_template_pc.detach().cpu().numpy(),gt_bbox.detach().cpu().numpy())

        # print('all shape',complete_template_pc.shape)
        # print('fix shapef',fix_num_canonical_cluster_pcs.shape)
        # print('instance num',original_target_pc.shape)
        # if original_target_pc.shape[0] < sample_points_per_cluster:
        #     print(fix_num_target_pc_pred)


        if True: #cid==0:
            from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
            # print(cluster_bbox)
            vis_actors_vtk([
                # get_pc_actor_vtk(coords_float.detach().cpu().numpy()),
                # get_pc_actor_vtk(original_target_pc.detach().cpu().numpy(),color = (1,0,0)), # world

                # get_pc_actor_vtk(template_pc_to_world,color = (1,0,0)),
                # get_pc_actor_vtk(fix_num_target_pc_pred.detach().cpu().numpy(), color=(1, 0, 0)),

                get_pc_actor_vtk(fix_num_target_pc_gt.detach().cpu().numpy(), color=(1, 0, 0)),
                get_pc_actor_vtk(complete_template_pc.detach().cpu().numpy(), color=(0, 0, 1)),

                # get_pc_actor_vtk(fix_num_world_cluster_pcs[cid].detach().cpu().numpy(), color=(0, 1, 0)),
            ])

        cluster_bboxes[cid] = cluster_bbox

    fix_num_canonical_cluster_pcs = fix_num_canonical_cluster_pcs[cluster_valid_mask.bool()]
    rescale_gt_pcs = rescale_gt_pcs[cluster_valid_mask.bool()]
    cluster_bboxes = cluster_bboxes[cluster_valid_mask.bool()]
    fix_num_world_cluster_pcs = fix_num_world_cluster_pcs[cluster_valid_mask.bool()]

    # from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
    #     vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
    # for i in range(fix_num_canonical_cluster_pcs.shape[0]):
    #     print(i)
    #     # if i!= 3:
    #     #     continue
    #     vis_actors_vtk([
    #         get_pc_actor_vtk(fix_num_canonical_cluster_pcs[i].detach().cpu().numpy(), color=(1, 0, 0)),
    #         get_pc_actor_vtk(rescale_gt_pcs[i].detach().cpu().numpy(), color=(0, 0, 1)),
    #
    #     ])
    return_dict = {'fix_num_canonical_cluster_pcs':fix_num_canonical_cluster_pcs,
                   'rescale_gt_pcs':rescale_gt_pcs,
                   'cluster_bboxes':cluster_bboxes,
                   'fix_num_world_cluster_pcs':fix_num_world_cluster_pcs,
                   'cluster_valid_mask' : cluster_valid_mask,

    }
    # return fix_num_canonical_cluster_pcs, rescale_gt_pcs ,cluster_bboxes ,fix_num_world_cluster_pcs
    return return_dict