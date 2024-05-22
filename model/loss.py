import torch
import torch.nn as nn
import torch.nn.functional as F
from util.config import cfg
# from lib.iou3d_nms import iou3d_nms_utils
from lib.rotated_iou import cal_giou_3d, cal_iou_3d
from util.consts import RFS2CAD_arr, CAD_weights, not_synthetic_mask,synthetic_mask, CAD_labels
from model.pcn import cd_loss_L1

#### criterion
semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label,reduction='none').cuda()
score_criterion = nn.BCELoss(reduction='none').cuda()
label_criterion = nn.CrossEntropyLoss(reduction='none').cuda()

#### some basic losses
def huber_loss(error, delta=1.0):
    """
    Also known as smooth l1 loss
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss
def giou_3d_loss(pred, target):
    '''
    cal_giou_3d() is in lib/rotated_iou/oriented_iou_loss.py
    It accepts (B,N,7) inputs.
    here we accept (N,7) instead.


    :param pred: [N,7] N bounding boxes or parameters (x,y,z,w,l,h, theta). Assume theta is the angle around z-axis
    :param target:  [N,7] N bounding boxes or parameters (x,y,z,w,l,h, theta). Assume theta is the angle around z-axis
    :return: the bbox between them
    '''
    results = cal_giou_3d(pred[None, ...], target[None, ...])
    iou_loss = results[0][0]
    iou = results[1][0]
    return iou_loss, iou



#### phase 1: point-wise losses
def pt_semantic_loss(semantic_scores, semantic_labels, loss_out):
    semantic_loss = semantic_criterion(semantic_scores, semantic_labels).mean()
    loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])
    return semantic_loss


def pt_offset_loss(pt_offsets, coords_float, instance_info, loss_out):
    gt_offsets = instance_info[:, 0:3] - coords_float  # (N, 3)
    pt_diff = pt_offsets - gt_offsets  # (N, 3)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
    # pt_valid = (instance_labels != cfg.ignore_label).float()
    pt_valid = (instance_info[:, 0] != -100).float()
    offset_norm_loss = torch.sum(pt_dist * pt_valid) / (torch.sum(pt_valid) + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
    pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
    offset_dir_loss = torch.sum(direction_diff * pt_valid) / (torch.sum(pt_valid) + 1e-6)

    loss_out['offset_norm_loss'] = (offset_norm_loss, pt_valid.sum())
    loss_out['offset_dir_loss'] = (offset_dir_loss, pt_valid.sum())
    return offset_norm_loss, offset_dir_loss


def pt_bbox_iou_loss(pt_bboxes, instance_bboxes, instance_info, instance_labels, loss_out):
    pred_7_dim_bbox = pt_bboxes
    # not pre load gt
    pt_valid = (instance_info[:, 9] != -100).bool()  # only supervise CAD-instances, class balanced.
    pred_7_dim_bbox_valid = pred_7_dim_bbox[pt_valid]
    if pred_7_dim_bbox_valid.shape[0] > 0:
        gt_7_dim_bboxes_valid = instance_bboxes[instance_labels[pt_valid]]  # this line seems wrong: cuda error here

        iou_loss, iou = giou_3d_loss(pred_7_dim_bbox_valid, gt_7_dim_bboxes_valid)
        bbox_loss = (iou_loss).mean()
        # print('pred_7_dim_bbox_valid',pred_7_dim_bbox_valid)
        # print('gt_7_dim_bboxes_valid',gt_7_dim_bboxes_valid)
        # print('gt_pred_bbox_differen',gt_7_dim_bboxes_valid - pred_7_dim_bbox_valid)
        # print('---------------------------------------')
        if torch.abs(bbox_loss) > 100:
            bbox_loss = torch.tensor(0).float().cuda()

        loss_out['bbox_loss'] = (bbox_loss, pt_valid.sum())
        loss_out['bbox_iou'] = (iou.mean(), pt_valid.sum())
    else:
        bbox_loss = torch.tensor(0).float().cuda()
        iou = torch.tensor(0).float().cuda()

        loss_out['bbox_loss'] = (bbox_loss, 1)
        loss_out['bbox_iou'] = (iou, 1)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("the whole scene has no CAD instances!")
        # from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
        #     vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
        # vis_actors_vtk([
        #     get_pc_actor_vtk(coords_float[instance_labels != -100].detach().cpu().numpy(), color=(0, 0, 1)),
        #
        # ])


    return bbox_loss, iou

def pt_bbox_reg_loss(use_bbox_reg_loss, pt_bboxes, instance_bboxes, instance_info, instance_labels, loss_out):
    if use_bbox_reg_loss:
        pred_7_dim_bbox = pt_bboxes
        # not pre load gt
        pt_valid = (instance_info[:, 9] != -100).bool()  # only supervise CAD-instances, class balanced.
        pred_7_dim_bbox_valid = pred_7_dim_bbox[pt_valid]
        if pred_7_dim_bbox_valid.shape[0] > 0:
            gt_7_dim_bboxes_valid = instance_bboxes[instance_labels[pt_valid]]  # this line seems wrong: cuda error here

            bbox_reg_loss = huber_loss(pred_7_dim_bbox_valid[:,0:6] - gt_7_dim_bboxes_valid[:, 0:6]).mean()

            loss_out['bbox_reg_loss'] = (bbox_reg_loss, pt_valid.sum())

        else:
            bbox_reg_loss = torch.tensor(0).float().cuda()
            loss_out['bbox_reg_loss'] = (bbox_reg_loss, 1)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("the whole scene has no CAD instances!")
    else:
        bbox_reg_loss = torch.tensor(0).float().cuda()
        loss_out['bbox_reg_loss'] = (bbox_reg_loss, 1)

    return bbox_reg_loss
def pt_angle_flip_loss(angle_parameter,angle_strict, instance_info, instance_semantic_CAD,
                       instance_labels, instance_bboxes, pred_angle_param, loss_out):
    # for points with CAD gts and unsynthetic
    point_CAD_mask = (instance_info[:,
                      9] != -100).bool()  # points that has CAD GT. only supervise CAD-instances, class balanced.
    semantic_labels_CAD_valid = instance_semantic_CAD[instance_labels[point_CAD_mask]].long()

   
    if angle_parameter == 'Mobius_flip' or angle_parameter == 'Mobius_cos':
        if not angle_strict:
            angle_flip_mask = torch.tensor(not_synthetic_mask).cuda()[
                semantic_labels_CAD_valid].bool()  # mask: only compute angle flip loss for not synthetic objects
        else:
            angle_flip_mask = torch.ones_like(semantic_labels_CAD_valid).cuda().bool() # modify all objects about flipping
        pt_angle_flip_parameter = pred_angle_param[:, 2]
   
        pred_angle_flip_valid = pt_angle_flip_parameter[point_CAD_mask][angle_flip_mask]
        if pred_angle_flip_valid.shape[0] > 0:
            check_points_in_instance = False
            if check_points_in_instance:
                valid_point_ids = torch.arange(instance_labels.shape[0])[point_CAD_mask][
                    angle_flip_mask].long().cuda()
                valid_point_instance_ids = instance_labels[valid_point_ids]
                CAD_valid = instance_zs_valid[valid_point_instance_ids]
               
            if angle_parameter == 'Mobius_flip':
                pred_angle_flip_valid = torch.sigmoid(pred_angle_flip_valid.view(-1))
                gt_angle_flip_valid = (
                        torch.abs(instance_bboxes[instance_labels[point_CAD_mask][angle_flip_mask]][:, 6]) > (
                        3.14159 / 2)).float()  # angle abs > 90 degree, need flipping
                angle_flip_criterion = nn.BCELoss(reduction='none').cuda()
                angle_flip_loss = angle_flip_criterion(pred_angle_flip_valid,
                                                       gt_angle_flip_valid).mean()  # this line error

            elif angle_parameter == 'Mobius_cos':
           
                pred_angle_flip_valid = torch.cos(pred_angle_flip_valid.view(-1))
                gt_angle_flip_valid = (torch.cos(
                    instance_bboxes[instance_labels[point_CAD_mask][angle_flip_mask]][:,
                    6])).float()  # angle abs > 90 degree, need flipping
                angle_flip_loss = huber_loss(pred_angle_flip_valid - gt_angle_flip_valid).mean()  # this line error

            check_points = False
            if check_points:
                for i in range(gt_angle_flip_valid.shape[0]):
                    global_point_id = valid_point_ids[i]
                    instance_id = instance_labels[global_point_id]
                    print('instance id', instance_id.item(),
                          instance_labels[point_CAD_mask][angle_flip_mask][i].item())  # 9
                    print('instance_semantic_CAD', instance_semantic_CAD[instance_id].item(),
                          semantic_labels_CAD_valid[angle_flip_mask][i].item())
                    print('instance_shapenet_catids', instance_shapenet_catids[instance_id])
                    print('instance_bboxes', instance_bboxes[instance_id][6].item(),
                          instance_bboxes[instance_labels[point_CAD_mask][angle_flip_mask]][i][6].item())
                    print('not_synthe',
                          torch.tensor(not_synthetic_mask).cuda()[instance_semantic_CAD[instance_id]].item())
                    print('gt flip',
                          gt_angle_flip_valid[i].item(), gt_angle_flip_valid2[i].item())
                    print('pred flip', pred_angle_flip_valid[i].item())
                    print('--------------------------------------------')
                    from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, \
                        vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                    vis_actors_vtk([
                        get_pc_actor_vtk(coords_float[instance_labels == instance_id].detach().cpu().numpy(),
                                         color=(0, 0, 1)),

                    ])
            angle_flip_right = (
                (torch.round(pred_angle_flip_valid).long() == gt_angle_flip_valid.long()).float()).sum()
            angle_flip_whole = gt_angle_flip_valid.shape[0]
            angle_flip_accuracy = angle_flip_right / angle_flip_whole
            loss_out['angle_flip_loss'] = (angle_flip_loss, (angle_flip_mask).sum())
            loss_out['angle_flip_accuracy'] = (angle_flip_accuracy, (angle_flip_mask).sum())
        else:
            angle_flip_loss = torch.tensor(0).float().cuda()
            angle_flip_accuracy = torch.tensor(0).float().cuda()
            loss_out['angle_flip_loss'] = (angle_flip_loss, 1)
            loss_out['angle_flip_accuracy'] = (angle_flip_accuracy, 1)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("the whole scene has no un-synthetic CAD instances!")
          
    elif angle_parameter == 'sin_cos' :
        sin = pred_angle_param[:, 0]
        cos = pred_angle_param[:, 1]
        norm = torch.pow(torch.pow(sin, 2) + torch.pow(cos, 2), 0.5)
        sin = sin / norm
        cos = cos / norm

        pred_pt_angle_sin = sin[point_CAD_mask]
        pred_pt_angle_cos = cos[point_CAD_mask]

        gt_pt_angle_sin = torch.sin(instance_bboxes[:, 6][instance_labels[point_CAD_mask]])
        gt_pt_angle_cos = torch.cos(instance_bboxes[:, 6][instance_labels[point_CAD_mask]])

        pred_pt_angle = torch.atan2(pred_pt_angle_sin, pred_pt_angle_cos)
        gt_pt_angle = instance_bboxes[:, 6][instance_labels[point_CAD_mask]]
      

        if not angle_strict:
            absolute_value_mask = torch.tensor(synthetic_mask).cuda()[
                semantic_labels_CAD_valid].bool()  # mask: synthetic objects, sin and cos do not require correct signs

            pred_pt_angle_sin[absolute_value_mask] = torch.abs(pred_pt_angle_sin[absolute_value_mask])
            pred_pt_angle_cos[absolute_value_mask] = torch.abs(pred_pt_angle_cos[absolute_value_mask])

            gt_pt_angle_sin[absolute_value_mask] = torch.abs(gt_pt_angle_sin[absolute_value_mask])
            gt_pt_angle_cos[absolute_value_mask] = torch.abs(gt_pt_angle_cos[absolute_value_mask])

        print_check = False
        if print_check:

          
            gt_pt_angle = instance_bboxes[:, 6][instance_labels[point_CAD_mask]] * 360 / (2 * 3.14159)
            pred_pt_angle = pred_7_dim_bbox_valid[:, 6] * 360 / (2 * 3.14159)

         

        angle_flip_loss = huber_loss(pred_pt_angle_sin - gt_pt_angle_sin).mean() \
                          + huber_loss(pred_pt_angle_cos - gt_pt_angle_cos).mean()


        angle_flip_accuracy = torch.tensor(0).cuda().float()
       
        loss_out['angle_flip_loss'] = (angle_flip_loss, (point_CAD_mask).sum())
        loss_out['angle_flip_accuracy'] = (angle_flip_accuracy, (point_CAD_mask).sum())
    elif angle_parameter == 'naive':
        pred_pt_angle = pred_angle_param[:, 0][point_CAD_mask]
        gt_pt_angle = instance_bboxes[:, 6][instance_labels[point_CAD_mask]]
        pred_pt_angle_sin = torch.sin(pred_pt_angle)
        pred_pt_angle_cos = torch.cos(pred_pt_angle)
        gt_pt_angle_sin = torch.sin(gt_pt_angle)
        gt_pt_angle_cos = torch.cos(gt_pt_angle)
        if not angle_strict:
            absolute_value_mask = torch.tensor(synthetic_mask).cuda()[
                semantic_labels_CAD_valid].bool()  # mask: synthetic objects, sin and cos do not require correct signs

            pred_pt_angle_sin[absolute_value_mask] = torch.abs(pred_pt_angle_sin[absolute_value_mask])
            pred_pt_angle_cos[absolute_value_mask] = torch.abs(pred_pt_angle_cos[absolute_value_mask])

            gt_pt_angle_sin[absolute_value_mask] = torch.abs(gt_pt_angle_sin[absolute_value_mask])
            gt_pt_angle_cos[absolute_value_mask] = torch.abs(gt_pt_angle_cos[absolute_value_mask])

        angle_flip_loss = huber_loss(pred_pt_angle_sin - gt_pt_angle_sin).mean() \
                          + huber_loss(pred_pt_angle_cos - gt_pt_angle_cos).mean()
        angle_flip_accuracy = torch.tensor(0).cuda().float()
        
        loss_out['angle_flip_loss'] = (angle_flip_loss, (point_CAD_mask).sum())
        loss_out['angle_flip_accuracy'] = (angle_flip_accuracy, (point_CAD_mask).sum())


    elif angle_parameter == 'bin':
        pt_valid = (instance_info[:, 9] != -100).float()  # only supervise CAD-instances, class balanced.
        gt_angle_label = instance_info[:, 9].long()
        gt_angle_label[gt_angle_label == -100] = 0  # invalid angles, will be masked out later
        gt_angle_residual = instance_info[:, 10]

        pt_angles = pred_angle_param
        angle_label = pt_angles[:, :12]
        angle_residual = pt_angles[:, 12:]

        angle_label_loss = label_criterion(angle_label, gt_angle_label)
        angle_label_loss = (angle_label_loss * pt_valid).sum() / (pt_valid.sum() + 1e-6)

        gt_angle_label_onehot = F.one_hot(gt_angle_label, 12).float()
        angle_residual = (angle_residual * gt_angle_label_onehot).sum(1)  # [nProp, 12] --> [nProp, ]
        gt_angle_residual = gt_angle_residual / (3.14159 / 12.0)  # normalized residual
        angle_residual_loss = huber_loss(angle_residual - gt_angle_residual)
        angle_residual_loss = (angle_residual_loss * pt_valid).sum() / (pt_valid.sum() + 1e-6)

        loss_out['angle_label_loss'] = (angle_label_loss, pt_valid.sum())
        loss_out['angle_residual_loss'] = (angle_residual_loss, pt_valid.sum())


    if angle_parameter != 'bin':
        return angle_flip_loss, angle_flip_accuracy
    else:
        return angle_label_loss,angle_residual_loss


def pt_bbox_score_loss(pt_bbox_scores, pt_valid, iou, loss_out):
    if pt_valid.sum() > 0:
    
        pt_bbox_scores_valid = pt_bbox_scores[pt_valid]
        pt_bbox_scores_valid = torch.sigmoid(pt_bbox_scores_valid.view(-1))
        pt_gt_bbox_scores_valid = iou.detach()

    
        bbox_score_loss = score_criterion(pt_bbox_scores_valid, pt_gt_bbox_scores_valid).mean()  # error here,
        loss_out['bbox_score_loss'] = (bbox_score_loss, pt_valid.sum())
    else:
        bbox_score_loss = torch.tensor(0).float().cuda()
        loss_out['bbox_score_loss'] = (bbox_score_loss, 1)
    return bbox_score_loss


#### phase 2: instance-wise losses

def instance_completion_CD_loss(use_completion_CD_loss,complete_pcs,rescale_gt_pcs,loss_out):

    if use_completion_CD_loss:
        completion_loss = cd_loss_L1(complete_pcs, rescale_gt_pcs)
       
    else:
        completion_loss = torch.tensor(0).cuda()
    loss_out['completion_loss'] = (completion_loss, 1)
    return completion_loss

def instance_mesh_reconstruction_loss(use_mesh_reconstruction_loss,pred_occ_logits,gt_occ, loss_out):

    loss_i = F.binary_cross_entropy_with_logits(
        pred_occ_logits, gt_occ, reduction='none')
    mesh_loss = loss_i.sum(-1).mean()
    with torch.no_grad():
    

        occ_right = (
            (torch.round(torch.sigmoid(pred_occ_logits)).long() == gt_occ.long()).float()).sum(1)
        occ_whole =  gt_occ.shape[1]
        mesh_accuracy = occ_right / occ_whole

    if not use_mesh_reconstruction_loss:
        mesh_loss = torch.tensor(0.0).cuda()

    loss_out['mesh_loss'] = (mesh_loss, 1)
    loss_out['mesh_accuracy'] = (mesh_accuracy.mean(), 1)
    return mesh_loss,mesh_accuracy

def instance_z_loss(use_z_loss,pred_zs,teacher_zs,loss_out):
    if use_z_loss:
        z_loss = torch.mean(huber_loss(pred_zs - teacher_zs))  # [nProposal, 264] --> [nProposal]
    else:
        z_loss = torch.tensor(0).cuda()
    loss_out['z_loss'] = (z_loss, 1)
    return z_loss