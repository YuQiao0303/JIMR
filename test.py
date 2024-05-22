import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import open3d as o3d

import torch
import torch.nn.functional as F
import time
import numpy as np
import random
import os

from util.config import cfg
cfg.task = 'test'

from util.log import logger
import util.utils as utils

from util.consts import *
# from model.bsp import PolyMesh
from util.utils import export_pc_xyz

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, cfg.result_path, 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'instance_seg'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))
    os.system('cp {} {}'.format(cfg.model_pipeline_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.model_loss_dir, backup_dir))

    global time_dump_dir
    time_dump_dir = os.path.join(result_dir, 'inference_time.csv')

    global semantic_label_idx
    
    semantic_label_idx = np.arange(25) # use pico ids

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch):

    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst import Dataset
            dataset = Dataset(test=True)
            dataset.testLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    dataloader = dataset.test_data_loader
    final_score_type = cfg.final_score_type

    with torch.no_grad():
        model = model.eval()
        start = time.time()
        print('totaly number of parameters:', sum(p.numel() for p in model.parameters() ))

        for i, batch in enumerate(dataloader):
            # print(i)

            # test batch is 1
            N = batch['feats'].shape[0]

            test_scene_name = dataset.test_files[int(batch['id'][0])]

            start1 = time.time()
            with torch.no_grad():
                # print('pred')
                preds = model_fn(batch, model, epoch)
                # print('done pred')
            end1 = time.time() - start1
            print('inference_time', end1)
            with open(time_dump_dir, "a", encoding="utf-8") as f:
                f.write(str(end1))
                f.write("\n")


            if (not cfg.use_gt_seg_and_bbox) and (not preds['has_instance']):
                # print('no instance in the scene')
                # print("preds['has_instance']",preds['has_instance'])
                continue
            ##### get predictions
            if not cfg.use_gt_seg_and_bbox :
                semantic_pred = preds['semantic_pred'] # CAD
                pt_offsets = preds['pt_offsets']
                # pt_angles = preds['pt_angles']

            if epoch > cfg.prepare_epochs:
                if not cfg.use_gt_seg_and_bbox:
                    gt_seg_ious = preds['gt_seg_ious']
                    clusters = preds['clusters']
                    cluster_seg_scores,cluster_bbox_scores,cluster_sem_scores,cluster_mesh_scores = preds['cluster_scores'] # preds['cluster_scores'] =
                    cluster_semantic_score = cluster_sem_scores
                    cluster_final_scores = preds['cluster_final_scores']
                    cluster_scores = cluster_final_scores

                    # i = 0
                    # print(cluster_seg_scores[i].item(),cluster_sem_scores[i].item(),cluster_bbox_scores[i].item(),cluster_mesh_scores[i].item())
                    # print(cfg.final_score_type)
                    # print((cluster_seg_scores[i]*cluster_sem_scores[i]).item() )
                    # print((cluster_scores[i]).item() )
                    #
                    # print('----------------------')


                    # cluster_semantic_score = preds['cluster_semantic_score']


                if  epoch > cfg.prepare_epochs_3:
                    cluster_meshes = preds['cluster_meshes']
                # print('cluster_semantic_id', cluster_semantic_id)

                cluster_semantic_id = preds['cluster_semantic_id']
                pred_corresponding_gt_mesh_id =preds['pred_corresponding_gt_mesh_id']
                if cfg.retrieval:
                    cluster_alignment = preds['cluster_alignment'] # dataframe

                # nclusters = len(cluster_meshes)

                nclusters = len(cluster_semantic_id)

            ##### save files
            start3 = time.time()

            # Phase 1
            if not cfg.use_gt_seg_and_bbox:
                # save semantics.
                if cfg.save_semantic:
                    os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                    semantic_np = semantic_pred.cpu().numpy()
                    np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

                # save offsets
                if cfg.save_pt_offsets:
                    os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                    pt_offsets_np = pt_offsets.cpu().numpy()
                    coords_np = batch['locs_float'].numpy()
                    coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                    np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)

                # # save angles
                # if cfg.save_pt_angles:
                #     os.makedirs(os.path.join(result_dir, 'angles'), exist_ok=True)
                #     pt_angles_np = pt_angles.cpu().numpy() # (N, )
                #     np.save(os.path.join(result_dir, 'angles', test_scene_name + '.npy'), pt_angles_np)

            # Phase 1.5
            if epoch > cfg.prepare_epochs and not cfg.use_gt_seg_and_bbox :
                # save instances.
                if cfg.save_instance:
                    f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                    for proposal_id in range(nclusters):
                        clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                        #semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                        semantic_label = cluster_semantic_id[proposal_id].item()
                        score = cluster_scores[proposal_id]
                        f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                        if proposal_id < nclusters - 1:
                            f.write('\n')
                        np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                    f.close()

                if cfg.save_instance_for_seg_eval:
                    for proposal_id in range(nclusters):
                        clusters_i = clusters[proposal_id].bool().cpu().numpy()  # (N)
                        # clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                        N = len(clusters_i)
                        cluster_point_idxs = np.arange(N)[clusters_i]
                        # print('cluster_point_idxs',cluster_point_idxs)

                        # semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                        semantic_label = cluster_semantic_id[proposal_id].item()
                        # score = (cluster_semantic_score[proposal_id] * cluster_scores[proposal_id]).item()
                        score = cluster_final_scores[proposal_id] .item()

                        np.savez(os.path.join(result_dir,'instance_seg', '{}_{:d}.npz'.format(test_scene_name , proposal_id )),
                                 cluster_point_idxs=cluster_point_idxs, semantic_label=semantic_label,score = score)
                    # print('instance segmentation for evaluation saved')

            if epoch > cfg.prepare_epochs_3 and cfg.save_mesh:
                # save meshes.
                os.makedirs(os.path.join(result_dir, 'meshes'), exist_ok=True)
                os.makedirs(os.path.join(result_dir, 'trimeshes'), exist_ok=True)

                scores = cluster_final_scores
                for proposal_id in range(nclusters):
                    mesh = cluster_meshes[proposal_id]
                    # not valid CAD label, skip.
                    if mesh is None: 
                        continue
                    # clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    
                    semantic_label = cluster_semantic_id[proposal_id].item()
                    #semantic_label_ = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))

                    print(f'[Mesh] save {proposal_id} / {nclusters}, label = {CAD_labels[semantic_label]}')

                    if not cfg.use_gt_seg_and_bbox:
                        #  seg, seg_cls , seg_cls_bbox, seg_cls_bbox_mesh, seg_cls_mesh
                        score = scores[proposal_id]

                    else:
                        score = 1

                    mesh.export(os.path.join(result_dir, "meshes", f"{test_scene_name}_{proposal_id}_gt_{pred_corresponding_gt_mesh_id[proposal_id]}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))
                    if isinstance(mesh, PolyMesh):
                        tmesh = mesh.to_trimesh()
                        tmesh.export(os.path.join(result_dir, "trimeshes", f"{test_scene_name}_{proposal_id}_gt_{pred_corresponding_gt_mesh_id[proposal_id]}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))

            if epoch > cfg.prepare_epochs_3 and cfg.save_cano_mesh:
                # save meshes.
                os.makedirs(os.path.join(result_dir, 'cano_meshes'), exist_ok=True)
                scores = cluster_final_scores
                cluster_meshes = preds['cluster_cano_meshes']
                for proposal_id in range(nclusters):
                    mesh = cluster_meshes[proposal_id]
                    # not valid CAD label, skip.
                    if mesh is None:
                        continue
                    # clusters_i = clusters[proposal_id].cpu().numpy()  # (N)

                    semantic_label = cluster_semantic_id[proposal_id].item()
                    # semantic_label_ = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))

                    print(f'[Mesh] save {proposal_id} / {nclusters}, label = {CAD_labels[semantic_label]}')

                    if not cfg.use_gt_seg_and_bbox:
                        #  seg, seg_cls , seg_cls_bbox, seg_cls_bbox_mesh, seg_cls_mesh
                        score = scores[proposal_id]

                    else:
                        score = 1
                    mesh.export(os.path.join(result_dir, "cano_meshes",
                                             f"{test_scene_name}_{proposal_id}_gt_{pred_corresponding_gt_mesh_id[proposal_id]}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))

            if epoch > cfg.prepare_epochs_3 and cfg.save_pc:
                cluster_completed_pcs =preds['cluster_completed_pc']
                cluster_partial_pcs =preds['cluster_partial_pc']

                os.makedirs(os.path.join(result_dir, 'completed_pc'), exist_ok=True)
                os.makedirs(os.path.join(result_dir, 'partial_pc'), exist_ok=True)
                for proposal_id in range(nclusters):
                    completed_pc = cluster_completed_pcs[proposal_id]
                    partial_pc = cluster_partial_pcs[proposal_id]

                    # clusters_i = clusters[proposal_id].cpu().numpy()  # (N)

                    semantic_label = cluster_semantic_id[proposal_id].item()
                    # semantic_label_ = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))

                    print(f'[Point Couds] save {proposal_id} / {nclusters}, label = {CAD_labels[semantic_label]}')

                    if not cfg.use_gt_seg_and_bbox:
                        score = cluster_scores[proposal_id]
                    else:
                        score = 1
                    export_pc_xyz(completed_pc,os.path.join(result_dir, "completed_pc",
                                             f"{test_scene_name}_{proposal_id}_gt_{pred_corresponding_gt_mesh_id[proposal_id]}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))

                    export_pc_xyz(partial_pc, os.path.join(result_dir, "partial_pc",
                                                             f"{test_scene_name}_{proposal_id}_gt_{pred_corresponding_gt_mesh_id[proposal_id]}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))

            if epoch > cfg.prepare_epochs_3 and hasattr(cfg, 'save_cano_pc'):
                if cfg.save_cano_pc:
                    cluster_cano_completed_pc = preds['cluster_cano_completed_pc']
                    cluster_cano_partial_pc = preds['cluster_cano_partial_pc']
                    # print('type(cluster_cano_completed_pc)',type(cluster_cano_completed_pc)) # tensor
                    # print('type(cluster_cano_partial_pc)',type(cluster_cano_partial_pc))

                    os.makedirs(os.path.join(result_dir, 'cano_completed_pc'), exist_ok=True)
                    os.makedirs(os.path.join(result_dir, 'cano_partial_pc'), exist_ok=True)
                    for proposal_id in range(nclusters):
                        completed_pc = cluster_cano_completed_pc[proposal_id]
                        partial_pc = cluster_cano_partial_pc[proposal_id]

                        # clusters_i = clusters[proposal_id].cpu().numpy()  # (N)

                        semantic_label = cluster_semantic_id[proposal_id].item()
                        # semantic_label_ = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))

                        print(f'[Point Couds] save {proposal_id} / {nclusters}, label = {CAD_labels[semantic_label]}')

                        if not cfg.use_gt_seg_and_bbox:
                            score = cluster_scores[proposal_id]
                        else:
                            score = 1
                        export_pc_xyz(completed_pc, os.path.join(result_dir, "cano_completed_pc",
                                                                 f"{test_scene_name}_{proposal_id}_gt_{pred_corresponding_gt_mesh_id[proposal_id]}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))

                        export_pc_xyz(partial_pc, os.path.join(result_dir, "cano_partial_pc",
                                                               f"{test_scene_name}_{proposal_id}_gt_{pred_corresponding_gt_mesh_id[proposal_id]}_{CAD_labels[semantic_label]}_{score:.4f}.ply"))

            # save bbox
            if epoch > cfg.prepare_epochs and cfg.save_bbox:
                cluster_bboxes = preds['cluster_bboxes']

                results = np.concatenate ([cluster_bboxes,
                                           cluster_final_scores.unsqueeze(1).detach().cpu().numpy(),
                                           cluster_semantic_id.unsqueeze(1).detach().cpu().numpy(),
                                           ],1)
                # print('cat.shape',results.shape)
                os.makedirs(os.path.join(result_dir, 'bbox'), exist_ok=True)
                np.save(os.path.join(result_dir, 'bbox', test_scene_name + '.npy'), results)

            # # save gt_mesh_id:
            # if epoch > cfg.prepare_epochs :
            #     os.makedirs(os.path.join(result_dir, 'gt_id'), exist_ok=True)
            #     np.save(os.path.join(result_dir, 'gt_id', test_scene_name + '.npy'), pred_corresponding_gt_mesh_id)


            # save alignments
            if epoch > cfg.prepare_epochs and cfg.retrieval:
                os.makedirs(os.path.join(result_dir, 'alignment'), exist_ok=True)

            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()

            ##### print
            if epoch > cfg.prepare_epochs:
                logger.info("instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(batch['id'][0] + 1, len(dataset.test_files), N, nclusters, end, end1, end3))
            else:
               logger.info("instance iter: {}/{} point_num: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(batch['id'][0] + 1, len(dataset.test_files), N, end, end1, end3))


if __name__ == '__main__':
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if cfg.network == 'rfs':
        from model.rfs import RfSNet as Network
        from model.rfs import model_fn_decorator
    elif cfg.network == 'network':
        from model.network import Network
        from model.model_fn_decorator import model_fn_decorator
    
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    # utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain_path)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, cfg.test_epoch)
