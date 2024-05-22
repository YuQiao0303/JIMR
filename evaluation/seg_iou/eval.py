# offline mesh IoU eval
# 2 modifications by Qiao
# extract_label, 3 or 5
# pred_files, replace
# use like:

# python evaluation/seg_iou/eval.py ./datasets/gt_seg ../dimr/exp/scannetv2/rfs/my_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/instance_seg/
# python evaluation/seg_iou/eval.py ./datasets/gt_seg ../dimr/exp/scannetv2/rfs/2022.10.18.14.58_test_40_dimrgroup_mean_testset_/result/epoch256_nmst0.3_scoret0.05_npointt100/val/instance_seg/
# python evaluation/seg_iou/eval.py ./datasets/gt_seg ../dimr/exp/scannetv2/rfs/2022.10.18.15.03_test_40_softgroup_mean_testset/result/epoch256_nmst0.3_scoret0.05_npointt100/val/instance_seg/
# python evaluation/seg_iou/eval.py ./datasets/gt_seg ../dimr/exp/scannetv2/rfs/2022.10.18.15.09_test_40_softgroup_mean_trainset/result/epoch256_nmst0.3_scoret0.05_npointt100/val/instance_seg/
# python evaluation/seg_iou/eval.py ./datasets/gt_seg ../dimr/exp/scannetv2/rfs/2022.10.18.15.11_test_40_dimrgroup_mean_trainset/result/epoch256_nmst0.3_scoret0.05_npointt100/val/instance_seg/

import open3d as o3d
import numpy as np
import os
import glob
import trimesh

from metrics import *


def eval(gt_dir, pred_dir, threshs):
    assert (os.path.exists(gt_dir))
    assert (os.path.exists(pred_dir))
    log_file = open(f'eval_log_seg_iou.txt', 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]


    # collect bboxes (npy) # modify here, load data not as meshes
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npz')))
    # pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir.replace('[', '?').replace(']', '?'), '*.npz')))

    data = {}

    for f in gt_files:
        scene_name = os.path.basename(f)[:12]
        if scene_name not in data: data[scene_name] = {}
        if 'gt' not in data[scene_name]: data[scene_name]['gt'] = []
        if 'pred' not in data[scene_name]: data[scene_name]['pred'] = []
        data[scene_name]['gt'].append(f)

    for f in pred_files:
        scene_name = os.path.basename(f)[:12]
        data[scene_name]['pred'].append(f)
        # print(scene_name)

    scene_names = data.keys()
    # scene_names = ['scene0011_00']

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):
        # gt seg
        gt_files_scene = data[scene_name]['gt']
        info_mesh_gts = []
        for f in gt_files_scene: # only one
            gt_data = np.load(f)# mesh = trimesh.load(f, process=False)
            label = (gt_data['instance_semantic_labels'])
            seg = (gt_data['instance_labels'])
            instance_pointnum = gt_data['instance_pointnum']
            info_mesh_gts.append((label,seg, instance_pointnum))# info_mesh_gts.append((label, mesh))

        # pred seg
        pred_files_scene = data[scene_name]['pred']

        info_mesh_preds = []
        for f in pred_files_scene:
            pred_data = np.load(f)# mesh = trimesh.load(f, process=False)

            label =  int(pred_data['semantic_label'])
            score = pred_data['score']
            cluster_point_idxs = pred_data['cluster_point_idxs']
            info_mesh_preds.append((label, cluster_point_idxs,score))#info_mesh_preds.append((label, mesh, score))

        # record
        for calc in ap_calculator_list:
            calc.step(info_mesh_preds, info_mesh_gts)

        print(
            f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(gt_files_scene)},  #pred = {len(pred_files_scene)}')

    # output
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- thresh = {threshs[i]} -----')
        print(f'----- thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            if 'Q_mesh' in k: continue
            if 'mesh' not in k: continue
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file)

    log_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str)
    parser.add_argument('pred_dir', type=str)

    args = parser.parse_args()

    eval(args.gt_dir, args.pred_dir, threshs=[0.25,0.5])


