# offline mesh IoU eval
# 2 modifications by Qiao
# extract_label, 3 or 5
# pred_files, replace
# use like:
# python evaluation/bbox_iou/eval.py ./datasets/gt_bboxes ../dimr/exp/scannetv2/rfs/2022.10.17.10.22_test_train_all_104_mean/result/epoch256_nmst0.3_scoret0.05_npointt100/val/bbox/
# python evaluation/bbox_iou/eval.py ./datasets/gt_bboxes ../dimr/exp/scannetv2/rfs/2022.10.17.09.59_test_train_all_40_testset14_weighted_centerness/result/epoch256_nmst0.3_scoret0.05_npointt100/val/bbox/
# python evaluation/bbox_iou/eval.py ./datasets/gt_bboxes ../dimr/exp/scannetv2/rfs/2022.10.17.09.57_test_train_all_40_testset14_weighted_cls/result/epoch256_nmst0.3_scoret0.05_npointt100/val/bbox/
# python evaluation/bbox_iou/eval.py ./datasets/gt_bboxes ../dimr/exp/scannetv2/rfs/2022.10.17.09.31_test_train_all_40_testset14_mean/result/epoch256_nmst0.3_scoret0.05_npointt100/val/bbox/
# python evaluation/bbox_iou/eval.py ./datasets/gt_bboxes ../dimr/exp/scannetv2/rfs/my_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/bbox/
# python evaluation/bbox_iou/eval.py ./datasets/gt_bboxes ../original_dimr/exp/scannetv2/rfs/training_set_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/bbox/
# python evaluation/bbox_iou/eval.py ./datasets/gt_bboxes ../original_dimr/exp/scannetv2/rfs/test_set_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/bbox/

import open3d as o3d
import numpy as np
import os
import glob
import trimesh

from metrics import *
import pickle
shapenet_cat_id_str2int = {None:-100, '04379243':0, '03001627':1, '02871439':2, '04256520':3, '02747177':4,
                           '02933112':5, '03211117':6, '02808440':7}
def eval(gt_dir, pred_dir, threshs):
    '''

    :param gt_dir: datasets/scannet/processed_data
    :param pred_dir:
    :param threshs:
    :return:
    '''
    log_file = open(f'eval_log_bbox_iou.txt', 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]


    # collect bboxes (ours as npy, gt as pkl)
    data = {}

    # gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
    # for f in gt_files:
    #     scene_name = os.path.basename(f)[:12]
    #     if scene_name not in data: data[scene_name] = {}
    #     if 'gt' not in data[scene_name]: data[scene_name]['gt'] = []
    #     if 'pred' not in data[scene_name]: data[scene_name]['pred'] = []
    #     data[scene_name]['gt'].append(f)

    test_split_path = 'datasets/splits/val.txt'
    with open(test_split_path, "r", encoding="utf-8") as file_obj:
        # data = file_obj.read()
        gt_test_scenes = file_obj.readlines()  # list
        # data = file_obj.readline()
        # print(data)
    gt_test_scenes = sorted(gt_test_scenes)
    for i in range(len(gt_test_scenes)):
        gt_test_scenes[i] = gt_test_scenes[i].replace("\n", "")
    scene_names = gt_test_scenes

    for scene_name in scene_names:
        data[scene_name] = {}
        data[scene_name]['gt'] = []
        data[scene_name]['pred'] = []
        data[scene_name]['gt'].append('datasets/scannet/processed_data/{scene_name}/bbox.pkl')


    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.npy')))
    for f in pred_files:
        scene_name = os.path.basename(f)[:12]
        data[scene_name]['pred'].append(f)

    scene_names = data.keys()
    # scene_names = ['scene0011_00']

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):
        # gt mesh
        gt_files_scene = data[scene_name]['gt']
        info_mesh_gts = []
        for f in gt_files_scene: # only one
            # bboxes = np.load(f)# mesh = trimesh.load(f, process=False)
            # label = bboxes[:,8]
            # bboxes = bboxes[:, :7]
            # info_mesh_gts.append((label, bboxes))# info_mesh_gts.append((label, mesh))
            with open(f'datasets/scannet/processed_data/{scene_name}/bbox.pkl', 'rb') as f:
                bbox_info = pickle.load(f)  # list of dictionaries
            bboxes = []
            labels = []
            for i in range(len(bbox_info)):
                box = bbox_info[i]['box3D']
                cls_str = bbox_info[i]['shapenet_catid']
                cls_id = shapenet_cat_id_str2int[cls_str]
                bboxes.append(box)
                labels.append(cls_id)
            bboxes = np.vstack(bboxes)
            print('bboxes.shape',bboxes.shape)
            labels = np.array(labels)
            print('labels.shape', labels.shape)
            info_mesh_gts.append((labels, bboxes))  # info_mesh_gts.append((label, mesh))
            # return


        # pred mesh
        pred_files_scene = data[scene_name]['pred']

        info_mesh_preds = []
        for f in pred_files_scene:
            bboxes = np.load(f) # mesh = trimesh.load(f, process=False)
            label =  bboxes[:,8]
            score = bboxes[:,7]
            bboxes = bboxes[:,:7]
            info_mesh_preds.append((label, bboxes,score))#info_mesh_preds.append((label, mesh, score))

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


