import sys
sys.path.append('.')
import trimesh
import numpy as np
import os
import pickle
from glob import glob
from visualization_utils import vis_actors_vtk,get_pc_actor_vtk,get_mesh_actor_vtk,pc_fit2standard_rotation_only # visualization

from trimesh.exchange.binvox import voxelize_mesh
from iou.metrics import compute_mesh_iou
from cd.metrics import chamfer_distance, sample_and_calc_chamfer



################################# test stuff################################

def get_scene_names(root_path='../original_dimr/exp/scannetv2/rfs/rfs_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes/'):
    file_names = os.listdir(root_path)
    scene_names = list(set([filename[0:12] for filename in file_names]))
    return scene_names

def get_scene_meshes(pred_mesh_root_path='../original_dimr/exp/scannetv2/rfs/rfs_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes/',
                     scene_name ='scene0606_02',
                     gt_mesh_root_path = '../dimr/datasets/gt_meshes/test_meshes/',
                     gt_bbox_root_path = '../2dimr/datasets/gt_bboxes/'):


    gt_meshes = []
    pred_meshes = []

    gt_bbox_file = os.path.join(gt_bbox_root_path,scene_name+'.npy')
    gt_boboxes = np.load(gt_bbox_file)
    pred_bbox_file = os.path.join(pred_mesh_root_path.replace('meshes','bbox'),scene_name+'.npy')
    pred_bboxes = np.load(pred_bbox_file)

    GT_mesh_paths = []
    proposal_mesh_paths = glob(os.path.join(pred_mesh_root_path.replace('[','?').replace(']','?'),scene_name+'*'))




    for targetId, proposal_mesh_path in enumerate(proposal_mesh_paths):
        # print(proposal_mesh_path)
        pred_mesh = trimesh.load(proposal_mesh_path)
        pred_mesh.vertices  = pc_fit2standard_rotation_only(pred_mesh.vertices,pred_bboxes[targetId])
        pred_meshes.append(pred_mesh)

        proposal_mesh_path_last = os.path.basename(proposal_mesh_path)

        # gt_id = re.findall(r"_(\d)+_gt", path)[0].replace('_', '').replace('gt', '')
        gt_id = proposal_mesh_path_last[:-4].split('_')[4]  # modified by Qiao
        # match_str = os.path.join(gt_mesh_root_path,scene_name+'_'+gt_id+'_*')
        # print(match_str)
        gt_mesh_path = glob(os.path.join(gt_mesh_root_path,scene_name+'_'+gt_id+'_*'))[0]
        gt_mesh = trimesh.load(gt_mesh_path)
        gt_mesh.vertices = pc_fit2standard_rotation_only(gt_mesh.vertices, gt_boboxes[int(gt_id)])

        gt_meshes.append(gt_mesh)
        # print('gt', os.path.basename(gt_mesh_path))
        # gt_meshes[targetId].show()
        # print('pred',proposal_mesh_path_last)
        # pred_meshes[targetId].show()
        # print("----------------------------------")
        vis_actors_vtk([
            get_mesh_actor_vtk(gt_mesh),
            get_mesh_actor_vtk(pred_mesh),

        ])

    return pred_meshes,gt_meshes




if __name__ == '__main__':
    import os
    print(os.path.exists('../2dimr/datasets/gt_bboxes/scene0606_02.npy'))
    root_path = '../original_dimr/exp/scannetv2/rfs/rfs_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes/'
    scene_names = get_scene_names(root_path)
    # print(scene_names)
    # print(len(scene_names))
    get_scene_meshes()


    # # scene_names = ['scene0606_02','scene0377_02']
    # scene_num = len(scene_names)
    #
    # n_points = 5000
    # iou_res = [32,64]
    # evaluator = MeshEvaluator(n_points=n_points)
    #
    #
    # mean_completeness_per_scene = []
    # mean_accuracy_per_scene = []
    #
    # mean_chamfer_L2_per_scene = []
    # mean_chamfer_L1_per_scene = []
    # mean_iou_per_scene = []
    #
    # pred_num_per_scene = []
    #
    # for scene_id,scene_name in enumerate(scene_names):
    #
    #     completeness = 0
    #     accuracy = 0
    #     chamfer_L2 = 0
    #     chamfer_L1 = 0
    #     # iou = 0
    #     iou ={}
    #     for res in iou_res:
    #         iou[res] = 0
    #     meshes, gt_meshes, proposal_mesh_paths, GT_mesh_paths = get_scene_meshes(root_path,scene_name)
    #
    #     pred_num =len(meshes)
    #     pred_num_per_scene.append(pred_num)
    #
    #     if meshes is not None:
    #         mesh_id = 0
    #         for gt_mesh, mesh in zip(gt_meshes, meshes):
    #         # for gt_mesh, mesh in zip(gt_meshes, gt_meshes):
    #             # vis_actors_vtk([get_trimesh_actor_vtk(gt_mesh),get_trimesh_actor_vtk(mesh)])
    #             mesh_id +=1
    #             print("scene %d/%d, %s, proposal %d/%d"%(scene_id,scene_num,scene_name,mesh_id,pred_num))
    #
    #             GT_point_cloud , idx = gt_mesh.sample(n_points, return_index=True)
    #             GT_point_cloud = GT_point_cloud.astype(np.float32)
    #             GT_normals = gt_mesh.face_normals[idx]
    #
    #             eval_dict_mesh = evaluator.eval_mesh(mesh, GT_point_cloud, None)
    #             # print(eval_dict_mesh)
    #             # vis_actors_vtk([get_trimesh_actor_vtk(mesh),get_pc_actor_vtk(GT_point_cloud[i].detach().cpu().numpy())])
    #
    #             completeness +=  eval_dict_mesh['completeness']
    #             accuracy += eval_dict_mesh['accuracy']
    #             chamfer_L2 += eval_dict_mesh['chamfer-L2']
    #             chamfer_L1 += eval_dict_mesh['chamfer-L1']
    #             '''voxelized iou'''
    #             for res in iou_res:
    #                 pred_voxels = voxels.voxelize_ray(mesh, res)
    #                 gt_voxels = voxels.voxelize_ray(gt_mesh, res)
    #
    #                 temp_iou = compute_iou(np.expand_dims(pred_voxels, 0),
    #                                    np.expand_dims(gt_voxels, 0))[0]
    #                 iou[res] = iou[res] + temp_iou
    #                 # print('difference', np.sum(pred_voxels.astype('int8') - gt_voxels.astype('int8')))
    #                 # print('temp_iou', temp_iou)
    #                 # print('iou',iou)
    #
    #
    #                 # print(pred_voxels)
    #                 # print('voxels_occ.shape',pred_voxels.shape)
    #                 # vis_occ_hat_voxel_vtk(file=None, data=pred_voxels, all=True)
    #                 # vis_occ_hat_voxel_vtk(file=None, data=gt_voxels[i], all=True)
    #                 # print('gt_voxels[i].shape', gt_voxels[i].shape)
    #
    #             '''normals'''
    #             # pointcloud, idx = mesh.sample(self.n_points, return_index=True)
    #             # pointcloud = pointcloud.astype(np.float32)
    #             # normals = mesh.face_normals[idx]
    #             # print('pointcloud.shape',pointcloud.shape)
    #             # print('normals.shape',pointcloud.shape)
    #
    #
    #
    #         # completeness = completeness/pred_num
    #         # accuracy = accuracy/pred_num
    #         chamfer_L2 = chamfer_L2/pred_num
    #         chamfer_L1 = chamfer_L1/pred_num
    #
    #         # print("before devide iou",iou)
    #         # print("pred_num",pred_num)
    #         for res in iou_res:
    #             iou[res] = iou[res]/pred_num
    #         # print("after devide iou", iou)
    #
    #         mean_chamfer_L2_per_scene.append(chamfer_L2)
    #         mean_chamfer_L1_per_scene.append(chamfer_L1)
    #         mean_iou_per_scene.append(iou)
    #     # break
    #
    # print(mean_iou_per_scene)
    # print(mean_chamfer_L2_per_scene)
    # print(mean_chamfer_L1_per_scene)
    # print(pred_num_per_scene)
    #
    #
    # scene_id = 0
    # # iou = 0
    # iou = {}
    # for res in iou_res:
    #     iou[res] = 0
    # CD_l2 = 0
    # CD_l1 = 0
    # all_pred_num = sum(pred_num_per_scene)
    # for scene_iou, scene_CD_l2, scene_CD_l1,scene_obj_num in \
    #     zip(mean_iou_per_scene,mean_chamfer_L2_per_scene,mean_chamfer_L1_per_scene,pred_num_per_scene):
    #     for res in iou_res:
    #         iou[res] += scene_iou[res] * scene_obj_num /all_pred_num
    #     CD_l2 += scene_CD_l2 * scene_obj_num /all_pred_num
    #     CD_l1 += scene_CD_l1 * scene_obj_num /all_pred_num
    #
    # print()
    # print("-------------------------------")
    # print("Metrics of",root_path)
    #
    # print("CD_l2:",CD_l2)
    # print("CD_l1:",CD_l1)
    # print("iou:", iou)
