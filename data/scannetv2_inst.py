'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
'''

import os, sys, glob, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import scipy.ndimage
import scipy.interpolate
import torch
import pickle
from torch.utils.data import DataLoader
import trimesh
import tqdm

sys.path.append('./')

from util.config import cfg
from util.log import logger
from util.bbox import BBoxUtils
from util.consts import *
from lib.pointgroup_ops.functions import pointgroup_ops

from util.utils import pc_fit2standard

from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, get_mesh_actor_vtk, \
    vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization

BBox = BBoxUtils()


def read_txt(file):
    with open(file, 'r') as f:
        output = [x.strip() for x in f.readlines()]
    return output

class Dataset:
    def __init__(self, test=False):
        #self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale # 50
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        if test:
            self.batch_size = 1 # must be 1 !
            self.test_split = cfg.split  # val or test or train
            self.test_workers = cfg.test_workers


    def trainLoader(self):
        self.train_files = read_txt(os.path.join('datasets/splits/', 'train.txt'))

        logger.info('Training samples: {}'.format(len(self.train_files)))

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=True)


    def valLoader(self):
        self.val_files = read_txt(os.path.join('datasets/splits/', 'val.txt'))

        logger.info('Validation samples: {}'.format(len(self.val_files)))

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers,
                                          shuffle=False, drop_last=False, pin_memory=True)


    def testLoader(self):
        self.test_files = read_txt(os.path.join('datasets/splits/', self.test_split + '.txt'))

        logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.test_files)))

        test_set = list(np.arange(len(self.test_files)))

        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.testMerge, num_workers=self.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True)

    #Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


    def getInstanceInfo(self, xyz, instance_label, zs, bboxes, bbox_labels, shapenet_catids, shapenet_ids, inst2bbox):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_num = int(instance_label.max()) + 1
        # instance_info = np.ones((xyz.shape[0], 11), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz, angle_label, angle_residual)
        instance_info = np.ones((xyz.shape[0], 16), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz, bbox7)
        instance_pointnum = []   # (nInst), int
        # instance_zs = np.zeros((instance_num, 256)) # [nInst, 256]
        instance_zs_valid = np.zeros((instance_num)) # [nInst]
        instance_bboxes = np.zeros((instance_num, 7))
        instance_bbox_center = np.zeros((instance_num, 3))
        instance_bbox_size = np.zeros((instance_num))
        instance_bbox_size_residual = np.zeros((instance_num, 3))
        instance_bbox_angle = np.zeros((instance_num))
        instance_bbox_angle_residual = np.zeros((instance_num))
        instance_shapenet_catids = [None] * instance_num
        instance_shapenet_ids = [None] * instance_num

        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_) # returns a one-element tuple, like ([0,1,29,43,...],)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            centroid_xyz_i = (min_xyz_i + max_xyz_i) / 2

            instance_info[inst_idx_i, 0:3] = centroid_xyz_i # mean_xyz_i
            instance_info[inst_idx_i, 3:6] = min_xyz_i
            instance_info[inst_idx_i, 6:9] = max_xyz_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

            ### find the corresponding z_vectors for this instance.
            if i_ in inst2bbox:
                instance_zs_valid[i_] = 1
                # instance_zs[i_] = zs[inst2bbox[i_]]
                instance_bboxes[i_] = bboxes[inst2bbox[i_]]
                instance_bbox_center[i_] = bboxes[inst2bbox[i_], 0:3]
                instance_bbox_size[i_] = bbox_labels[inst2bbox[i_]] # use class label to get mean bbox size
                instance_bbox_size_residual[i_] = bboxes[inst2bbox[i_], 3:6] - BBox.mean_size_arr[bbox_labels[inst2bbox[i_]], :]
                angle_class, angle_residual = BBox.angle2class(bboxes[inst2bbox[i_], 6])
                instance_bbox_angle[i_] = angle_class
                instance_bbox_angle_residual[i_] = angle_residual
                instance_shapenet_catids[i_] = shapenet_catids[inst2bbox[i_]]
                instance_shapenet_ids[i_] = shapenet_ids[inst2bbox[i_]]


                instance_info[inst_idx_i, 9] = angle_class  # here is angle parameter
                instance_info[inst_idx_i, 10] = angle_residual

                # # added 7-dim bbox parameters: xyz, whl, angle
                # instance_info[inst_idx_i, 9:12] = bboxes[inst2bbox[i_], 0:3] # xyz
                # instance_info[inst_idx_i, 12:15] = bboxes[inst2bbox[i_], 3:6] # whl
                # instance_info[inst_idx_i, 15] = bboxes[inst2bbox[i_], 6] # angle



        return instance_num, instance_info, instance_pointnum, \
               None, instance_zs_valid, instance_bbox_center, instance_bbox_size, instance_bbox_size_residual, instance_bbox_angle, instance_bbox_angle_residual, \
               instance_shapenet_catids, instance_shapenet_ids, instance_bboxes

    # original
    def dataAugment(self, xyz, boxes3D, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
            # xyz += np.random.randn(3, 3) * 0.01
        if flip:
            # m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
            # boxes3D[:, 6] = np.sign(boxes3D[:, 6]) * np.pi - boxes3D[:, 6]

            flip_rand = np.random.randint(0, 2) * 2 - 1
            # m[0][0] *= flip_rand  # flip x randomly. does not work
            if flip_rand == -1:
                # print('flip')
                m[0][0] = -1
                if boxes3D is not None:
                    boxes3D[:, 6] = np.sign(boxes3D[:, 6]) * np.pi - boxes3D[:, 6]
            else:
                # print('not flip')
                pass

        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
            if boxes3D is not None:
                boxes3D[:, 6] += theta
                boxes3D[:, 6] = np.mod(boxes3D[:, 6] + np.pi, 2 * np.pi) - np.pi

        xyz =  np.matmul(xyz, m)
        # print('xyz',xyz)
        if boxes3D is not None:
            boxes3D[:, 0:3] = np.matmul(boxes3D[:, 0:3], m)
        return xyz, boxes3D

    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        remap = {}

        j = 0
        while (j <= instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                remap[instance_label.max()] = j
                instance_label[instance_label == instance_label.max()] = j
            else:
                remap[j] = j
            j += 1

        return instance_label, remap

    def Merge(self, id, split='train', augment=True):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 11)
        instance_pointnum = []  # (total_nInst), int
        # instance_zs = [] # (total_nInst, 256)
        instance_zs_valid = [] # (total_nInst), bool
        instance_bboxes = []
        instance_bbox_center = []
        instance_bbox_size = []
        instance_bbox_size_residual = []
        instance_bbox_angle = []
        instance_bbox_angle_residual = []
        instance_meshes = []
        instance_pc =[]
        instance_points_for_occ =[]
        instance_occ =[]
        instance_semantic_CAD = []

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):

            if split == 'train':
                scan_name = self.train_files[idx]
            elif split == 'val':
                scan_name = self.val_files[idx]
            elif split == 'test':
                scan_name = self.test_files[idx] # in fact, this is the same as val_files

            scan_data = np.load(f'datasets/scannet/processed_data/{scan_name}/data.npz')

            point_cloud = scan_data['mesh_vertices'].astype(np.float32)
            xyz_origin = point_cloud[:, 0:3]

            rgb = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

            label = scan_data['semantic_labels'].astype(np.int32)
            label[label == 255] = -100 # ignore

            instance_label = scan_data['instance_labels'].astype(np.int32) - 1
            instance_label[instance_label == -1] = -100 # -100 == unannotated, instance id is 0-start

            ### load zs, bbox, find correponding with instance_labels
            # zs = np.load(f'datasets/bsp/zs/{scan_name}/zs.npz')['zs'] # [nInst, 256] mean + logvar

            ### load bbox
            with open(f'datasets/scannet/processed_data/{scan_name}/bbox.pkl', 'rb') as f:
                bbox_info = pickle.load(f)

            bbox2inst = []
            bboxes = []
            bbox_labels = []
            shapenet_catids = []
            shapenet_catids_int = []
            shapenet_ids = []

            # print('label.min max', label.min(), label.max()) # -100, 23
            # print('label.shape',label.shape) # all poitn num
            # print('instance_label.shape',instance_label.shape) # all poitn num
            # print('----------------------------------------')
            # instance_points_actor_list = []

            for item in bbox_info:
                # added by Qiao to filter out some instance
                instance_id = item['instance_id'] - 1 ## pointgoup instance is 0-start, while scannet original is 1-start.
                instance_points = xyz_origin[instance_label == instance_id]
                if instance_points.shape[0]>0:
                    # instance_points_semantic_ids = label[instance_label == instance_id]
                    # from scipy import stats
                    # instance_semnantic_label = stats.mode(instance_points_semantic_ids)[0][0]

                    instance_bbox = item['box3D']
                    instance_points_canonical = pc_fit2standard(instance_points, instance_bbox)

                    # filter out instances with WDH too bigger than gt (after normalization, biggest WDH for gt will be 1)
                    instance_points_canonical_scale_max = np.max(np.abs(instance_points_canonical))

                    if instance_points_canonical_scale_max > 0.8: # should be within 0.5
                        continue
                        # # print("this one is not ok!")
                        # # all_points_canonical = pc_fit2standard(xyz_origin, instance_bbox)
                        # from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, get_mesh_actor_vtk, \
                        #     vis_occ_hat_voxel_vtk  # vis_np_histogram,# visualization
                        # # vis_actors_vtk([
                        # #     get_pc_actor_vtk(instance_points_canonical,color = (1,0,0)),
                        # #     get_pc_actor_vtk(all_points_canonical,alpha = 0.1),
                        # # ])
                        #
                        # # use bbox and instance labels instead
                        #
                        # mask = get_points_inside_bbox(xyz_origin.copy(), instance_bbox)
                        # # mask = np.logical_and(mask,  label== instance_semnantic_label)  # exclude already labeled areas
                        # new_instance_points = xyz_origin[mask]
                        # new_instance_points = pc_fit2standard(new_instance_points, instance_bbox)
                        # vis_actors_vtk([
                        #     get_pc_actor_vtk(new_instance_points, color=(1, 0, 0)),
                        #
                        # ])

                    # instance_points_actor_list.append(get_pc_actor_vtk(instance_points))

                    bbox2inst.append(item['instance_id'] - 1) # pointgoup instance is 0-start, while scannet original is 1-start.
                    bboxes.append(item['box3D'])
                    bbox_labels.append(BBox.shapenetid2class[item['cls_id']])
                    shapenet_catids.append(item['shapenet_catid'])
                    # shapenet_catids_int.append(item['shapenet_catid'])
                    shapenet_ids.append(item['shapenet_id'])

            # vis_actors_vtk(instance_points_actor_list)

            # print('len(bbox_info)',len(bbox_info))
            # print('len(bboxes)',len(bboxes))
            if len(bboxes)>0:
                bboxes = np.stack(bboxes, axis=0)
                bbox_labels = np.stack(bbox_labels, axis=0)
            else:
                bboxes = None
                bbox_labels = None

            # so far not mapped
            # # print('instance_label.shape',instance_label.shape)
            # print('instance_label max',np.max(instance_label))
            # print('bboxes.shape',bboxes.shape)
            # print('bbox_labels.shape',bbox_labels.shape)
            # print("-----------------------------")
            # print('xyz_origin before transform_train', xyz_origin)
            ### jitter / flip x / rotation
            if augment:
                xyz_middle, bboxes = self.dataAugment(xyz_origin, bboxes)
            else:
                xyz_middle = xyz_origin
            # print('xyz_middle after dataAugment', xyz_middle)
            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            if augment:
                xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
                xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)
         
            ### offset
           
            xyz -= xyz.min(0)
      
            ### crop
            if augment:
                xyz, valid_idxs = self.crop(xyz)
            else:
                valid_idxs = np.ones(xyz.shape[0]).astype(bool)
        

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]

            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
           
            instance_label, instance_remap = self.getCroppedInstLabel(instance_label, valid_idxs) # instance_remap is a dict

            # print('instance_remap',instance_remap) # a dict of {int: int}
            # print('bbox2inst',bbox2inst) # list of int

            # remap
            inst2bbox = {}
            for ii in range(len(bbox2inst)):
                if bbox2inst[ii] in instance_remap:
                    inst2bbox[instance_remap[bbox2inst[ii]]] = ii

            # print('inst2bbox',inst2bbox) # a dict, instid:bboxid
            # print('-----------------------------------------')

            ### get instance information
            inst_num, inst_info, inst_pointnum, inst_zs, inst_zs_valid, inst_bbox_center, inst_bbox_size, inst_bbox_size_residual, inst_bbox_angle, inst_bbox_angle_residual, inst_shapenet_catids, inst_shapenet_ids, inst_bboxes = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), None, bboxes, bbox_labels, shapenet_catids, shapenet_ids, inst2bbox)
            # print('inst_info.shape',inst_info.shape) # point wise
            # print('inst_bboxes.shape',inst_bboxes.shape) # instance wise
            instance_label[np.where(instance_label != -100)] += total_inst_num # batchify
            total_inst_num += inst_num

            ### get gt instance meshes
            if split == 'test' and cfg.eval:
                meshes = []
                for mesh_id in range(len(inst_shapenet_catids)):

                    shapenet_catid = inst_shapenet_catids[mesh_id]
                    shapenet_id = inst_shapenet_ids[mesh_id]
                    bbox = inst_bboxes[mesh_id]

                    if shapenet_catid is None or shapenet_id is None:
                        meshes.append(None)
                        continue

                    mesh_file = os.path.join('datasets/ShapeNetv2_data/watertight_scaled_simplified', shapenet_catid, shapenet_id + '.off')
                    mesh = trimesh.load(mesh_file, process=False)

                    points = mesh.vertices

                    # swap axes
                    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
                    points = points.dot(transform_m.T)

                    # recenter + rescale
                    min_xyz = points.min(0)
                    max_xyz = points.max(0)
                    points = points - (max_xyz + min_xyz) / 2
                    points = points / (max_xyz - min_xyz) * bbox[3:6]

                    # rotate
                    orientation = bbox[6]
                    axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
                    points = points.dot(axis_rectified)

                    # translate
                    points = points + bbox[0:3]

                    mesh.vertices = points
                    meshes.append(mesh)

                instance_meshes.append(meshes)

            load_instance_pc = True
            if load_instance_pc:
                point_num = 2048
                point_clouds = []
                for mesh_id in range(len(inst_shapenet_catids)):

                    shapenet_catid = inst_shapenet_catids[mesh_id]
                    shapenet_id = inst_shapenet_ids[mesh_id]
                    bbox = inst_bboxes[mesh_id]

                    if shapenet_catid is None or shapenet_id is None:
                        # print(mesh_id, shapenet_catid,shapenet_id)
                        point_clouds.append({'point_cloud': np.zeros((point_num,3))})  #
                        continue

                    pc_file = os.path.join('datasets/ShapeNetv2_data/pointcloud', shapenet_catid,
                                             shapenet_id + '.npz')

                    points_dict = np.load(pc_file)
                    points = points_dict['points']
                    # print(points.shape) # 10w,3
                    # Break symmetry if given in float16:
                    if points.dtype == np.float16 and split == 'train':
                        points = points.astype(np.float32)
                        points += 1e-4 * np.random.randn(*points.shape)
                    else:
                        points = points.astype(np.float32)

                    # down sampling
                    idx = np.random.randint(points.shape[0], size=2048)
                    # print(idx.shape)
                    points = points[idx, :]
                    points = points.astype(np.float32)

                    # # swap axes
                    # transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
                    # points = points.dot(transform_m.T)
                    #
                    # # recenter + rescale
                    # min_xyz = points.min(0)
                    # max_xyz = points.max(0)
                    # points = points - (max_xyz + min_xyz) / 2
                    # points = points / (max_xyz - min_xyz) * bbox[3:6]
                    #
                    # # rotate
                    # orientation = bbox[6]
                    # axis_rectified = np.array(
                    #     [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0],
                    #      [0, 0, 1]])
                    # points = points.dot(axis_rectified)
                    #
                    # # translate
                    # points = points + bbox[0:3]
                    point_clouds.append({'point_cloud': points})
                # print('point_clouds before recursive',point_clouds)

                point_clouds = recursive_cat_to_numpy(point_clouds)['point_cloud']
                # print('point_clouds after recursive',point_clouds)
                # print('point_clouds.keys()',point_clouds.keys())
                # print('point_clouds.shape',point_clouds.shape)
                # print(instance_pc)

                # instance_pc.append({'point_cloud': point_clouds})
                instance_pc.append(torch.from_numpy(point_clouds))

            load_instance_p_occ = True
            if load_instance_p_occ:
                point_num = 2048
                points_for_occ = []
                for mesh_id in range(len(inst_shapenet_catids)):
                    shapenet_catid = inst_shapenet_catids[mesh_id]
                    shapenet_id = inst_shapenet_ids[mesh_id]
                    bbox = inst_bboxes[mesh_id]
                    if shapenet_catid is None or shapenet_id is None:
                        # print(mesh_id, shapenet_catid,shapenet_id)
                        points_for_occ.append({
                            'points': np.zeros((point_num,3)),
                            'occ': np.zeros(point_num),
                        })  #
                        continue

                    pc_file = os.path.join('datasets/ShapeNetv2_data/point', shapenet_catid, shapenet_id + '.npz')

                    points_dict = np.load(pc_file)
                    points = points_dict['points']
                    # print(points.shape) # 10w,3
                    # Break symmetry if given in float16:
                    if points.dtype == np.float16 and split == 'train':
                        points = points.astype(np.float32)
                        points += 1e-4 * np.random.randn(*points.shape)
                    else:
                        points = points.astype(np.float32)
                    occupancies = points_dict['occupancies']
                    # if self.points_unpackbits: # seem like true
                    occupancies = np.unpackbits(occupancies)[:points.shape[0]]
                    occ = occupancies.astype(np.float32)

                    # down sampling
                    Nt_out = 1024 # choose Nt_out points outside the shape surface
                    Nt_in = 1024 # same for inside
                    occ_binary = (occ >= 0.5)
                    if (occ_binary.shape[0] == 1):
                        occ_binary.reshape(100000)
                    points0 = points[~occ_binary]
                    points1 = points[occ_binary]

                    if split  == 'test':
                        idx0 = np.arange(0, Nt_out)
                        idx1 = np.arange(0, Nt_in)
                    else:
                        idx0 = np.random.randint(points0.shape[0], size=Nt_out)
                        idx1 = np.random.randint(points1.shape[0], size=Nt_in)

                    points0 = points0[idx0, :]
                    points1 = points1[idx1, :]
                    points = np.concatenate([points0, points1], axis=0)

                    occ0 = np.zeros(Nt_out, dtype=np.float32)
                    occ1 = np.ones(Nt_in, dtype=np.float32)
                    occ = np.concatenate([occ0, occ1], axis=0)

                    data={
                        'points': points,
                        'occ': occ
                    }

                    points_for_occ.append(data)
                # print('point_clouds before recursive',point_clouds)
                # print('points_for_occ',points_for_occ)
                temp = recursive_cat_to_numpy(points_for_occ)
                # print('temp',temp)
                points_for_occ = temp['points']
                occ = temp['occ']
                # print(points_for_occ.shape,'points_for_occ') # so far right
                # print(occ.shape,'occ')

                # instance_pc.append({'point_cloud': point_clouds})
                instance_points_for_occ.append(torch.from_numpy(points_for_occ))
                instance_occ.append(torch.from_numpy(occ))


            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            # print('xyz before append to locs',xyz.shape,xyz)
            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))

            locs_float.append(torch.from_numpy(xyz_middle))

            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

            # instance_zs.append(torch.from_numpy(inst_zs))
            instance_zs_valid.append(torch.from_numpy(inst_zs_valid))
            instance_bbox_center.append(torch.from_numpy(inst_bbox_center))
            instance_bbox_size.append(torch.from_numpy(inst_bbox_size))
            instance_bbox_size_residual.append(torch.from_numpy(inst_bbox_size_residual))
            instance_bbox_angle.append(torch.from_numpy(inst_bbox_angle))
            instance_bbox_angle_residual.append(torch.from_numpy(inst_bbox_angle_residual))
            instance_bboxes.append(torch.from_numpy(inst_bboxes))
            instance_semantic_CAD.append( torch.tensor(np.array([shapenet_cat_id_str2int[i] for i in inst_shapenet_catids])) )


        # instance_pc = recursive_cat_to_numpy(instance_pc)['point_cloud']
        # print('instance_pc.shape', instance_pc.shape)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0], in fact this is not used later...
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3), this is the used points, nearly original...
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 11) (meanxyz, minxyz, maxxyz, angles)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        # instance_zs = torch.cat(instance_zs, 0).to(torch.float32) # float (total_nInst, 256)
        instance_zs_valid = torch.cat(instance_zs_valid, 0).to(torch.float32) # int (total_nInst,)

        instance_bbox_center = torch.cat(instance_bbox_center, 0).to(torch.float32)
        instance_bbox_size = torch.cat(instance_bbox_size, 0).to(torch.long)
        instance_bbox_size_residual = torch.cat(instance_bbox_size_residual, 0).to(torch.float32)
        instance_bbox_angle = torch.cat(instance_bbox_angle, 0).to(torch.long)
        instance_bbox_angle_residual = torch.cat(instance_bbox_angle_residual, 0).to(torch.float32)
        instance_bboxes = torch.cat(instance_bboxes, 0).to(torch.float32)
        instance_pc = torch.cat(instance_pc, 0).to(torch.float32)
        instance_points_for_occ = torch.cat(instance_points_for_occ, 0).to(torch.float32)
        instance_occ = torch.cat(instance_occ, 0).to(torch.float32)
        instance_semantic_CAD = torch.cat(instance_semantic_CAD, 0).to(torch.long) # added by Qiao
        # print('instance_points_for_occ.shape',instance_points_for_occ.shape) # so far right
        # print('instance_occ.shape',instance_occ.shape)

        # print('inst_shapenet_catids',inst_shapenet_catids)
        # print(type(inst_shapenet_catids))
        # inst_shapenet_catids =torch.cat(inst_shapenet_catids, 0).to(torch.float32)


        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)
    
        # print('self.batch_size', self.batch_size)
        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        ### added by Qiao

        # print('(inst_shapenet_catids)',(inst_shapenet_catids))
        # print('len(inst_shapenet_catids)', len(inst_shapenet_catids))
        # print('len(instance_meshes)', len(instance_meshes))
        # print('instance_bboxes.shape',instance_bboxes.shape)
        # print('instance_pc.shape',instance_pc.shape)
        # # print('------------------------------')
        # # instance_semantic_CAD = torch.tensor(np.array([shapenet_cat_id_str2int[i] for i in inst_shapenet_catids]))#.cuda()
        # print('instance_semantic_CAD',instance_semantic_CAD)


        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                # 'instance_zs': instance_zs,
                'instance_zs_valid': instance_zs_valid,
                'instance_bbox_center': instance_bbox_center, 'instance_bbox_size': instance_bbox_size, 'instance_bbox_size_residual': instance_bbox_size_residual, 'instance_bbox_angle': instance_bbox_angle, 'instance_bbox_angle_residual': instance_bbox_angle_residual, 
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                'instance_meshes': instance_meshes, 'instance_bboxes': instance_bboxes,

                'instance_pc':instance_pc,
                'instance_points_for_occ':instance_points_for_occ,
                'instance_occ':instance_occ, 'instance_shapenet_catids':inst_shapenet_catids,
                # 'instance_bboxes':inst_bboxes# added by qiao
                'instance_semantic_CAD':instance_semantic_CAD, # added by Qiao
                }

    def trainMerge(self, id):
        return self.Merge(id, 'train', False) # True

    def valMerge(self, id):
        return self.Merge(id, 'val', False)

    # If there are GTs, use this. Support online eval.
    def testMerge(self, id):
        return self.Merge(id, 'test', False)

    # If there is no GT, use this.
    def testMerge_(self, id):
        locs = []
        locs_float = []
        feats = []

        batch_offsets = [0]

        for i, idx in enumerate(id):

            scan_name = self.test_files[idx]
            scan_data = np.load(f'datasets/scannet/processed_data/{scan_name}/data.npz')

            point_cloud = scan_data['mesh_vertices'].astype(np.float32)
            xyz_origin = point_cloud[:, 0:3]
            rgb = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

            ### flip x / rotation
            xyz_middle = xyz_origin

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                         # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)           # float (N, 3)
        feats = torch.cat(feats, 0)                                       # float (N, C)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}

def recursive_cat_to_numpy(data_list):
    '''Covert a list of dict to dict of numpy arrays.'''
    out_dict = {}


    for key, value in data_list[0].items():
        if isinstance(value, np.ndarray):
            # print("recursive_cat_to_numpy,np",value)
            # print(data_list)
            # print([data[key][np.newaxis] for data in data_list])
            # print([data[key] for data in data_list])
            out_dict = {**out_dict, key: np.concatenate([data[key][np.newaxis] for data in data_list], axis=0)}
            # print(**out_dict)
            # print()
        elif isinstance(value, dict):
            # print("recursive_cat_to_numpy,dict", value)
            out_dict =  {**out_dict, **recursive_cat_to_numpy(value)}
        elif np.isscalar(value):
            out_dict = {**out_dict, key: np.concatenate([np.array([data[key]])[np.newaxis] for data in data_list], axis=0)}
        elif isinstance(value, list):
            # print("recursive_cat_to_numpy,list", value)
            out_dict = {**out_dict, key: np.concatenate([np.array(data[key])[np.newaxis] for data in data_list], axis=0)}
    # print("recursive_cat_to_numpy, end")
    return out_dict

def get_points_inside_bbox(xyz, box, expansion=0.05):
    # xyz: [N, 3]
    # box: [7], center, scale, rotz

    # subtract center
    xyz -= box[:3]
    # rotate xyz
    cos_ = np.cos(-box[6])
    sin_ = np.sin(-box[6])
    xyz[:, 0], xyz[:, 1] = xyz[:, 0] * cos_ - xyz[:, 1] * sin_, xyz[:, 0] * sin_ + xyz[:, 1] * cos_
    # mask by scale
    mask_x = np.abs(xyz[:, 0]) <= box[3] / 2 + expansion
    mask_y = np.abs(xyz[:, 1]) <= box[4] / 2 + expansion
    mask_z = np.abs(xyz[:, 2]) <= box[5] / 2 + expansion
    mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

    return mask


# added by Qiao
shapenet_cat_id_str2int = {None:-100, '04379243':0, '03001627':1, '02871439':2, '04256520':3, '02747177':4,
                           '02933112':5, '03211117':6, '02808440':7}


# extract GT meshes
if __name__ == '__main__':
    dataset = Dataset(test=True)
    dataset.testLoader()

    os.makedirs('datasets/gt_meshes/meshes', exist_ok=True)
    os.makedirs('datasets/gt_bboxes', exist_ok=True)
    os.makedirs('datasets/gt_partial_pc/partial_pc', exist_ok=True)
    os.makedirs('datasets/gt_seg', exist_ok=True)

    i=0
    for batch in tqdm.tqdm(dataset.test_data_loader):

        test_scene_name = dataset.test_files[int(batch['id'][0])]

        # tmp
        #if test_scene_name != 'scene0553_01': continue

        gt_valid_mask = batch['instance_zs_valid'].cpu().numpy()
        gt_labels = batch['instance_bbox_size'].cpu().numpy()
        gt_meshes = batch['instance_meshes'][0]
        gt_bboxes = batch['instance_bboxes'].cpu().numpy()
        instance_labels = batch['instance_labels'].cpu().numpy()

        instance_semantic_labels = batch['instance_semantic_CAD'].cpu().numpy()

        '''print few CAD instance scenes'''
        CAD_instance_num = gt_valid_mask.sum()
        if CAD_instance_num <2:
            print(i,test_scene_name,CAD_instance_num,'!!!!!!!!!!!!!!!!!!!!!!!!!')
        elif i%100 == 0:
            print(i)
        i+=1


        '''save bbox'''
        save_bbox = False
        if save_bbox:
            instance_semantic_labels = instance_semantic_labels[[not np.all(gt_bboxes[i] == 0) for i in range(gt_bboxes.shape[0])]]
            gt_bboxes = gt_bboxes[[not np.all(gt_bboxes[i] == 0) for i in range(gt_bboxes.shape[0])], :] # remove all zero lines

            results = np.concatenate([gt_bboxes,
                                      np.expand_dims(np.ones_like(instance_semantic_labels), 1)  ,
                                      np.expand_dims(instance_semantic_labels,1)
                                      ], 1)
            np.save(os.path.join('datasets/gt_bboxes',  test_scene_name + '.npy'), results)

        '''save seg'''
        save_cad_instance_seg = False
        if save_cad_instance_seg:
            instance_pointnum = batch['instance_pointnum'].cpu().numpy()
            instance_id_old2new = np.cumsum(gt_valid_mask) -1  # old_instance_id[instance_id_old2new] = new_instance_id
            N = instance_labels.shape[0]
            valid_instance_id = [i for i in range(N) if instance_labels[i]!= -100]
            non_cad_instance_id = [i for i in valid_instance_id if gt_valid_mask[instance_labels[i]] == 0]
            cad_instance_id = [i for i in valid_instance_id if gt_valid_mask[instance_labels[i]] == 1]

            instance_labels[non_cad_instance_id] = -100
            # print(np.unique(instance_labels[cad_instance_id]))
            instance_labels[cad_instance_id] = instance_id_old2new[np.array(instance_labels[cad_instance_id])]
            # print(np.unique(instance_labels[cad_instance_id]))
            # print("------------------------------")
            print(np.unique(instance_labels),np.unique(instance_labels).shape[0])
            # np.save(os.path.join('datasets/gt_seg', test_scene_name + '.npy'), instance_labels)
            instance_pointnum = instance_pointnum[gt_valid_mask.astype(np.bool_())]
            instance_semantic_labels = instance_semantic_labels[gt_valid_mask.astype(np.bool_())]
            np.savez(os.path.join('datasets/gt_seg', test_scene_name + '.npz'),
                     instance_labels = instance_labels, instance_pointnum=instance_pointnum,instance_semantic_labels=instance_semantic_labels)
            print(instance_pointnum,instance_pointnum.shape[0])
            # print("------------------------------")


        '''save mesh'''
        save_mesh = False
        if save_mesh:
            cnt = 0
            for idx, mesh in enumerate(gt_meshes):
                # scene0568_00_1_table.ply
                #if gt_valid_mask[idx]:
                if mesh is not None:
                    label = gt_labels[idx]
                    out_file = os.path.join('datasets/gt_meshes', f'{test_scene_name}_{cnt}_{CAD_labels[label]}_1.ply')
                    mesh.export(out_file)
                    ##############
                    print('mesh saved at',out_file)
                    ############
                    cnt += 1


        save_gt_instance_partial_pc = False
        if save_gt_instance_partial_pc:
            from plyfile import PlyData, PlyElement
            coords_float = batch['locs_float'].cpu().numpy()
            cnt = 0
            for idx, mesh in enumerate(gt_meshes):
                # scene0568_00_1_table.ply
                # if gt_valid_mask[idx]:
                if mesh is not None:
                    label = gt_labels[idx]
                    out_file = os.path.join('datasets/gt_partial_pc/partial_pc', f'{test_scene_name}_{cnt}_{CAD_labels[label]}_1.ply')

                    pc = coords_float[instance_labels==idx]

                    vertices = np.empty(pc.shape[0],dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                    vertices['x'] = pc[:, 0].astype('f4')
                    vertices['y'] = pc[:, 1].astype('f4')
                    vertices['z'] = pc[:, 2].astype('f4')
                    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
                    ply.write(out_file)

                    ##############
                    print('partial pc saved at', out_file)
                    ############
                    cnt += 1


