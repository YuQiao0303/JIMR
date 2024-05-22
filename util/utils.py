import torch, glob, os, numpy as np
from plyfile import PlyData, PlyElement
import sys
sys.path.append('../')


from util.log import logger
from util.bbox import BBoxUtils
BBox = BBoxUtils()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch / step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_restore(model, exp_path, exp_name, use_cuda=True, epoch=0, dist=False, f=''):
    if use_cuda:
        model.cpu()
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
            assert os.path.isfile(f), f
        else:
            # f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            f = sorted(glob.glob(os.path.join(exp_path.replace('[','?').replace(']','?'), exp_name.replace('[','?').replace(']','?') + '-*.pth')))
          
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    if len(f) > 0:
        logger.info('Restore from ' + f)
        checkpoint = torch.load(f)
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint)
        else:
            ### tmp
            checkpoint = {k:v for k,v in checkpoint.items() if 'z_linear.6.' not in k} # added by Qiao
            model.load_state_dict(checkpoint, strict=False)

    if use_cuda:
        model.cuda()
    return epoch + 1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def checkpoint_save(model, exp_path, exp_name, epoch, save_freq=16, use_cuda=True):
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    logger.info('Saving ' + f)
    model.cpu()
    torch.save(model.state_dict(), f)
    if use_cuda:
        model.cuda()

    #remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_model_param(model, pretrained_dict, prefix=""):
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: "0.conv.weight"     pretrain_dict: "FC_layer.0.conv.weight"
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {k[len_prefix:]: v for k, v in pretrained_dict.items() if k[len_prefix:] in model_dict
                              and prefix in k
                              and 'z_linear.6' not in k}
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def print_error(message, user_fault=False):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    if user_fault:
      sys.exit(2)
    sys.exit(-1)
# added by Qiao
def get_rotation_matrix (theta, axis):
    if axis == 'x':
        rotation_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        rotation_matrix = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        rotation_matrix = np.array(
            [[np.cos(theta), np.sin(-theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]])
    else:
        print('wrong axis:',axis)
        rotation_matrix = 'error'
    return rotation_matrix
def mesh_fit_scan(mesh,bbox):
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
    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    points = points.dot(axis_rectified)

    # translate
    points = points + bbox[0:3]

    mesh.vertices = points

    return mesh

def pc_fit_scan(points,bbox):

    # swap axes
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    points = points.dot(transform_m.T)

    # recenter + rescale
    min_xyz = points.min(0)
    max_xyz = points.max(0)
    points = points - (max_xyz + min_xyz) / 2 # scale happens in world coordiante system
    points = points / (max_xyz - min_xyz) * bbox[3:6]

    # rotate
    orientation = bbox[6]
    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    points = points.dot(axis_rectified)

    # translate
    points = points + bbox[0:3]
    return points

def pc_fit2standard(points,bbox):
    # translate: move to origin
    points = points -bbox[0:3]
    # rotate
    orientation = - bbox[6]
    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    points = points.dot(axis_rectified)

    # rescale
    # min_xyz = points.min(0)
    # max_xyz = points.max(0)
    # lengh_xyz = max_xyz - min_xyz
    scale = np.max(bbox[3:6])
    # # points = points - (max_xyz + min_xyz) / 2
    # # points = points / lengh_xyz.max(0) # choose between theses2
    points /= scale  # choose between these 2

    # swap axes
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    points = points.dot(transform_m)

    # print('np.max(points)',np.max(points))
    # print('np.min(points)',np.min(points))

    # print(scale,np.max(lengh_xyz))
    return points

def pc_fit2standard_torch(points, bbox):
    # this one is strcitly tested
    # translate: move to origin
    device = points.device
    points = points - bbox[0:3]
    # rotate
    orientation =  -bbox[6].item()
    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    axis_rectified = torch.tensor(axis_rectified).to(device).float()
    points = torch.matmul(points,axis_rectified)
    #
    # rescale
    min_xyz = torch.min(points,0)[0]
    # print(min_xyz)
    max_xyz =  torch.max(points,0)[0]
    lengh_xyz = max_xyz - min_xyz
    scale =  torch.max(bbox[3:6]) #lengh_xyz
    # # points = points - (max_xyz + min_xyz) / 2
    # # points = points / lengh_xyz.max(0) # choose between theses2
    points /= scale  # choose between these 2

    # swap axes : xyz, -z,-x,y
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    transform_m = torch.tensor(transform_m).to(device).float()
    points = torch.matmul(points,transform_m)

    # print(scale, torch.max(lengh_xyz))

    return points

# pcn
def pc_cano2pcn_torch(points):
    device = points.device
    # rotate
    orientation = 0.5 * np.pi
    axis_rectified = get_rotation_matrix(orientation, 'y')
    axis_rectified = torch.tensor(axis_rectified).to(device).float()
    points = torch.matmul(points, axis_rectified)

    return points

def pc_pcn2cano(points):
    '''
    :param points: torch.float, [shape_num, point_num_per_shape, 3]
    :return:
    '''
    device = points.device
    # rotate
    orientation = -0.5 * np.pi
    axis_rectified = get_rotation_matrix(orientation, 'y')
    axis_rectified = torch.tensor(axis_rectified).to(device).float()
    points = torch.matmul(points, axis_rectified)

    # rescale
    min_xyz = torch.min(points, 1)[0]

    max_xyz = torch.max(points, 1)[0]
    lengh_xyz = max_xyz - min_xyz
    scale = torch.max(lengh_xyz,1)[0]   #
    # print('min_xyz.shape',min_xyz.shape)
    # print('max_xyz.shape',max_xyz.shape)
    # print('lengh_xyz.shape',lengh_xyz.shape)
    # print('scale.shape',scale.shape)
    # print('points.shape',points.shape)
    #
    # print('---------------------')
    points = points.transpose(0,2)
    points /= scale  
    points = points.transpose(0, 2)
    return points


from matplotlib import pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
def visualize_voxels(voxels, out_file=None, show=False):
    '''
    Visualizes voxel data.
    :param voxels (tensor): voxel data
    :param out_file (string): output file
    :param show (bool): whether the plot should be shown
    :return:
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)
def get_box_corners(center, vectors):
    '''
    Convert box center and vectors to the corner-form
    :param center:
    :param vectors:
    :return: corner points related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

    corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
    corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    return corner_pnts



# save pc xyz
def export_pc_xyz(pc,path):
    vertices = np.empty(pc.shape[0],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertices['x'] = pc[:, 0].astype('f4')
    vertices['y'] = pc[:, 1].astype('f4')
    vertices['z'] = pc[:, 2].astype('f4')
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(path)



# bbox parameters
def standard_7_dim_bbox_to_bbox_surface_dist(points, bbox7s):
    bbox7s = bbox7s.repeat((points.shape[0],1))
    # print('in bbox7s',bbox7s,bbox7s.shape)
    centers = bbox7s[:, :3]
    x_scales = bbox7s[:,3]
    y_scales = bbox7s[:,4]
    z_scales = bbox7s[:,5]
    thetas = bbox7s[:,6]


    # move to oritin
    points = points - centers

    # rotate -theta around z
    cos_ = torch.cos(-thetas)
    sin_ = torch.sin(-thetas)
    points[:, 0], points[:, 1] = points[:, 0] * cos_ - points[:, 1] * sin_,\
     points[:, 0] * sin_ + points[:, 1] * cos_

    # get dist
    x_dist1 = points[:,0] - (-x_scales/2)
    x_dist2 = x_scales/2 - points[:,0]
    y_dist1 = points[:,1] -  (-y_scales/2)
    y_dist2 = y_scales/2 - points[:,1]
    z_dist1 = points[:, 2] - (-z_scales / 2)
    z_dist2 = z_scales / 2 - points[:, 2]
    bbox_dists = torch.stack([x_dist1, x_dist2, y_dist1, y_dist2, z_dist1, z_dist2], dim=1)

    return  bbox_dists

def _bbox_pred_to_bbox( points, bbox_pred,yaw_parametrization ='Mobius', fix = False):
    if bbox_pred.shape[0] == 0:
        return bbox_pred

    x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
    y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
    z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

    # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
    base_bbox = torch.stack([
        x_center,
        y_center,
        z_center,
        bbox_pred[:, 0] + bbox_pred[:, 1],
        bbox_pred[:, 2] + bbox_pred[:, 3],
        bbox_pred[:, 4] + bbox_pred[:, 5],
    ], -1)

    if bbox_pred.shape[1] == 6:
        return base_bbox

    if yaw_parametrization == 'naive':
        if fix:
            # recalculate points centers:
            # 1.rotate P by -theta; 2. minus v0 and v1 will get rotated corners (leftdown and uptop),
            # 3. rotate back will get the corners; 4. average the corners gives the center
            # do some math and the result will be : P + 0.5 (rotated v1-v0)
            theta = bbox_pred[:,6]
            cos_ = torch.cos(theta)
            sin_ = torch.sin(theta)


            v0 = bbox_pred[:, [0,2]]
            v1 = bbox_pred[:, [1, 3]]
            v1_0 = v1-v0
            v1_0[:,0], v1_0[:,1] = v1_0[:, 0] * cos_ - v1_0[:, 1] * sin_,\
                                   v1_0[:, 0] * sin_ + v1_0[:, 1] * cos_,

            base_bbox[:, 0] = points[:,0] + v1_0[:,0]/2
            base_bbox[:, 1] = points[:,1] + v1_0[:,1]/2


        return torch.cat((
            base_bbox,
            bbox_pred[:, 6:7]
        ), -1)
    elif yaw_parametrization == 'sin_cos':
        # ..., sin(a), cos(a)
        norm = torch.pow(torch.pow(bbox_pred[:, 6:7], 2) + torch.pow(bbox_pred[:, 7:8], 2), 0.5)
        sin = bbox_pred[:, 6:7] / norm
        cos = bbox_pred[:, 7:8] / norm
        angle = torch.atan2(sin, cos)

        if fix:
            # recalculate points centers:
            # 1.rotate P by -theta; 2. minus v0 and v1 will get rotated corners (leftdown and uptop),
            # 3. rotate back will get the corners; 4. average the corners gives the center
            # do some math and the result will be : P + 0.5 (rotated v1-v0)
            theta = angle
            cos_ = torch.cos(theta)
            sin_ = torch.sin(theta)

            v0 = bbox_pred[:, [0, 2]]
            v1 = bbox_pred[:, [1, 3]]
            v1_0 = v1 - v0
            v1_0[:, 0], v1_0[:, 1] = v1_0[:, 0] * cos_ - v1_0[:, 1] * sin_, \
                                     v1_0[:, 0] * sin_ + v1_0[:, 1] * cos_,

            base_bbox[:, 0] = points[:, 0] + v1_0[:, 0] / 2
            base_bbox[:, 1] = points[:, 1] + v1_0[:, 1] / 2

        return torch.cat((
            base_bbox,
            angle
        ), -1)
    elif yaw_parametrization == 'Mobius':
        # ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7]) # the possible range of angle alpha is -pi/2, pi/2

        return torch.stack((
            x_center,
            y_center,
            z_center,
            scale / (1 + q),
            scale / (1 + q) * q,
            bbox_pred[:, 5] + bbox_pred[:, 4],
            alpha
        ), dim=-1)
    elif yaw_parametrization == 'bin':
        # torch.autograd.set_detect_anomaly(True)
        angles = bbox_pred[:,6:]
        points_angles_label = angles[:, :12]  # [sumNPoint, 12]
        points_angles_residual = angles[:, 12:]  # [sumNPoint, 12]

        points_angles_label = torch.softmax(points_angles_label, dim=1) # [sumNPoint,12]

        # decode angles
        points_angles_label = torch.argmax(points_angles_label, dim=1)  # [sumNPoint, ] long
        points_angles_residual = torch.gather(points_angles_residual * np.pi / 12, 1,
                                                     points_angles_label.unsqueeze(1)).squeeze(1)

        points_angles = BBox.class2angle_cuda(points_angles_label, points_angles_residual)#.detach()
        points_angles = points_angles #.unsqueeze(1)

        if fix:
            # recalculate points centers:
            # 1.rotate P by -theta; 2. minus v0 and v1 will get rotated corners (leftdown and uptop),
            # 3. rotate back will get the corners; 4. average the corners gives the center
            # do some math and the result will be : P + 0.5 (rotated v1-v0)
            theta = points_angles
            cos_ = torch.cos(theta)
            sin_ = torch.sin(theta)

            v0 = bbox_pred[:, [0, 2]]
            v1 = bbox_pred[:, [1, 3]]
            v1_0 = v1 - v0
            v1_1_0_ = v1 - v0
            v1_1_0_[:, 0], v1_1_0_[:, 1] = v1_0[:, 0] * cos_ - v1_0[:, 1] * sin_, \
                                     v1_0[:, 0] * sin_ + v1_0[:, 1] * cos_,

            base_bbox[:, 0] = points[:, 0] + v1_1_0_[:, 0] / 2
            base_bbox[:, 1] = points[:, 1] + v1_1_0_[:, 1] / 2

        return torch.cat((
                    base_bbox,
                    # points_angles
                    points_angles.unsqueeze(1)
                ), -1)