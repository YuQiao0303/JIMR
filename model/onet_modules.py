import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim
from torch import autograd
import numpy as np
import trimesh
import mcubes
# from lib.libsimplify import simplify_mesh
from lib.libmise import MISE
import time
import datetime #added by Qiao
from torch import distributed, nn

# -------------------------ONet decoder and generator --------------
class DecoderCBatchNorm(nn.Module): # Onet decoder
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of CResNet blocks.
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
        #added by Qiao
        multires: L for position encoding (log2 of max freq for positional encoding (3D location))
        i_embed: set 0 for default positional encoding, -1 for none
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, legacy=False, multires=10,i_embed=-1):
        super().__init__()
        self.z_dim = z_dim
        self.multires = multires
        self.i_embed = i_embed
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        if self.i_embed == -1:
            # -------------- original --------------
            self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        else:
            # --------------added by Qiao --------------
            embed_fn, input_ch = get_embedder(self.multires, self.i_embed)
            self.fc_p_encoding = nn.Conv1d(input_ch, hidden_size,1)  # deal with points
            # print(input_ch)
            # -----------------------------------------

        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)
        ])

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, p, z, c):

        block_result_list = []
        return_values = ()
        if self.i_embed == -1:
            # -------------- original -------------
            # print("*****original decoder********")
            # print("in",p.shape)
            p = p.transpose(1, 2)
            net = self.fc_p(p)
            # print("out",net.shape)
        else:
            # -------------- modified by Qiao -------------
            # print("*******modified decoder*********")
            # print("in",p.shape)
            embed_fn, input_ch = get_embedder(self.multires, self.i_embed)
            embedded = embed_fn(p)
            embedded = embedded.transpose(1, 2)


            # print("*******decoder*********")
            # print(p.shape)
            # print(embedded.shape)
            # print(self.fc_p_encoding(embedded).shape)
            # print("****************")
            net = self.fc_p_encoding(embedded)
            # print("out",net.shape)
        # ---------------------------------------------
        return_values = return_values + (net,)

        # if self.z_dim != 0:
        #     net = net + self.fc_z(z).unsqueeze(2)

        for block in self.blocks:
            net = block(net, c)
            return_values = return_values + (net,)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return_values = return_values + (out,)

        return return_values #return_values # out


class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 use_cls_for_completion=False,
                 simplify_nfaces=None,
                 preprocessor=None,no_grad = True):
        self.model = model
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor
        self.use_cls_for_completion = use_cls_for_completion
        self.no_grad=True

    def generate_mesh(self, object_features, cls_codes, teacher=False):
        ''' Generates the output mesh.

        Args:
            object_features (tensor): data tensor
            cls_codes (tensor): class one-hot codes.
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = object_features.device
        batch_size = object_features.size(0)
        kwargs = {}

        if self.model.use_cls_for_completion:
            object_features = torch.cat([object_features, cls_codes], dim=-1)

        meshes = []
        # print("genertor,embedding:",object_features)
        # print("genertor,embedding:",object_features.shape)

        for batch_id in range(batch_size):
            # z = self.model.get_z_from_prior((1,), sample=self.sample, device=device)
            z = None
            # if batch_id ==0 :
            #     print('generator.py z',z)
            mesh = self.generate_from_latent(z, object_features[[batch_id]], device, teacher=teacher,**kwargs)
            meshes.append(mesh)
            # break # remember to comment added by Qiao
        return meshes

    # def get_grid_values(self, object_features, cls_codes, return_stats=True):
    #     ''' Generates the output grid values to create mesh.
    #
    #             Args:
    #                 object_features (tensor): data tensor
    #                 cls_codes (tensor): class one-hot codes.
    #                 return_stats (bool): whether stats should be returned
    #             '''
    #     self.model.eval()
    #     device = object_features.device
    #     batch_size = object_features.size(0)
    #     kwargs = {}
    #
    #     if self.model.use_cls_for_completion:
    #         object_features = torch.cat([object_features, cls_codes], dim=-1)
    #
    #     value_grids = []
    #     value_grids = torch.zeros((batch_size, self.resolution0,self.resolution0,self.resolution0)).to(object_features.device)
    #     for batch_id in range(batch_size):
    #         z = self.model.get_z_from_prior((1,), sample=self.sample, device=device)
    #         value_grid = self.get_grids_value_from_latent(z, object_features[[batch_id]], device, **kwargs)
    #         # value_grids.append(value_grid)
    #         value_grids[batch_id,:] = value_grid
    #
    #     # print(value_grids.shape)
    #
    #     return value_grids


    def generate_from_latent(self, z, c=None, device='cuda',  teacher=False, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        # print("generator, teacher",teacher)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, z, c, device, teacher=teacher, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
            # print(value_grid)
            # print('value_grid.shape', value_grid.shape)
            # from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, vis_np_histogram,vis_occ_hat_voxel_vtk  # visualization
            # vis_occ_hat_voxel_vtk(file=None, data=value_grid, all=True)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, z, c, device,teacher=teacher,  **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()
            # from visualization_utils import vis_actors_vtk, get_pc_actor_vtk, vis_np_histogram, \
            #     vis_occ_hat_voxel_vtk  # visualization
            # vis_occ_hat_voxel_vtk(file=None, data=value_grid, all=True)



        # Extract mesh
        mesh = self.extract_mesh(value_grid, z, c)
        return mesh

    # def get_grids_value_from_latent(self, z, c=None, device='cuda',  **kwargs):
    #     ''' Generates mesh from latent.
    #
    #     Args:
    #         z (tensor): latent code z
    #         c (tensor): latent conditioned code c
    #     '''
    #     threshold = np.log(self.threshold) - np.log(1. - self.threshold)
    #
    #     # Compute bounding box size
    #     box_size = 1 + self.padding
    #
    #     # Shortcut
    #     if self.upsampling_steps == 0:
    #         nx = self.resolution0
    #         pointsf = box_size * make_3d_grid(
    #             (-0.5,)*3, (0.5,)*3, (nx,)*3
    #         )
    #         values = self.eval_points(pointsf, z, c, device, **kwargs).cpu().numpy()
    #         value_grid = values.reshape(nx, nx, nx)
    #         value_grid = torch.tensor(value_grid)
    #     else:
    #         mesh_extractor = MISE(
    #             self.resolution0, self.upsampling_steps, threshold)
    #
    #         points = mesh_extractor.query()
    #
    #         while points.shape[0] != 0:
    #             # Query points
    #             pointsf = torch.FloatTensor(points).to(device)
    #             # Normalize to bounding box
    #             pointsf = pointsf / mesh_extractor.resolution
    #             pointsf = box_size * (pointsf - 0.5)
    #             # Evaluate model and update
    #             values = self.eval_points(
    #                 pointsf, z, c, device, **kwargs).cpu().numpy()
    #             values = values.astype(np.float64)
    #             mesh_extractor.update(points, values)
    #             points = mesh_extractor.query()
    #
    #         value_grid = mesh_extractor.to_dense()
    #         value_grid = torch.tensor(value_grid)
    #         # print("*******************")
    #         # print('type(value_grid)',type(value_grid))
    #         # print("*******************")
    #
    #     return value_grid
    def eval_points(self, p, z, c=None, device='cuda', teacher = False,**kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(device)
            if self.no_grad:
                with torch.no_grad():
                    # print('generator,eval_points, embedding',c)
                    occ_hat = self.model.decode(pi, z, c,  teacher=teacher,**kwargs)[0].logits
                    # print('generator,eval_points, occ_hat',occ_hat)
                    # return
            else:
                occ_hat = self.model.decode(pi, z, c,  teacher=teacher, **kwargs)[0].logits

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, z, c=None):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        # print("*************************************")
        # np.set_printoptions(threshold=np.inf)
        # print("occ_hat",occ_hat)
        # print("occ_hat_padded",occ_hat_padded)
        # print("occ_hat_shape", occ_hat.shape)
        # print("occ_hat_max", np.max(occ_hat))
        # print("occ_hat_min", np.min(occ_hat))

        # print("occ_hat_padded_shape", occ_hat_padded.shape)
        # print("occ_hat_padded_max", np.max(occ_hat_padded))
        # print("occ_hat_padded_min", np.min(occ_hat_padded))
        # np.savez(str(datetime.datetime.now()).replace(":","-") +"_occ_hat_epoch10.npz", occ_hat=occ_hat)
        # np.savez("_occ_hat9.npz", occ_hat=occ_hat)
        # print("*************************************")
        vertices, triangles = mcubes.marching_cubes(
            occ_hat_padded, threshold)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            normals = self.estimate_normals(vertices, z, c)

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)

        # Refine mesh
        if self.refinement_step > 0:
            self.refine_mesh(mesh, occ_hat, z, c)

        return mesh

    def estimate_normals(self, vertices, z, c=None, device='cuda'):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        z, c = z.unsqueeze(0), c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, z, c)[0].logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, z, c=None, device='cuda'):
        ''' Refines the predicted mesh.

        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), z, c)[0].logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh

# ------------------------ONet teacher Encoder-----------------------------
class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim # added by qiao

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()
        net = self.fc_pos(p)
        net = self.block_0(net)

        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))
        return c


#-------------------------modules used in ONet decoder and teacher encoder----------------------
class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU(inplace=True)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        # self.actvn = nn.ReLU(inplace=True)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out
#-------------------------position encoding-----------------------------

class Embedder:
    '''
    position encoding from NeRF
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)  # lambda input1, input2, ..., input n: output
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']  # multires-1
        N_freqs = self.kwargs['num_freqs']  # multires

        if self.kwargs['log_sampling']:  # torch.linspace: start, end(include), how many points
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # 2^0, 2^1, 2^2, ..., 2^max_freq
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)  # 2^0,  linear divide  ,2^max_freq,

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:  # sin and cos
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))  # sin(x*2^k) cos(x*2^k)
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    '''
    position encoding from NeRF
    '''
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

# --------------------- other ------------------------------
def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def cosinematrix(A):
    prod = torch.mm(A, A.t())  # 分子
    norm = torch.norm(A, p=2, dim=1).unsqueeze(0)  # 分母
    cos = prod.div(torch.mm(norm.t(), norm))
    return cos