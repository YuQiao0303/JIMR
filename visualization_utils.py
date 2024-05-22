import sys;
if sys.platform=="linux":
    sys.path.append("/home/yuqiao/RfDNet/")
    sys.path.append("/home/yuqiao/RfDNet/windows")

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import open3d as o3d
import datetime


import matplotlib.pyplot as plt
import vtk
# from utils import pc_util
# from prepare_refine_data import get_shapenet_model_by_scene

from glob import glob
import pickle

import trimesh
from plyfile import PlyData, PlyElement

transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
transform_GTtxt2ply = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
CLS_STRS = [ "table", "chair", "bookshelf","sofa", "trash_bin", "cabinet", "display","bathtub"]



# ----------------vis grid occupancy values  ----------------#

def vis_occ_hat_voxel_vtk(file= r"out\iscnet\2022.05.16.15.14_SA_10_epochs_occ_hat\occ_hat_epoch0.npz",
                          data=None,
                          all=False):

    if file == None or file == "":
        try:
            occ_hat = data.detach().cpu().numpy() # np.ndarray [res,res,res]
        except:
            occ_hat = data
    else:
        data = np.load(file)
        occ_hat = data['occ_hat'] # np.ndarray [res,res,res], res = 32


    if not all:
        points = np.argwhere(occ_hat > 0)
        color = (1,0,0)
        alpha = 0.2
        pc_actor = get_pc_actor_vtk(points, color, alpha,point_size=10)
        vis_actors_vtk([pc_actor])

    else:
        # modify this as needed
        #b
        thred1 = -22
        thred2 = -7
        thred3 = 0

        #after sigmoid
        thred1 = 0.05
        thred2 = 0.1
        thred3 = 0.5
        colorVeryOut = ([0.1, 0.1, 0.5])  # blue
        colorMediumOut = ([0.1, 0.5, 0.1])  # green
        colorALittleOut = ([0.7, 0.5, 0.0])  # yellow
        colorIn = ([1, 0, 0])  # red
        alpha = 0.05
        pointsVeryOut = np.argwhere(occ_hat< thred1)
        pointsMediumOut = np.argwhere((occ_hat> thred1)&(occ_hat< thred2))
        pointsALittleOut = np.argwhere((occ_hat> thred2)&(occ_hat< thred3))
        pointsIn = np.argwhere(occ_hat> thred3)
        point_size = 5
        pc_actors = [get_pc_actor_vtk(pointsVeryOut, colorVeryOut, 0.05, 1),
                     get_pc_actor_vtk(pointsMediumOut, colorMediumOut, 0.1,7),
                     get_pc_actor_vtk(pointsALittleOut, colorALittleOut, 0.1,7),
                     get_pc_actor_vtk(pointsIn, colorIn, 0.1,10),]
        vis_actors_vtk(pc_actors)



def vis_np_histogram(all_data,thresh = 100):
    import matplotlib
    matplotlib.use('TkAgg')
    # thresh = 1.0 # 2500
    # all_data[0] [all_data[0] > thresh] = thresh
    # all_data[0] [all_data[0] < -thresh] = -thresh
    #
    # all_data[1] [all_data[1] > thresh] = thresh
    # all_data[1] [all_data[1] < -thresh] = -thresh

    # print(2, matplotlib.get_backend())
    title = os.path.basename(" histogram")
    labels = ['AE','pipeline']
    plt.title(title)

    plt.hist(all_data, bins=20, edgecolor="r", histtype="bar", alpha=0.5, label=labels,density=True)
    plt.legend()

    # ticks = np.arange(int(min), int(max), (int(max) - int(min)) / 20).tolist()
    # ticks.append(0)
    # ticks.append(max)
    #
    # plt.xticks(ticks)
    plt.show()

def vis_multi_occ_hat_histogram(files,positiveOnly = False):
    file_num = len(files)
    # all_data = np.zeros((file_num,32*32*32))
    all_data = []
    labels = []
    min = 0
    max=0
    for i,file in enumerate(files):
        print("%d/%d"%((i+1),file_num))
        data = np.load(file)
        occ_data = data['occ_hat'].flatten()
        if positiveOnly:
            occ_data = occ_data[occ_data>0]
        if min > np.min(occ_data):
            min = np.min(occ_data)
        if max < np.max(occ_data):
            max = np.max(occ_data)
        # occ_hat = data['occ_hat'].flatten()
        # all_data[i,:] = data['occ_hat'].flatten()

        all_data.append(occ_data)
        labels.append(os.path.basename(file))
    title = os.path.basename("occ histogram")
    print(labels)
    plt.title(title)

    plt.hist(all_data, bins=20, edgecolor="r", histtype="bar", alpha=0.5, label=labels)
    plt.legend()

    ticks = np.arange(int(min), int(max), (int(max) - int(min)) / 20).tolist()
    ticks.append(0)
    ticks.append(max)

    plt.xticks(ticks)
    plt.show()



# ----------------   vtk_utils   ----------------
def vis_actors_vtk(actors,parallel=False):
    '''set renderer'''
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # 设置背景颜色

    # Renderer Window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.FullScreenOn()  # 相当于SetFullScreen(True)
    window.BordersOn()  # 相当于SetBorders(True)

    # System Event
    win_render = vtk.vtkRenderWindowInteractor()
    win_render.SetRenderWindow(window)

    # Style
    interactor_camera = vtk.vtkInteractorStyleMultiTouchCamera()
    # interactor_camera = vtk.vtkInteractorStyleImage() # cannot rotate
    # interactor_camera = vtk.vtkInteractorStyleRubberBand3D() # cannot rotate,can draw a rectangle
    # print(interactor_camera.GetAutoAdjustCameraClippingRange()) #1
    interactor_camera.AutoAdjustCameraClippingRangeOff()
    # interactor_camera.SetClippingRange(0, 1000000)  # default is (0.1,1000) # not working
    win_render.SetInteractorStyle(interactor_camera) # what if we comment it here
    # win_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

    if parallel:
        '''set camera''' # this part is new added
        renderer.ResetCamera()
        camera = vtk.vtkCamera()
        # cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        # camera_center = np.array([0, -3, 3])
        # centroid = camera_center
        # camera = set_camera(centroid, [[0., 0., 0.], [-centroid[0], -centroid[1],
        #                                                    centroid[0] ** 2 / centroid[2] + centroid[1] ** 2 / centroid[
        #                                                        2]]], cam_K)
        camera.ParallelProjectionOn()
        # camera.SetClippingRange(0,1000000) # default is (0.1,1000) # not working
        # camera.SetObliqueAngles(0,100) # (45,90) by default. # not working too
        # camera.SetViewAngle(90) # 30 by default
        renderer.SetActiveCamera(camera)

    # Insert Actor
    for actor in actors:
        renderer.AddActor(actor)
    # add axis actor
    axes = vtk.vtkAxesActor()
    renderer.AddActor(axes)
    # visaulize
    win_render.Initialize()
    win_render.Start()
def get_pc_actor_vtk(pc_np,color = (0,0,1),alpha = 0.3,point_size = 3,
                     normalize=False,bbox = None,bobox_rotation_only=False,
                     use_translation=False,translation=None):
    if isinstance(pc_np,str):

        if os.path.exists(pc_np):
            pc_np = read_ply(pc_np)
        else:
            # print('wrong pc input:',pc_np)
            pc_np = np.zeros((1,3))
            alpha = 0
    obj_points = pc_np
    if bbox is not None:
        # print('bbox is not None', bbox)
        if bobox_rotation_only:
            obj_points = pc_fit2standard_rotation_only(obj_points, bbox)
        else:
            obj_points = pc_fit2standard(obj_points, bbox)

    if normalize and (bbox is None):
        # get loc and scale for normalization
        mesh = trimesh.load(meshfile, process=False)

        # # swap axes
        transform_GTtxt2ply = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        obj_points = obj_points.dot(transform_GTtxt2ply)
        mesh.vertices = obj_points
        # get loc and scale
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2  # mesh.apply_translation(-loc)
        scale = (bbox[1] - bbox[0]).max()  # mesh.apply_scale(1 / scale)
        # move to origin
        # obj_points = obj_points - (obj_points.max(0) + obj_points.min(0)) / 2.
        obj_points = obj_points - loc
        # scale
        obj_points = obj_points / scale


    if use_translation:
        # # get points from object
        # rotate
        # rotate through z axis, angle: theta  GTorientation - proposal_orientation
        theta = np.pi / 9  #
        obj_points = obj_points.dot(get_rotation_matrix(theta + np.pi, 'y'))  # rotate through axis by  theta
        # print(get_rotation_matrix(theta,'x'))
        # print(get_rotation_matrix(theta,'y'))
        obj_points = obj_points.dot(get_rotation_matrix(-theta, 'x'))  # rotate through axis by  theta

        # move
        obj_points = obj_points + np.array(translation)


    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    # 导入点数据
    points.SetData(numpy_to_vtk(obj_points))
    # 新建 vtkPolyData 实例
    polydata = vtk.vtkPolyData()
    # 设置点坐标
    polydata.SetPoints(points)

    # 顶点相关的 filter
    vertex = vtk.vtkVertexGlyphFilter()
    vertex.SetInputData(polydata)

    # mapper 实例
    mapper = vtk.vtkPolyDataMapper()
    # 关联 filter 输出
    mapper.SetInputConnection(vertex.GetOutputPort())

    # actor 实例
    actor = vtk.vtkActor()
    # 关联 mapper
    actor.SetMapper(mapper)

    # actor.GetProperty().SetColor(1, 0, 0)  # 设置点颜色
    actor.GetProperty().SetColor(color)  # 设置点颜色
    actor.GetProperty().SetPointSize(point_size)
    actor.GetProperty().SetOpacity(alpha)

    return actor
def get_colorful_pc_actor_vtk(pc_np,point_colors = None,point_size = 3, opacity = 0.3,palette_name = 'crest_r',light = 1,cut=True):
    pc_np = convert_torch2np(pc_np)
    if point_colors is not None:
        point_colors = convert_torch2np(point_colors)

    centroid = np.array([3, 0, 3])
    def set_points_property( point_clouds, point_colors):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        x3 = point_clouds[:, 0]
        y3 = point_clouds[:, 1]
        z3 = point_clouds[:, 2]

        for x, y, z, c in zip(x3, y3, z3, point_colors):
            id = points.InsertNextPoint([x, y, z])
            colors.InsertNextTuple3(*c)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point
    colors = np.linalg.norm(pc_np - centroid, axis=1)
    colors = depth_palette[np.int16((colors - colors.min()) / (colors.max() - colors.min()) * 99)]

    mapper = vtk.vtkPolyDataMapper()
    if point_colors is None:
        point_colors =255 * colors* light
    if point_colors.max() <=1.0:
        point_colors *=255
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(set_points_property(pc_np, point_colors))
    else:
        mapper.SetInputData(set_points_property(pc_np, point_colors))

    point_actor = vtk.vtkActor()
    point_actor.SetMapper(mapper)

    point_actor.GetProperty().SetPointSize(point_size)
    point_actor.GetProperty().SetOpacity(opacity)
    point_actor.GetProperty().SetInterpolationToPBR()
    return point_actor


def get_text_actor_vtk(str="hihihihihihihihihihi",position = [0,0,0]):
    text_source = vtk.vtkVectorText()
    text_source.SetText(str)
    # text_source.SetTextProperty().SetFontSize(20) # not ok
    polyDataMapper = vtk.vtkPolyDataMapper()
    polyDataMapper.SetInputConnection(text_source.GetOutputPort()) # this is the key
    # actor = vtk.vtkActor()
    actor = vtk.vtkFollower()
    actor.SetMapper(polyDataMapper)
    actor.GetProperty().SetColor(1.0,0,0)
    scale_factor = 0.1
    actor.SetScale(scale_factor,scale_factor,scale_factor)

    offset_factor = 0.05

    actor.SetPosition(position[0], position[1],position[2]);

    return actor
def pc_fit2standard(points,bbox):
    # translate: move to origin

    # print('pc_fit2standard bbox',bbox[0:3])
    # print('pc_fit2standard bbox',bbox[0:3].shape)
    points = points - bbox[0:3]
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


def pc_fit2standard_rotation_only(points,bbox):

    # translate: move to origin

    min_xyz = points.min(0)
    max_xyz = points.max(0)
    # points = points - bbox[0:3]
    points = points - (max_xyz+min_xyz)/2
    # rotate
    orientation = - bbox[6]
    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    points = points.dot(axis_rectified)

    # rescale
    min_xyz = points.min(0)
    max_xyz = points.max(0)
    lengh_xyz = max_xyz - min_xyz
    # scale = np.max(bbox[3:6])
    scale = np.max(lengh_xyz)
    points /= scale  # choose between these 2

    # swap axes
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    points = points.dot(transform_m)

    # print('np.max(points)',np.max(points))
    # print('np.min(points)',np.min(points))

    # print(scale,np.max(lengh_xyz))
    return points
def get_mesh_actor_vtk(meshfile,
                       color = [1.0,0.67,0.6],opacity =1,
                       use_translation = False, translation =(0,0,0),
                       normalize = False,bbox = None, bobox_rotation_only = False):

    '''get data'''
    if isinstance(meshfile,str):
        if ".ply" in meshfile: # proposal mesh
            vtk_object = vtk.vtkPLYReader()
            vtk_object.SetFileName(meshfile)
            vtk_object.Update()
            polydata = vtk_object.GetOutput()

        elif ".obj" in meshfile: # GT mesh
            vtk_object = vtk.vtkOBJReader()
            vtk_object.SetFileName(meshfile)
            vtk_object.Update()
            polydata = vtk_object.GetOutput()

        elif ".off" in meshfile:
            polydata = read_off_vtk(meshfile)
            # print("off", type(vtk_object))
        else:
            print("wrong",meshfile)
    else:
        polydata = read_trimesh_vtk(meshfile)
    if bbox is not None:
        # print('bbox is not None', bbox)
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)

        if bobox_rotation_only:
            obj_points = pc_fit2standard_rotation_only(obj_points,bbox)
        else:
            obj_points = pc_fit2standard(obj_points, bbox)
        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)

    if normalize and (bbox is None):
        # print('normalize')
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
        # get loc and scale for normalization
        mesh = trimesh.load(meshfile, process=False)

        # # swap axes
        transform_GTtxt2ply = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        obj_points = obj_points.dot(transform_GTtxt2ply)
        mesh.vertices = obj_points
        # get loc and scale
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2  # mesh.apply_translation(-loc)
        scale = (bbox[1] - bbox[0]).max()  # mesh.apply_scale(1 / scale)
        # move to origin
        # obj_points = obj_points - (obj_points.max(0) + obj_points.min(0)) / 2.
        obj_points = obj_points - loc
        # scale
        obj_points = obj_points / scale
        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)

    if use_translation:
        # # get points from object
        # polydata = vtk_object.GetOutput()
        # read points using vtk_to_numpy
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
        if 'obj' in meshfile: # need extra normalize (ply are all normalized already)
            # get loc and scale for normalization
            mesh = trimesh.load(meshfile, process=False)
            bbox = mesh.bounding_box.bounds

            loc = (bbox[0] + bbox[1]) / 2 # mesh.apply_translation(-loc)
            scale = (bbox[1] - bbox[0]).max() # mesh.apply_scale(1 / scale)
            # move to origin
            # obj_points = obj_points - (obj_points.max(0) + obj_points.min(0)) / 2.
            obj_points = obj_points - loc
            # scale
            obj_points = obj_points/scale

        # rotae
        # rotate through z axis, angle: theta  GTorientation - proposal_orientation
        theta =  np.pi / 9 #
        obj_points = obj_points.dot(get_rotation_matrix(theta+np.pi ,'y')) # rotate through axis by  theta
        # print(get_rotation_matrix(theta,'x'))
        # print(get_rotation_matrix(theta,'y'))
        obj_points = obj_points.dot(get_rotation_matrix(-theta,'x')) # rotate through axis by  theta

        # move
        obj_points = obj_points + np.array(translation)

        # print(obj_points[:,0].max(0),obj_points[:,0].min(0))
        # print(obj_points[:,1].max(0),obj_points[:,1].min(0))
        # print(obj_points[:,2].max(0),obj_points[:,2].min(0))
        # # obj_points = obj_points.dot(np.diag(1 / (obj_points.max(0) - obj_points.min(0)))).dot(
        # #     np.diag(box['box3D'][3:6]))

        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)
        # vtk_object.Update()


    '''set mapper'''
    mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputConnection(vtk_object.GetOutputPort()) # vtk_object is vtkmodules.vtkCommonDataModel.vtkPolyData

    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    '''set actor'''
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    '''material: doesn't seem helpful'''
    # actor.GetProperty().SetDiffuse(0.1) #
    actor.GetProperty().SetAmbient(0.1)
    # actor.GetProperty().SetSpecular(0.2)

    return actor

def get_bbox_actor_vtk(bboxes,color = [1.0,0.67,0.6],opacity =1):

    bbox_param = bboxes
    centers = bbox_param[:,:3]
    orientation = bbox_param[:,6]
    sizes = bbox_param[:,3:6]

    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    vectors = np.diag(sizes / 2.).dot(axis_rectified)

    print(vectors)
    corners = get_box_corners(center,vectors)


    # center = bbox_param[:3]
    # orientation = bbox_param[6]
    # sizes = bbox_param[3:6]
    #
    # axis_rectified = np.array(
    #     [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    # vectors = np.diag(sizes / 2.).dot(axis_rectified)
    #
    # corners = get_box_corners(center,vectors)


def get_box_corners( center, vectors):
    '''
    Convert box center and vectors to the corner-form
    :param center:
    :param vectors:
    :return: corner points and faces related to the box
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

    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3)]

    return corner_pnts, faces



def vis_pc_vtk(pc_np):
    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    # 导入点数据
    points.SetData(numpy_to_vtk(pc_np))
    # 新建 vtkPolyData 实例
    polydata = vtk.vtkPolyData()
    # 设置点坐标
    polydata.SetPoints(points)

    # 顶点相关的 filter
    vertex = vtk.vtkVertexGlyphFilter()
    vertex.SetInputData(polydata)

    # mapper 实例
    mapper = vtk.vtkPolyDataMapper()
    # 关联 filter 输出
    mapper.SetInputConnection(vertex.GetOutputPort())

    # actor 实例
    actor = vtk.vtkActor()
    # 关联 mapper
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(1, 0, 0) # 设置点颜色
    actor.GetProperty().SetPointSize(3)
    # renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1) # 设置背景颜色

    # Renderer Window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.FullScreenOn()  # 相当于SetFullScreen(True)
    window.BordersOn()  # 相当于SetBorders(True)

    # System Event
    win_render = vtk.vtkRenderWindowInteractor()
    win_render.SetRenderWindow(window)

    # Style
    interactor_camera = vtk.vtkInteractorStyleMultiTouchCamera()
    print(interactor_camera.GetAutoAdjustCameraClippingRange())
    interactor_camera.AutoAdjustCameraClippingRangeOn()

    win_render.SetInteractorStyle(interactor_camera)

    # win_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

    # Insert Actor
    renderer.AddActor(actor)
    win_render.Initialize()
    win_render.Start()


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


def read_trimesh_vtk(mesh):
    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)
    nodes = v
    elements = f

    # Make the building blocks of polyData attributes
    Mesh = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    Cells = vtk.vtkCellArray()

    # Load the point and cell's attributes
    for i in range(len(nodes)):
        Points.InsertPoint(i, nodes[i])

    for i in range(len(elements)):
        Cells.InsertNextCell(mkVtkIdList(elements[i]))

    # Assign pieces to vtkPolyData
    Mesh.SetPoints(Points)
    Mesh.SetPolys(Cells)
    return Mesh
def read_off_vtk(meshfile):
    mesh = trimesh.load(meshfile)
    return read_trimesh_vtk(mesh)
    # v = np.array(mesh.vertices)
    # f = np.array( mesh.faces)
    # nodes = v
    # elements = f
    #
    # # Make the building blocks of polyData attributes
    # Mesh = vtk.vtkPolyData()
    # Points = vtk.vtkPoints()
    # Cells = vtk.vtkCellArray()
    #
    # # Load the point and cell's attributes
    # for i in range(len(nodes)):
    #     Points.InsertPoint(i, nodes[i])
    #
    # for i in range(len(elements)):
    #     Cells.InsertNextCell(mkVtkIdList(elements[i]))
    #
    # # Assign pieces to vtkPolyData
    # Mesh.SetPoints(Points)
    # Mesh.SetPolys(Cells)
    # return Mesh


def mkVtkIdList(it):
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil
def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array
# ---------------- vis scene ----------------
def main_compare_dimr_scenes(mesh = True, pc = False):
    # root_path = '../dimr/datasets/gt_meshes/'
    root_paths = [
        # 'datasets/gt_meshes/',
        # 'datasets/dimr_meshes/',
        # 'exp/scannetv2/rfs/rfs_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # 'exp/scannetv2/rfs/my_testteacher_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',

        # 'exp/scannetv2/rfs/my_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',

        # 'exp/scannetv2/rfs/rfs_phase2_epoch9_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # f'exp/scannetv2/rfs/[2022.09.02]pcn_test_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # f'exp/scannetv2/rfs/[2022.09.05]test_pcn_train_on_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # 'exp/scannetv2/rfs/[2022.09.05]test_pcn_train_on_pretrained_high_thre_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/',
        # 'exp/scannetv2/rfs/[2022.09.05]test_self_trained_pcn_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/',
        # 'exp/scannetv2/rfs/[2022.09.08]test_trained_gt_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # 'exp/scannetv2/rfs/[2022.09.08.19.22]test_trained_gt2_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/',

        # 'exp/scannetv2/rfs/my_phase2_epoch5_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # 'exp/scannetv2/rfs/my_phase2_epoch5_scannet_student/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # '../original_dimr/exp/scannetv2/rfs/rfs_pretrained_0.4_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/',
        # '../dimr/exp/scannetv2/rfs/2022.10.12_test_all_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # '../2dimr/exp/scannetv2/rfs/my_test_phase2_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # '../2dimr/exp/scannetv2/rfs/2022.10.25.11.24_test_phase2_epoch4/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # '../dimr/exp/scannetv2/rfs/2022.10.27.15.45_test_fixbox_all_epoch4/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        # '../dimr/exp/scannetv2/rfs/2022.10.31.15.07_test_nobboxscore_seg_cls_bbox_mesh/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',
        '../dimr/exp/scannetv2/rfs/2022.10.27.3finalscore/2022.10.27.16.50_test_fixbox_all_epoch4/result/epoch256_nmst0.3_scoret0.05_npointt100/val/',


    ]

    colors = [
        # green,
        blue,
        red,
    ]

    # root_path = 'exp/scannetv2/rfs/rfs_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes'
    # root_path = 'exp/scannetv2/rfs/my_pretrai ned_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes'

    # scene_names = ['scene0474_03', 'scene0704_01','scene0377_02','scene0606_02']
    # scene_names = ['scene0423_01']
    scene_names = ['scene0606_02']
    scene_names = [
'scene0606_02',
'scene0046_00',
'scene0207_02',
'scene0700_01',
'scene0552_01',
'scene0377_00',
'scene0377_02',
'scene0025_00',
'scene0334_01',
'scene0423_02',
'scene0203_01',
'scene0591_00',
'scene0406_00',
'scene0193_00'
                   ]

    if len(root_paths)>1:
        opacity = 0.5
    else:
        opacity =1
    for scene_name in scene_names:
        print(scene_name)
        actor_list = []

        for i,root_path in enumerate(root_paths):
            if not os.path.exists(root_path):
                print("root path doesn't exist",root_path)
            else:
                print(root_path)
            root_path = root_path.replace('[','?').replace(']','?')
            # mesh
            if mesh:
                match_str = os.path.join(root_path, 'meshes',scene_name + "*")
                mesh_paths = glob(match_str)
                print(match_str)
                print(mesh_paths)
                for path in mesh_paths:
                    if os.path.exists(path):
                        actor_list.append(get_mesh_actor_vtk(path, opacity=opacity,color=colors[i]))
                    else:
                        print("path doens't exists:", path)
            # pc
            if pc:
                match_str = os.path.join(root_path, 'completed_pc', scene_name + "*")
                completed_pc_paths = glob(match_str)

                for path in completed_pc_paths:
                    if os.path.exists(path):
                        # print(path)
                        pc_np = read_ply(path)
                        actor_list.append(get_pc_actor_vtk(pc_np, alpha=1, color=colors[i]))
                    else:
                        print("path doens't exists:", path)


        vis_actors_vtk(actor_list)

def main_vis_dimr_scenes():
    # test_scene_names = get_test_scene_names()

    # root_path = '../dimr/datasets/gt_meshes/'
    # root_path = 'datasets/gt_meshes/'
    root_path = 'exp/scannetv2/rfs/rfs_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes'
    # root_path = 'exp/scannetv2/rfs/my_pretrai ned_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes'
    scene_name = "scene0474_03"
    # scene_names = ['scene0474_03', 'scene0704_01']
    # scene_names = ['scene0377_02']
    scene_names = ['scene0606_02']
    for scene_name in scene_names:
        print(scene_name)
        match_str = os.path.join(root_path, scene_name + "*")
        mesh_paths = glob(match_str)

        actor_list = []
        for path in mesh_paths:

            if os.path.exists(path):
                actor_list.append(get_mesh_actor_vtk(path, opacity=1))
            else:
                print("path doens't exists:", path)
        vis_actors_vtk(actor_list)


# ------------------vis genearted meshes------------------------------------

def all_target_dict2by_type_dict(all_target_dicts,processed_data_path = 'datasets/scannet/processed_data/'
                             ):
    return_dict = {
        "table":{},
        "chair":{},
        "bookshelf":{},
        "sofa":{},
        "trash_bin":{},
        "cabinet":{},
        "display":{},
        "bathtub":{},
    }
    for dict in all_target_dicts:
        for scene_name, target_id_list in dict.items():
            '''prepare scene GT data'''
            GT_data_file = os.path.join(processed_data_path, scene_name + "/", 'bbox.pkl')
            with open(GT_data_file, 'rb') as file:
                bboxes = pickle.load((file))
            GT_mesh_paths = [''] * len(target_id_list)

            for i, target_id in enumerate(target_id_list):
                '''get GT mesh'''
                if not os.path.exists(GT_mesh_paths[i]):
                    for box_target_id, box in enumerate(bboxes):
                        if box_target_id == target_id:
                            target_cls_str = ShapeNetIDMap[box['shapenet_catid'][1:]]
                            # print(target_cls_str)
                            if scene_name not in return_dict[target_cls_str].keys():
                                return_dict[target_cls_str][scene_name] = []
                            return_dict[target_cls_str][scene_name].append(target_id)
    return return_dict



def vis_all_scene_data_vtk_new(start_id=0, skip=1, selected_object_dict=None,
                               root_paths=["out/iscnet/test_all_1489_scenes/visualization/"],
                               simplified = True,processed_data_path = 'datasets/scannet/processed_data/',
                               tittles = [''],by_row = True):

    root_path = root_paths[0]
    global_GT_mesh_path_dict = {} # {"scene0000_02":[GTpath1,GTpath2,...]}
    global_proposal_mesh_path_dicts = [] # list of dicts
    for i, root_path in enumerate(root_paths):
        global_proposal_mesh_path_dicts.append({})

    for scene_name, target_id_list in selected_object_dict.items():
        '''prepare scene GT data'''
        GT_data_file = os.path.join(processed_data_path, scene_name + "/", 'bbox.pkl')
        with open(GT_data_file, 'rb') as file:
            bboxes = pickle.load((file))
        GT_mesh_paths = ['']*len(target_id_list)

        '''go through all root paths'''
        for root_path_id, root_path in enumerate(root_paths):
            # proposal_mesh_path_dict = {}
            proposal_mesh_paths = []

            for i,target_id in enumerate(target_id_list):
                # print('target_id',target_id)
                # for all strings in proposal_pc_pathes, select one include "target_targetId"
                '''get proposal_mesh'''
                proposal_mesh_path_match_str = os.path.join(root_path, "*" +scene_name + "/",
                                                            'proposal_*_target_%d_*mesh.ply' % target_id)
                # print(proposal_mesh_path_match_str)
                temp = glob(proposal_mesh_path_match_str)
                if (len(temp) == 0):  # this object is not detected
                    proposal_mesh_paths.append("")
                    print("proposal noet found:",proposal_mesh_path_match_str)
                else:
                    proposal_mesh_paths.append(temp[0])
                '''get GT mesh'''
                if not os.path.exists(GT_mesh_paths[i]):
                    for box_target_id, box in enumerate(bboxes):
                        if box_target_id == target_id:
                            target_cls_str = ShapeNetIDMap[box['shapenet_catid'][1:]]
                            box['box3D'][6] = np.mod(box['box3D'][6] + np.pi, 2 * np.pi) - np.pi
                            if simplified:
                                ShapeNetv2_path = "datasets/ShapeNetv2_data/watertight_scaled_simplified/"
                                shapenet_model_path = os.path.join(ShapeNetv2_path, box['shapenet_catid'] + "/",
                                                                   box['shapenet_id'] + '.off')
                            else:
                                ShapeNetv2_path = "datasets/ShapeNetCore.v2/"
                                shapenet_model_path = os.path.join(ShapeNetv2_path, box['shapenet_catid'] + "/",
                                                                   box['shapenet_id'] + "/models/model_normalized.obj")
                            # print(shapenet_model_path,os.path.exists(shapenet_model_path))
                            GT_mesh_paths[i] = shapenet_model_path

            global_proposal_mesh_path_dicts[root_path_id][scene_name]=proposal_mesh_paths
            global_GT_mesh_path_dict[scene_name] = GT_mesh_paths

    '''visualize data'''
    tittles = [*tittles,'GT']
    cat_dicts = [*global_proposal_mesh_path_dicts, global_GT_mesh_path_dict]
    # print(cat_dicts)
    # for scene_name, path_list in global_GT_mesh_path_dict.items():
    #     for path in path_list:
    #         print("'" +path + "',")

    all_mesh_actors = []
    red = [255.0 /255, #1
           171.0 /255, # 0.67
           153.0 /255] # 0.6
    margin = 1.2
    x_offset_factor = 0.05
    for root_path_id, mesh_dict in enumerate(cat_dicts): # same row
        if by_row:
            y_shift = - root_path_id * margin
            x_shift = 0
            offset_factor = 1
            text_position =[x_shift - x_offset_factor * len(tittles[root_path_id]) -1 ,
                            y_shift ,#+ offset_factor,
                            0]

        else:
            x_shift = root_path_id*margin
            y_shift = 0

            text_position = [x_shift - x_offset_factor * len(tittles[root_path_id]),
                             y_shift - 1,
                             0]
        all_mesh_actors.append(get_text_actor_vtk(tittles[root_path_id], text_position))

        for scene_name,pathes in mesh_dict.items():
            # print(scene_name,":",pathes)
            for path in pathes: # same volume
                if os.path.exists(path):
                    pass
                    all_mesh_actors.append(get_mesh_actor_vtk(path, red,1.0,True,(x_shift,y_shift,0)))
                if by_row:
                    x_shift = x_shift + margin
                else:
                    y_shift = y_shift +margin


    vis_actors_vtk(all_mesh_actors,parallel=True)



def rename_files(path):
    scene_path_list = os.listdir(path)
    for i, scene_path in enumerate(scene_path_list):
        if ("scene" in scene_path):
            # print(scene_path)
            new_scene_path = "scene%s" % scene_path.split("scene")[1]
            print(new_scene_path)
            command = "mv %s %s" % (os.path.join(path, scene_path), os.path.join(path, new_scene_path))
            print(command)
            # os.system(command)
            os.rename(os.path.join(path, scene_path), os.path.join(path, new_scene_path))




def get_test_scene_names():
    with open ("../dimr/datasets/splits/test.txt","r", encoding="utf-8") as f:

        # data = f.read()
        data = f.readlines() # list
        # data = f.readline()
        for i in range(len(data)):
            data[i] = data[i].replace("\n","")

        return data

def get_dicts_from_gt_meshes(gt_mesh_root_path='../RfDNet/out/12_scenes/gt_meshes', by = "cls"):
    all_gt_mesn_paths = os.listdir(gt_mesh_root_path)

    if by == 'scene':
        dict_list = []
        file_names = os.listdir(gt_mesh_root_path)
        scene_names = list(set([filename[0:12] for filename in file_names]))
        for scene_id, scene_name in enumerate(scene_names):
            # if '0377_00' not in scene_name:
            #     continue
            gt_mesh_paths = glob(os.path.join(gt_mesh_root_path, scene_name + '*'))
            # print('gt_mesh_paths',gt_mesh_paths)
            gt_ids = [int(os.path.basename(gt_mesh_path)[:-4].split('_')[2])  for gt_mesh_path in gt_mesh_paths]
            print('gt_ids',gt_ids)
            dict_list.append({scene_name:gt_ids})

    elif by =='cls':
        # all cls lists: CLS_STRS
        dict_list = []
        for i, cls in enumerate(CLS_STRS):
            dict_list.append({})
            cls_gt_paths = glob(os.path.join(gt_mesh_root_path,'*'+cls+'*'))
            for gt_mesh_path in cls_gt_paths:
                # scene0025_00_0_display_1.ply
                gt_mesh_path = os.path.basename(gt_mesh_path)
                scene_name = gt_mesh_path[0:12] #
                gt_id = int(gt_mesh_path[13:].split('_')[0])
                # print(scene_name,gt_id)
                if scene_name not in dict_list[i].keys():
                    dict_list[i][scene_name] = []
                dict_list[i][scene_name].append(gt_id)
    return dict_list

def get_dicts_from_scenes(scene_list=None,by = 'cls'):
    if scene_list is None:
        scene_list = ['scene0339_00',
                      'scene0349_01',
                      'scene0345_01',
                      'scene0557_02',
                      'scene0013_02',
                      'scene0047_00',
                      'scene0501_01',
                      'scene0158_02',
                      'scene0419_00',
                      'scene0529_00',
                      'scene0524_00',
                      'scene0613_01',
                      'scene0666_00',
                      'scene0120_00', ]
    gt_mesh_root_path = '../dimr/datasets/gt_meshes/meshes'
    all_gt_mesn_paths = os.listdir(gt_mesh_root_path)

    if by == 'scene':
        dict_list = []
        for gt_mesh_path in all_gt_mesn_paths:
            scene_name = gt_mesh_path[0:12]  # 5+4+1+2-1
            cur_dict = {scene_name: []}

            # first we need to have all scene_names
            # per_scene_gt_mesh_path_list = glob
            # to be continued

    elif by == 'cls':
        # all cls lists: CLS_STRS
        dict_list = []
        for i, cls in enumerate(CLS_STRS):
            dict_list.append({})
            for scene_name in scene_list:
                cls_gt_paths = glob(os.path.join(gt_mesh_root_path, scene_name+'*' + cls + '*'))
                print(os.path.join(gt_mesh_root_path, scene_name+'*' + cls + '*'))
                for gt_mesh_path in cls_gt_paths:
                    # scene0025_00_0_display_1.ply
                    gt_mesh_path = os.path.basename(gt_mesh_path)
                    scene_name = gt_mesh_path[0:12]  #
                    gt_id = int(gt_mesh_path[13:].split('_')[0])
                    # print(scene_name,gt_id)
                    if scene_name not in dict_list[i].keys():
                        dict_list[i][scene_name] = []
                    dict_list[i][scene_name].append(gt_id)
    return dict_list
def new_test_scene_dict(by_type = False):
    dicts = []
    if by_type:
        all_targets_dicts = new_test_scene_dict(by_type=False)

    else:
        dicts.append({"scene0025_00":[0,1,2,3,4,5,6,7,8,9,10]})
        dicts.append({"scene0046_00":[0,1,2,3,4,5,6,7,8,9]})
        dicts.append({"scene0203_01":[0,1,2,3,4,5,6,7,8,9,10,11]})
        dicts.append({"scene0207_02":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]})
        dicts.append({"scene0334_01":[0,1]})
        dicts.append({"scene0377_00":[0,1]})
        dicts.append({"scene0377_02":[0,1,2,3,4]})
        dicts.append({"scene0423_02":[0,1,2,3,4]})
        dicts.append({"scene0552_01":[0,1,2,3,4,5,6,7,8,9]})
        dicts.append({"scene0591_00":[0,1,2,3,4,5,6,7,8,9,10,11]})
        dicts.append({"scene0606_02":[0,1,2,3,4,5,6]})
        dicts.append({"scene0700_01":[0,1,2,3,4,5,6,7,8]})
    return dicts

def main_vis_scenes_vtk(new = True):
    print("here")
    # # root_path = "out/iscnet/2022-07-20T12;05_raw_ONet_AE_test_23/visualization/"
    # # root_path = "out/iscnet/2022-07-22T09;34_test_RfD_ONet/visualization/"
    # # root_path = "out/iscnet/2022-07-22T10;54_detailed_ONet_test1/visualization/"
    # # root_path = "out/iscnet/2022-07-22T15;22_test_skip_prop_84/visualization/"

    # # root_path = "out/iscnet/2022-08-09T12;18_test_student_teacherdecoder_detail/visualization/"
    # # root_path = "out/iscnet/2022-08-09T16;42_test_encoder_3e-5_epoch_5/visualization/"

    # if new:
    #     # dicts = [new_test_scene_dict()[11]] # 11
    #     dicts = new_test_scene_dict()
    # else:
    #     dicts = old_23_scene_dict()

    type_dict = all_target_dict2by_type_dict(new_test_scene_dict(by_type=False))


    # for dict in dicts:
    for type_str, dict in type_dict.items():
        print(type_str)
        print(dict)
        vis_all_scene_data_vtk_new(start_id=0, skip=1, selected_object_dict=dict,
                                   root_paths=[
                                       # "out/iscnet/2022-08-09T16;42_test_encoder_3e-5_epoch_5/visualization/",
                                       # "out/iscnet/2022-07-22T09;34_test_RfD_ONet/visualization/"
                                       # 'out/iscnet/2022-08-12T17;02_learn_code_test_12_scenes/visualization/',
                                       # 'out/iscnet/2022-08-14T16;39_test_teacher/visualization/'

                                       'out/iscnet/2022-08-15T19;47_RfDNet_test/visualization/',
                                       # 'out/iscnet/2022-08-21T10;20_test_BCE/visualization/',
                                       # 'out/iscnet/2022-08-22T12;05_test_inner/visualization/',
                                       'out/iscnet/2022-08-15T19;28_test_student_25_epoch/visualization/',
                                       # 'out/iscnet/2022-08-22T16;04_test_freeze_BCE/visualization/',


                                        'out/iscnet/2022-08-22T17;16_test_CNN_student_12_scenes/visualization',
                                        'out/iscnet/2022-08-14T16;39_test_teacher/visualization/',
                                        'out/iscnet/2022-08-22T16;49_test_CNN_teacher_12_scenes/visualization',


                                       # 'out/iscnet/2022-08-18T20;09_test_PE_1e-5/visualization/',
                                   ],
                                   # tittles=['RfD', 'KD_code_res','KD_inner','KD_code', 'teacher'],
                                   tittles=['RfD', 'ONet_S','CNN_S','ONet_T' ,'CNN_T'],
                                   # tittles=[ 'PE'],
                                   by_row=False)




red = [1,0.67,0.6]
blue = [0.6,0.67,1]
green = [0.6,1,0.67]




def vis_dimr_vtk_old(selected_object_dict={'scene0019_01':[0,3,5],'scene0077_00':[0,2,3]},
               gt_mesh_path = 'test_dimr/gt_meshes/',
               root_paths=[
                   "test_dimr/pred_meshes/",
                   "test_dimr/pred_meshes2/"
               ],
               tittles=['dimr1', 'dimr2'],
               by_row=True,
                 cat_path_dicts = None):
    if cat_path_dicts is None:
        # get all pred_meshes
        temp_global_pred_mesh_path_list = [] # all mesh paths of all experiment
        temp_global_pred_mesh_list = []
        for root_path in root_paths:
            expe_pred_mesh_paths = os.listdir(root_path) # all mesh paths of one experiment
            expe_pred_meshes = []
            for i, expe_mesh_path in enumerate(expe_pred_mesh_paths):
                expe_pred_mesh_paths[i] = os.path.join(root_path,expe_mesh_path)
                mesh = trimesh.load(expe_pred_mesh_paths[i])
                expe_pred_meshes.append(mesh)

            temp_global_pred_mesh_list.append(expe_pred_meshes)
            temp_global_pred_mesh_path_list.append(expe_pred_mesh_paths)

        global_GT_mesh_path_list = []
        global_pred_mesh_path_dicts =[] # a list of dicts, each dict belongs to an experiment
        global_GT_mesh_path_dict ={}
        for i in range(len(root_paths)):
            global_pred_mesh_path_dicts.append({})

        # get gt
        for scene_name, object_list in selected_object_dict.items():
            global_GT_mesh_path_dict[scene_name] = []
            for object_id in object_list:
                match_str =  os.path.join(gt_mesh_path, scene_name + '_%d_*'%object_id )
                temp = glob(match_str) #
                global_GT_mesh_path_list.append(temp[0]) # put into the list
                global_GT_mesh_path_dict[scene_name].append(temp[0]) # put into the dict
        # get pred
        for scene_name, object_list in selected_object_dict.items():
            for expe_id in range(len(root_paths)):
                expe_scene_pred_path_list = []
                for object_id , gt_path in enumerate(global_GT_mesh_path_dict[scene_name]):
                    gt_mesh = trimesh.load(gt_path)
                    gt_centroid = gt_mesh.centroid
                    # gt_box_volume = gt_mesh.bounding_box_oriented.volume

                    best_pred_id = 0
                    best_centroid_diff = 100000000
                    for pred_mesh_id, pred_mesh in \
                            enumerate(temp_global_pred_mesh_list[expe_id]):  # modify this, should be this scene only
                        pred_centroid = pred_mesh.centroid
                        # pred_box_volume = pred_mesh.bounding_box_oriented.volume
                        centroid_diff = np.dot((gt_centroid - pred_centroid),gt_centroid - pred_centroid)
                        if centroid_diff < best_centroid_diff:
                            best_centroid_diff = centroid_diff
                            best_pred_id = pred_mesh_id
                    # now we have the best pred_id for this experiment
                    expe_scene_pred_path_list.append(temp_global_pred_mesh_path_list[expe_id][best_pred_id])
                # now we have the pred_paths for this experiment for this scene
                global_pred_mesh_path_dicts[expe_id][scene_name] = expe_scene_pred_path_list

        # for scene_name,mesh_path_list in global_pred_mesh_path_dicts[0].items():
        #     print(scene_name,len(mesh_path_list))

        tittles = [*tittles, 'GT']
        cat_dicts = [*global_pred_mesh_path_dicts, global_GT_mesh_path_dict]
        # save dict
        # print(datetime.date.now())
        save_path = str(datetime.datetime.now()).replace(':',';') +'_vis_path_dict.npy'
        print(datetime.datetime.now())
        print(save_path)
        np.save(save_path, cat_dicts)

        # load dict
        # new_dict = np.load('file.npy', allow_pickle='TRUE')
    else:
        print('start else')
        cat_dicts = cat_path_dicts
        global_GT_mesh_path_dict = cat_dicts[-1]
    # print(cat_dicts)
    for scene_name, path_list in global_GT_mesh_path_dict.items():
        for path in path_list:
            print("'" + path + "',")

    all_mesh_actors = []
    red = [255.0 / 255,  # 1
           171.0 / 255,  # 0.67
           153.0 / 255]  # 0.6
    margin = 1.2
    x_offset_factor = 0.05
    for root_path_id, mesh_dict in enumerate(cat_dicts):  # same row
        if by_row:
            y_shift = - root_path_id * margin
            x_shift = 0
            offset_factor = 1
            text_position = [x_shift - x_offset_factor * len(tittles[root_path_id]) - 1,
                             y_shift,  # + offset_factor,
                             0]

        else:
            x_shift = root_path_id * margin
            y_shift = 0

            text_position = [x_shift - x_offset_factor * len(tittles[root_path_id]),
                             y_shift - 1,
                             0]
        all_mesh_actors.append(get_text_actor_vtk(tittles[root_path_id], text_position))

        for scene_name, pathes in mesh_dict.items():
            # print(scene_name,":",pathes)
            for path in pathes:  # same volume
                if os.path.exists(path):
                    pass
                    all_mesh_actors.append(get_mesh_actor_vtk(path, red, opacity=1.0,
                                                              use_translation=True,
                                                              translation= (x_shift, y_shift, 0),
                                                              normalize=True))
                if by_row:
                    x_shift = x_shift + margin
                else:
                    y_shift = y_shift + margin
    vis_actors_vtk(all_mesh_actors,parallel=True)

def vis_dimr_vtk(selected_object_dict={'scene0019_01':[0,3,5],'scene0077_00':[0,2,3]},
               gt_mesh_path = 'test_dimr/gt_meshes/',
               root_paths=[
                   "test_dimr/pred_meshes/",
                   "test_dimr/pred_meshes2/"
               ],
               tittles=['dimr1', 'dimr2'],
               by_row=True,
                partial_pc = False):

    global_GT_mesh_path_list = []
    global_pred_mesh_path_list = []

    global_pred_mesh_path_dicts =[] # a list of dicts, each dict belongs to an experiment
    global_GT_mesh_path_dict ={}
    for i in range(len(root_paths)):
        global_pred_mesh_path_dicts.append({})

    # get gt
    for scene_name, object_list in selected_object_dict.items():
        global_GT_mesh_path_dict[scene_name] = []
        for object_id in object_list:
            match_str =  os.path.join(gt_mesh_path, scene_name + '_%d_*'%object_id )
            temp = glob(match_str) #
            # print(match_str)

            global_GT_mesh_path_list.append(temp[0]) # put into the list
            global_GT_mesh_path_dict[scene_name].append(temp[0]) # put into the dict
    # get pred
    for root_path_id, rootpath in enumerate(root_paths):
        for scene_name, object_list in selected_object_dict.items():
            expe_scene_pred_path_list = []
            for   gt_path in global_GT_mesh_path_dict[scene_name]:
                # import re
                # gt_id = re.findall(scene_name + r"_(\d)+_", gt_path)[0].replace('_', '').replace('gt', '') #2
                gt_id = os.path.basename(gt_path).split('_')[2]
                gt_id = int(gt_id)

                match_str = os.path.join(rootpath, scene_name + '*_gt_%d_*' % gt_id)
                match_str = match_str.replace('[','?').replace(']','?')
                temp = glob(match_str)  #
                if len(temp) == 0:
                    expe_scene_pred_path_list.append('')
                    # print('not detected',match_str)
                    # print(os.path.exists(rootpath),rootpath)
                    continue

                pred_mesh_path = temp[0]
                # print('matched:', pred_mesh_path)
                expe_scene_pred_path_list.append(pred_mesh_path)
             # now we have the pred_paths for this experiment for this scene
            global_pred_mesh_path_dicts[root_path_id][scene_name] = expe_scene_pred_path_list

    # for scene_name,mesh_path_list in global_pred_mesh_path_dicts[0].items():
    #     print(scene_name,len(mesh_path_list))

    tittles = [*tittles, 'GT']
    cat_dicts = [*global_pred_mesh_path_dicts, global_GT_mesh_path_dict]
    # print(global_pred_mesh_path_dicts)
    # save dict
    # print(datetime.date.now())


    # print(cat_dicts)
    # for scene_name, path_list in global_GT_mesh_path_dict.items():
    #     for path in path_list:
    #         print("'" + path + "',")

    all_mesh_actors = []
    red = [255.0 / 255,  # 1
           171.0 / 255,  # 0.67
           153.0 / 255]  # 0.6
    margin = 1.2
    x_offset_factor = 0.05

    for root_path_id, mesh_dict in enumerate(cat_dicts):  # same row
        if root_path_id != (len(cat_dicts) -1): # not gt
            rootpath = root_paths[root_path_id]
        if by_row:
            y_shift = - root_path_id * margin
            x_shift = 0
            offset_factor = 1
            text_position = [x_shift - x_offset_factor * len(tittles[root_path_id]) - 1,
                             y_shift,  # + offset_factor,
                             0]

        else:
            x_shift = root_path_id * margin
            y_shift = 0

            text_position = [x_shift - x_offset_factor * len(tittles[root_path_id]),
                             y_shift - 1,
                             0]
        all_mesh_actors.append(get_text_actor_vtk(tittles[root_path_id], text_position))

        if partial_pc:
            mesh_opacity = 0.8
        else:
            mesh_opacity = 1
        pc_opacity = 0.8
        for scene_name, pathes in mesh_dict.items():
            bbox_path =  os.path.join(rootpath.replace('meshes', 'bbox'), scene_name+'.npy')
            if os.path.exists(bbox_path):
                bboxes = np.load(bbox_path)
            else:
                bboxes = None
            gt_bboxes = np.load(os.path.join('datasets/gt_bboxes', scene_name+'.npy'))


            # print('pathes',pathes)

            for path in pathes:  # same volume
                # print(path)
                if os.path.exists(path):
                    # import re
                    if 'gt_meshes' in path:
                        # gt_id = re.findall(scene_name + r"_(\d)+_",path)[0].replace('_','').replace('gt','')
                        gt_id =  os.path.basename(path).split('_')[2]
                        id = int(gt_id)
                        all_mesh_actors.append(get_mesh_actor_vtk(path, red, opacity=mesh_opacity,
                                                                  use_translation=True,
                                                                  translation=(x_shift, y_shift, 0),
                                                                  normalize=False, bbox=gt_bboxes[id]))
                        # if partial_pc:
                        #
                        #     all_mesh_actors.append(get_pc_actor_vtk(path.replace('meshes', 'partial_pc'), blue,
                        #                                             alpha=pc_opacity,
                        #                                             use_translation=True,
                        #                                             translation=(x_shift, y_shift, 0),
                        #                                             normalize=False, bbox=gt_bboxes[id]))

                    else:
                        # pred_id = re.findall(r"_(\d)+_gt",path)[0].replace('_','').replace('gt','')
                        pred_id =  os.path.basename(path).split('_')[2]
                        id = int(pred_id)
                        if bboxes is not None:
                            bbox = bboxes[id]
                        else:
                            bbox = None
                        # print('id',id)
                        if bbox is not None:
                            all_mesh_actors.append(get_mesh_actor_vtk(path, red, opacity=mesh_opacity,
                                                                      use_translation=True,
                                                                      translation= (x_shift, y_shift, 0),
                                                                      normalize=False,bbox=bbox))
                            if partial_pc:
                                all_mesh_actors.append(get_pc_actor_vtk(path.replace('meshes','partial_pc'),  blue,
                                                                        alpha=pc_opacity,
                                                                      use_translation=True,
                                                                      translation= (x_shift, y_shift, 0),
                                                                      normalize=False,bbox=bbox))
                        else:
                            all_mesh_actors.append(get_mesh_actor_vtk(path, red, opacity=1.0,
                                                                      use_translation=True,
                                                                      translation=(x_shift, y_shift, 0),
                                                                      normalize=True, bbox=bbox))

                if by_row:
                    x_shift = x_shift + margin
                else:
                    y_shift = y_shift + margin
    vis_actors_vtk(all_mesh_actors,parallel=True)

def main_compare_vis_dimr_scenes_vtk():
    print("here")
    dicts =  get_dicts_from_gt_meshes()
    # dicts =  get_dicts_from_scenes()
    # dicts = [{'scene0606_02': [0,1,2,3,4,6]}]
    # dicts = [{'scene0207_02': [11,13]}]
    # dicts = [{'scene0700_01': [0,1,2,3,4,5,6,7,8]}]

    # cat_path_dicts = []
    # dict_file_paths = glob('*dict.npy')
    # for dict_file_path in dict_file_paths:
    #     cat_path_dicts.append(np.load(dict_file_path, allow_pickle='TRUE')[0])
    #     print("np.load(dict_file_path, allow_pickle='TRUE')",np.load(dict_file_path, allow_pickle='TRUE'))

    # dicts = [chair_dict]
    for dict in dicts:
        print(dict)
        vis_dimr_vtk(selected_object_dict=dict,
               gt_mesh_path = '../2dimr/datasets/gt_meshes/meshes/',
               root_paths=[
                # 'wrong_path',
                   '../dimr/exp/scannetv2/rfs/my_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes/',
                #    '../2dimr/exp/scannetv2/rfs/[2022.09.10.14.49]test_trained_gt2_test_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes/',
                   '../2dimr/exp/scannetv2/rfs/2022.09.13.16.52.test_gt_fix_decoder_with_pc_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes/',
                   # '../2dimr/exp/scannetv2/rfs/[2022.09.11.16.48]test_gt_free_decoder_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes',
                   # '../2dimr/exp/scannetv2/rfs/[2022.09.12.09.14]test_fix_pcn_pre_onet_test_set_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes',
                   # '../2dimr/exp/scannetv2/rfs/[2022.09.11.17.09]test_gt_free_decoder_teacher_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes',
                   # '../2dimr/exp/scannetv2/rfs/[2022.09.12.10.07]test_all_pretrained_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes',
                   # '../original_dimr/exp/scannetv2/rfs/rfs_pretrained_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes/',
                   # '../dimr/exp/scannetv2/rfs/[2022.09.13.10.19]test_train_onet_from_scratch_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes/',
                   '../2dimr/exp/scannetv2/rfs/2022.09.13.17.59.test_not_gt_fix_decoder_scannet/result/epoch256_nmst0.3_scoret0.4_npointt100/val/meshes/',
                   # '../2dimr/exp/scannetv2/rfs/2022.09.14.19.34_test_not_gt_248_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes/',

                   '../dimr/exp/scannetv2/rfs/2022.10.12_test_all_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/meshes/',

               ],

               tittles=[
                        'my_pretrain',
                        # 'fix_decoder',
                        'fix_decoder', # w_pc
                        # 'free_decoder',
                        # 'fix_pcn_pre',
                        # 'free_decoder_teacher',
                        # 'pretrained',
                        # 'dimr',
                        # 'fix_pcn_scratch'
                        'not gt 48', #48
                        # 'not gt 248', #248
                        'new_'
                        ],
               # tittles=['fix_decoder','dimr'],
               by_row=True,)


def scan_mesh_normalize(mesh):
    points = mesh.vertices
    # swap axes
    transform_GTtxt2ply = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    points = points.dot(transform_GTtxt2ply.T)
    mesh.vertices = points

    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2  # mesh.apply_translation(-loc)
    scale = (bbox[1] - bbox[0]).max()  # mesh.apply_scale(1 / scale)
    # move to origin
    # obj_points = obj_points - (obj_points.max(0) + obj_points.min(0)) / 2.
    points = points - loc
    # scale
    points = points / scale
    mesh.vertices = points
    return mesh




if __name__ == "__main__":
    print("start")
    # test_vtk_mesh()
    '''vis occ values'''
    # file = r"out\iscnet\2022.05.16.15.14_SA_10_epochs_occ_hat\occ_hat_epoch100.npz"
    # file = r"out\iscnet\2022-05-17T_test_uniform_4096_30\occ_hat.npz"
    # vix_occ_hat_voxel_vtk(file=file,all = True)
    # vis_multi_occ_hat_histogram([file])


    '''get test scene type'''
    # test_scene_names = get_test_scene_names()
    # with open("scene_type.csv", "r") as f:
    #     data = f.readlines()
    #
    # for scene_data in data:
    #     scene_name, scene_type = scene_data.split(',')
    #     with open("test_scene_type.csv", "a") as f:
    #         if scene_name in test_scene_names:
    #             f.write("test,"+scene_data)
    #         else:
    #             f.write("train,"+scene_data)


    '''vis dimr result by scenes'''
    # main_vis_dimr_scenes()
    main_compare_dimr_scenes() # !!!!!!!!!!!!!!!!!!!! world

    '''compare vis different results by vtk'''
    # main_compare_vis_dimr_scenes_vtk() #!!!!!!!!!!!!!!!!!!!!  canonical
    # main_vis_scenes_vtk() # not dimr

