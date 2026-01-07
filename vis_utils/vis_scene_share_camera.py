'''
Visualization tools for Scannet.
author: ynie
date: July, 2020

Modified by Qiao Yu
October, 2022
'''
import sys

sys.path.append('.')
# sys.path.append('../RfDNet/')
import os

import vtk
from vis_utils.vis_scannet import Vis_Scannet
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import random
import seaborn as sns

from glob import glob


ScanNet_OBJ_CLASS_IDS = np.array([1, 7, 8, 13, 20, 31, 34, 43])


class Vis_base(Vis_Scannet):
    '''
    visualization class for scannet frames.
    '''

    def __init__(self, scene_points, instance_models, center_list, vector_list, class_ids):
        self.scene_points = scene_points
        self.instance_models = instance_models
        self.cam_K = np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]])
        self.center_list = center_list
        self.vector_list = vector_list
        self.class_ids = class_ids
        self.palette_cls = np.array([*sns.color_palette("hls", len(ScanNet_OBJ_CLASS_IDS))])
        self.depth_palette = np.array(sns.color_palette("crest_r", n_colors=100))

        self.draw_boxes = False
        self.draw_axis = False

    def set_ply_property(self, plyfile):

        plydata = vtk.vtkPLYReader()
        plydata.SetFileName(plyfile)
        plydata.Update()
        return plydata

    def set_arrow_actor(self, startpoint, vector):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: an vtk arrow actor
        '''
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.2)
        arrow_source.SetTipRadius(0.08)
        arrow_source.SetShaftRadius(0.02)

        vector = vector / np.linalg.norm(vector) * 0.5

        endpoint = startpoint + vector

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)

        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor

    def set_bbox_line_actor(self, corners, faces, color):
        edge_set1 = np.vstack([np.array(faces)[:, 0], np.array(faces)[:, 1]]).T
        edge_set2 = np.vstack([np.array(faces)[:, 1], np.array(faces)[:, 2]]).T
        edge_set3 = np.vstack([np.array(faces)[:, 2], np.array(faces)[:, 3]]).T
        edge_set4 = np.vstack([np.array(faces)[:, 3], np.array(faces)[:, 0]]).T
        edges = np.vstack([edge_set1, edge_set2, edge_set3, edge_set4])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        pts = vtk.vtkPoints()
        for corner in corners:
            pts.InsertNextPoint(corner)

        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        for edge in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            colors.InsertNextTuple3(*color)

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        linesPolyData.SetLines(lines)
        linesPolyData.GetCellData().SetScalars(colors)

        return linesPolyData

    def get_bbox_line_actor(self, center, vectors, color, opacity, width=10):
        corners, faces = self.get_box_corners(center, vectors)
        bbox_actor = self.set_actor(self.set_mapper(self.set_bbox_line_actor(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        bbox_actor.GetProperty().SetLineWidth(width)
        return bbox_actor

    def set_render(self, centroid, only_points, camera,  pointsize = 1):  #  pointsize = 3 for a single image
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw world system'''
        # renderer.AddActor(self.set_axes_actor())

        '''set camera'''
        if camera is not None:
            pass
        else:
            camera = self.set_camera(centroid, [[0., 0., 0.], [-centroid[0], -centroid[1],
                                                               centroid[0] ** 2 / centroid[2] + centroid[1] ** 2 /
                                                               centroid[2]]], self.cam_K)
        renderer.SetActiveCamera(camera)

        '''draw scene points'''
        colors = np.linalg.norm(self.scene_points[:, :3] - centroid, axis=1)
        colors = self.depth_palette[np.int16((colors - colors.min()) / (colors.max() - colors.min()) * 99)]
        point_actor = self.set_actor(
            self.set_mapper(self.set_points_property(self.scene_points[:, :3], 255 * colors), 'box'))
        point_actor.GetProperty().SetPointSize(pointsize)
        point_actor.GetProperty().SetOpacity(0.3)
        point_actor.GetProperty().SetInterpolationToPBR()
        renderer.AddActor(point_actor)

        if not only_points:
            '''draw shapenet models'''
            palette_inst = np.array([*sns.color_palette("hls", 20)])

            # dists = np.linalg.norm(np.array(self.center_list) - centroid, axis=1)
            # min_max_dist = [min(dists), max(dists)]
            # dists = (dists - min_max_dist[0])/(min_max_dist[1]-min_max_dist[0])
            # dists = np.clip(dists, 0, 1)
            # inst_color_ids = np.round(dists*(palette_inst.shape[0]-1)).astype(np.uint8)
            inst_color_ids = self.class_ids
            for obj, cls_id, color_id in zip(self.instance_models, self.class_ids, inst_color_ids):
                object_actor = self.set_actor(self.set_mapper(obj, 'model'))
                object_actor.GetProperty().SetColor(self.palette_cls[cls_id])
                object_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(object_actor)

            '''draw bounding boxes'''
            if self.draw_boxes:
                for center, vectors, cls_id, color_id in zip(self.center_list, self.vector_list, self.class_ids,
                                                             inst_color_ids):
                    # box_line_actor = self.get_bbox_line_actor(center, vectors, 255*self.palette_cls[cls_id], 1., 10)
                    # box_line_actor.GetProperty().SetInterpolationToPBR()
                    # renderer.AddActor(box_line_actor)

                    corners, faces = self.get_box_corners(center, vectors)
                    bbox_actor = self.set_actor(
                        self.set_mapper(self.set_cube_prop(corners, faces, 255 * self.palette_cls[cls_id]), 'box'))
                    bbox_actor.GetProperty().SetOpacity(0.2)
                    bbox_actor.GetProperty().SetInterpolationToPBR()
                    renderer.AddActor(bbox_actor)

                    # draw orientations
                    color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                    for index in range(vectors.shape[0]):
                        arrow_actor = self.set_arrow_actor(center, vectors[index])
                        arrow_actor.GetProperty().SetColor(color[index])
                        renderer.AddActor(arrow_actor)
            '''draw axis'''
            if self.draw_axis:
                # add axis actor
                axes = vtk.vtkAxesActor()
                renderer.AddActor(axes)
        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, -10, 10), (-10, -10, 10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)

        return renderer,camera

    def set_render_window(self, centroid, only_points, camera):
        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render(centroid, only_points, camera)
        renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(*np.int32((self.cam_K[:2, 2] * 2)))

        return render_window

    def visualize(self, centroid=np.array([0, -2.5, 2.5]), save_path=None, only_points=False, offscene=False,
                  renderers=None):
        '''
        Visualize a 3D scene.
        '''

        render_window_interactor = vtk.vtkRenderWindowInteractor()

        # render_window = self.set_render_window(centroid, only_points, camera)
        render_window = vtk.vtkRenderWindow()
        # add renderers
        for renderer in renderers:
            render_window.AddRenderer(renderer)

        render_window.SetSize(*np.int32((self.cam_K[:2, 2] * 2)))


        render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())  ## added by Qiao

        render_window.FullScreenOn()  # equals to: SetFullScreen(True)
        render_window.BordersOn()  # equals to: SetBorders(True)

        render_window_interactor.SetRenderWindow(render_window)
        if offscene:
            render_window.OffScreenRenderingOn()  # don't show window, directly save png

        render_window.Render()

        if save_path is not None:
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(render_window)
            windowToImageFilter.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(save_path)
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            writer.Write()

        if not offscene:
            render_window.OffScreenRenderingOff()  # don't show window, directly save png
            render_window_interactor.Start()
            camera = render_window.GetRenderers().GetFirstRenderer().GetActiveCamera()
            print(camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
            return camera


CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']

def vis_all_renderers(renderers,cam_K=np.array([[2000, 0, 2400], [0, 2000, 1600], [0, 0, 1]]),offscene=False,save_path = None):
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    # render_window = self.set_render_window(centroid, only_points, camera)
    render_window = vtk.vtkRenderWindow()
    # add renderers
    for renderer in renderers:
        render_window.AddRenderer(renderer)

    render_window.SetSize(*np.int32((cam_K[:2, 2] * 2)))

    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())  ## added by Qiao

    render_window.FullScreenOn()  # 相当于SetFullScreen(True)
    render_window.BordersOn()  # 相当于SetBorders(True)

    render_window_interactor.SetRenderWindow(render_window)
    if offscene:
        render_window.OffScreenRenderingOn()  # don't show window, directly save png

    render_window.Render()

    if save_path is not None:
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(render_window)
        windowToImageFilter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(save_path)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    if not offscene:
        render_window.OffScreenRenderingOff()  # don't show window, directly save png
        render_window_interactor.Start()
        camera = render_window.GetRenderers().GetFirstRenderer().GetActiveCamera()
        print(camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())


        return camera


def extract_label(f):
    if 'gt' in f:
        clsname = f[:-4].split('_')[5]  # modified by Qiao
    else:
        clsname = f[:-4].split('_')[3]  # default
    if clsname == 'trash': clsname = 'trash_bin'
    return CAD_labels.index(clsname)


def getVisBaseObjectAndSavePath(fused_points, instance_mesh_root_path, scene_dirname):
    print(os.path.exists(instance_mesh_root_path),instance_mesh_root_path)
    save_root_path = instance_mesh_root_path.replace('meshes', 'pngs')
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)

    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    instance_models = []
    center_list = []
    vector_list = []
    class_ids = []

    # mesh_files = os.listdir(instance_mesh_root_path)
    # match_str = (instance_mesh_root_path + scene_dirname + "*")
    match_str = os.path.join(instance_mesh_root_path,  scene_dirname + "*")
    # print(match_str)

    mesh_files = glob(match_str)
    # print(mesh_files)
    for mesh_file in mesh_files:
        # get mesh
        # print(mesh_file)
        vtk_object = vtk.vtkPLYReader()
        vtk_object.SetFileName((mesh_file).replace('\\', '/'))
        vtk_object.Update()
        # get points from object
        polydata = vtk_object.GetOutput()
        # read points using vtk_to_numpy
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64) # np.float or np.float64
        ###########################
        if 'jimr_pro'  in mesh_file:
            obj_points += fused_points.mean(0)
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(obj_points))
            polydata.SetPoints(points)
            vtk_object.Update()
        ###########################
        instance_models.append(vtk_object)

        # get class
        class_ids.append(extract_label(os.path.basename(mesh_file)))

    scene = Vis_base(scene_points=fused_points, instance_models=instance_models, center_list=center_list,
                     vector_list=vector_list, class_ids=class_ids)
    # save_path = (save_root_path + scene_dirname + '.png')
    return scene #, save_path

def SaveSceneToFile(file_name, camera):
    # Actor
    #   Position, orientation, origin, scale, usrmatrix, usertransform
    # Camera
    #   FocalPoint, Position, ViewUp, ViewAngle, ClippingRange

    fp_format = '{0:.6f}'
    res = dict()
    res['Camera:FocalPoint'] = ', '.join(fp_format.format(n) for n in camera.GetFocalPoint())
    res['Camera:Position'] = ', '.join(fp_format.format(n) for n in camera.GetPosition())
    res['Camera:ViewUp'] = ', '.join(fp_format.format(n) for n in camera.GetViewUp())
    res['Camera:ViewAngle'] = fp_format.format(camera.GetViewAngle())
    res['Camera:ClippingRange'] = ', '.join(fp_format.format(n) for n in camera.GetClippingRange())
    with open(file_name, 'w') as f:
        for k, v in res.items():
            f.write(k + ' ' + v + '\n')


def RestoreSceneFromFile(file_name, camera):
    import re

    # Some regular expressions.

    reCP = re.compile(r'^Camera:Position')
    reCFP = re.compile(r'^Camera:FocalPoint')
    reCVU = re.compile(r'^Camera:ViewUp')
    reCVA = re.compile(r'^Camera:ViewAngle')
    reCCR = re.compile(r'^Camera:ClippingRange')
    keys = [reCP, reCFP, reCVU, reCVA, reCCR]

    # float_number = re.compile(r'[^0-9.\-]*([0-9e.\-]*[^,])[^0-9.\-]*([0-9e.\-]*[^,])[^0-9.\-]*([0-9e.\-]*[^,])')
    # float_scalar = re.compile(r'[^0-9.\-]*([0-9.\-e]*[^,])')

    res = dict()
    with open(file_name, 'r') as f:
        for cnt, line in enumerate(f):
            if not line.strip():
                continue
            line = line.strip().replace(',', '').split()
            for i in keys:
                m = re.match(i, line[0])
                if m:
                    k = m.group(0)
                    if m:
                        #  Convert the rest of the line to floats.
                        v = list(map(lambda x: float(x), line[1:]))
                        if len(v) == 1:
                            res[k] = v[0]
                        else:
                            res[k] = v
    for k, v in res.items():
        if re.match(reCP, k):
            camera.SetPosition(v)
        elif re.match(reCFP, k):
            camera.SetFocalPoint(v)
        elif re.match(reCVU, k):
            camera.SetViewUp(v)
        elif re.match(reCVA, k):
            camera.SetViewAngle(v)
        elif re.match(reCCR, k):
            camera.SetClippingRange(v)
    return camera

if __name__ == '__main__':
    point_size = 1
    all_test_scenes = False #
    quick_check = False # if quick_check, then don't save any pngs
    shuffle = False
    processed_data_path = './datasets/scannet/processed_data/'
    output_path = f'vis_results/pointsize_{point_size}/'
    camera_param_path = 'vis_results/camera_param/'
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    if not os.path.exists(camera_param_path):
        os.makedirs(camera_param_path,exist_ok=True)


    #### modify here to set the paths !!!!!!!!!!!!!!!
    instance_mesh_root_paths = [
        # 'dimr_meshes/ours/',  # DIMR
        'exp/scannetv2/rfs/test_phase2_scannet/result/epoch256_nmst0.3_scoret0.01_npointt100/val/meshes',
        # jimr
        # 'gt_meshes' #GT
    ]


    

    if all_test_scenes:
        test_split_path = 'datasets/splits/test.txt'
        with open(test_split_path, "r", encoding="utf-8") as file_obj:
            # data = file_obj.read()
            gt_test_scenes = file_obj.readlines()  # list
            # data = file_obj.readline()
            # print(data)
        gt_test_scenes = sorted(gt_test_scenes)
        for i in range(len(gt_test_scenes)):
            gt_test_scenes[i] = gt_test_scenes[i].replace("\n", "")
        scene_names = gt_test_scenes
    else:
        scene_names = [
            'scene0406_00', # put the scenes you want to visualize here
        ]


    stride = 1.0 / len(instance_mesh_root_paths) #0.25
    margin = stride/20
    xmins = np.arange(0,1,stride)
    xmaxs = xmins + stride - margin
    ymins = np.ones(len(instance_mesh_root_paths)) *0.5 -0.7* stride #
    ymaxs = np.ones(len(instance_mesh_root_paths)) *0.5 + 0.7* stride

    if shuffle:
        random.shuffle(scene_names)
    for scene_id,scene_dirname in enumerate(scene_names):
        # if scene_id <4:
        #     continue
        print(scene_id,scene_dirname)

        save_path = output_path + scene_dirname + '.png'
        ## seek existing camera parameters
        camera = None
        scene_camera_param_path = camera_param_path + scene_dirname +'.txt'
        if os.path.exists(scene_camera_param_path):
            print("exist camera")
            camera = RestoreSceneFromFile(scene_camera_param_path,vtk.vtkCamera())
            # print(camera)

        # point_cloud_path = os.path.join(processed_data_path, scene_dirname, 'full_scan.npz')
        point_cloud_path = os.path.join(processed_data_path, scene_dirname, 'data.npz')

        fused_points = np.load(point_cloud_path)['mesh_vertices'][:, :3]
        # with open(os.path.join(path_config.processed_data_path, scene_dirname, 'bbox.pkl'), 'rb') as file:
        #     bboxes = pickle.load((file))

        scene_list = []
        save_path_list = []
        renderer_list = []

        ######## visualize and select point view

        if quick_check:
            for i, instance_mesh_root_path in enumerate(instance_mesh_root_paths):
                scene = getVisBaseObjectAndSavePath(fused_points, instance_mesh_root_path, scene_dirname)
                if i == 0:
                    renderer, camera = scene.set_render(centroid=np.array([3, 0, 3]), only_points=False,
                                                        camera=camera,pointsize=point_size)
                else:
                    renderer, _ = scene.set_render(centroid=np.array([3, 0, 3]), only_points=False, camera=camera,
                                                   pointsize=point_size)
                renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
                renderer_list.append(renderer)
            vis_all_renderers(renderer_list, save_path=None, offscene=False)  # vis, don' save
        else:
            if camera is not None:
                print('use existing camera...')
                for i, instance_mesh_root_path in enumerate(instance_mesh_root_paths):
                    scene = getVisBaseObjectAndSavePath(fused_points, instance_mesh_root_path, scene_dirname)
                    renderer, _ = scene.set_render(centroid=np.array([3, 0, 3]), only_points=False, camera=camera,
                                                   pointsize=point_size)
                    renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
                    renderer_list.append(renderer)
                vis_all_renderers(renderer_list,save_path= output_path+scene_dirname+'.png',offscene=True) # save, don't vis
            else:
                print('no existing camera, vis')
                #### first vis and save camera
                for i, instance_mesh_root_path in enumerate(instance_mesh_root_paths):
                    scene = getVisBaseObjectAndSavePath(fused_points, instance_mesh_root_path, scene_dirname)
                    if i == 0:
                        # camera = scene.visualize(centroid=np.array([3, 0, 3]), save_path=None, offscene=False)
                        renderer, camera = scene.set_render(centroid=np.array([3, 0, 3]), only_points=False,
                                                            camera=camera)

                    else:
                        renderer, _ = scene.set_render(centroid=np.array([3, 0, 3]), only_points=False, camera=camera)
                    renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
                    renderer_list.append(renderer)
                camera = vis_all_renderers(renderer_list, save_path=None, offscene=False)  # vis
                SaveSceneToFile(scene_camera_param_path,camera)
                ########## use the camera and save without vis
                for i, instance_mesh_root_path in enumerate(instance_mesh_root_paths):
                    scene = getVisBaseObjectAndSavePath(fused_points, instance_mesh_root_path, scene_dirname)
                    if i == 0:
                        # camera = scene.visualize(centroid=np.array([3, 0, 3]), save_path=None, offscene=False)
                        renderer, camera = scene.set_render(centroid=np.array([3, 0, 3]), only_points=False,
                                                            camera=camera,pointsize=point_size)
                        # print(i)
                    else:
                        renderer, _ = scene.set_render(centroid=np.array([3, 0, 3]), only_points=False, camera=camera,
                                                       pointsize=point_size)
                        # print(i)
                    renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
                    renderer_list.append(renderer)

                vis_all_renderers(renderer_list, save_path=save_path, offscene=True)  # vis

        print('--------------------------------------')
        print('saved in :', save_path)





