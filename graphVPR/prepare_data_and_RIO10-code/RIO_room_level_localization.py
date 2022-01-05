import open3d as o3d
import os
from pathlib import Path
import numpy as np
import cv2
import sys
import yaml

from plyfile import PlyData, PlyElement

def camera_intrinsics(camera_params):
        fx, fy, cx, cy  = camera_params['camera_intrinsics']['model']
        width, height = camera_params['camera_intrinsics']['width'], camera_params['camera_intrinsics']['height']
        cam_int = o3d.camera.PinholeCameraIntrinsic(width,height,fx,fy,cx,cy)
        #print(width, height, fx, fy, cx, cy)
        #print("Camera intrinsics matrix:")
        #print(cam_int.intrinsic_matrix)
        return cam_int

def pcd_from_depth(camera_params, rgb_path, depth_path):
    ''' Usage:
    pcd = pcd_from_depth(camera_params,
                rgb_path = seq_path + frame_id + rgb_ext, depth_path=seq_path+frame_id+depth_ext)
    '''

    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    #o3d.visualization.draw_geometries([depth_raw])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,depth_scale=1000.0,depth_trunc=1000.0,convert_rgb_to_intensity=False)

    cam_int = camera_intrinsics(camera_params)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, cam_int)
    o3d.visualization.draw_geometries([pcd])

    return pcd

def models_viz(path, names):
    ''' Usage:
        models_viz(models_path, models_name)
    '''
    model_0 = o3d.io.read_triangle_mesh(path + names[0], True)
    o3d.visualization.draw_geometries([model_0])

    model_1 = o3d.io.read_triangle_mesh(path + names[1])
    print(np.asarray(model_1.red))
    o3d.visualization.draw_geometries([model_1])


def ply_parser(name):
    ''' Usage:
        ply_parser(models_path + models_name[1])
    '''
    plydata = PlyData.read(name)
    object_ids = (plydata.elements[0].data['objectId'])
    object_set = set(object_ids) #len(object_set) = 40, thus, same as instances.txt
    print(len(object_set))



if __name__=='__main__':
    #seq_path = "/home/shubodh/hdd1/Shubodh/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"
    seq_path = "/home/shubodh/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"

    frame_id = "frame-004339" #000016, 003615
    rgb_ext = ".color.jpg"; pose_ext = ".pose.txt";  depth_ext = ".rendered.depth.png"
    camera_params = yaml.load(open(seq_path + "camera.yaml"), Loader=yaml.FullLoader)

    models_path = "/home/shubodh/hdd1/Shubodh/Downloads/data-non-onedrive/RIO10_data/scene01/models01/seq01_02/"
    models_name = ['mesh.obj', 'labels.ply']
    ply_parser(models_path + models_name[1])