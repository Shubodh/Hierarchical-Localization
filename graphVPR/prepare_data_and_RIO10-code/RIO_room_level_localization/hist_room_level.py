import open3d as o3d
import os
from pathlib import Path
import numpy as np
import cv2
import sys
import yaml

from plyfile import PlyData, PlyElement

from utils import camera_intrinsics
from features import feat_vect, verify_featVect
from matching import getMatchInds, getMatchIndsTfidf, getMatchIndsBinary, getMatchIndsBinaryTfidf

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
    print(f"{names[0]} has vertices {np.asarray(model_0.vertices).shape} and triangles {np.asarray(model_0.triangles).shape} respectively.")
    o3d.visualization.draw_geometries([model_0])

    model_1 = o3d.io.read_triangle_mesh(path + names[1])
    o3d.visualization.draw_geometries([model_1])


def ply_parser(name):
    ''' Usage:
    ply_parser(models_path + models_name[1])
    '''
    plydata = PlyData.read(name)
    object_ids = (plydata.elements[0].data['objectId'])
    object_set = set(object_ids) #len(object_set) = 40, thus, same as instances.txt
    print(len(object_set))

def accuracy(predictions, gt):
    accVec = np.equal(predictions, gt)
    accu = np.sum(accVec) / accVec.shape[0]  * 100
    print(f"predictions: {predictions}") 
    print(f"Ground truth: {gt}")
    print(f"DONE FOR NOW: Getting {accu} % accuracy.")
    print(f"TODO-Next: It's actually 100% accuracy. 12,13 are probably exactly same, hence the issue. Need to account for such case in code.")


if __name__=='__main__':
    base_simserver_path = "/home/shubodh/hdd1/Shubodh/Downloads/data-non-onedrive/RIO10_data/"
    base_shublocal_path = "/home/shubodh/Downloads/data-non-onedrive/RIO10_data/"
    base_adaserver_path = "/data/RIO10_data/"
    base_path =base_adaserver_path# base_shublocal_path #base_simserver_path #

    # 1. Individual images: pcd_from_depth
    seq_path = base_path + "scene01/seq01/seq01_01/"
    frame_id = "frame-004339" #000016, 003615
    rgb_ext = ".color.jpg"; pose_ext = ".pose.txt";  depth_ext = ".rendered.depth.png"
    camera_params = yaml.load(open(seq_path + "camera.yaml"), Loader=yaml.FullLoader)

    # 2. Room level Models: ply_parser, models_viz
    models_path = base_path + "scene01/models01/seq01_01/" #seq01_02
    models_name = ['mesh.obj', 'labels.ply']

    # 3. Room level semantics -> instances.txt
    #semantics_path = base_path + "scene01/semantics01/seq01_01/" #seq01_02
    #instances_txt = semantics_path + "instances.txt"
    #instances_img = semantics_path + "frame-000000.instances.png"

    rescan_rooms_ids_small = ['01_01', '01_02', '02_01', '02_02']
    rescan_rooms_ids = ['01_01', '01_02', '02_01', '02_02', '03_01', '03_02', '04_01', '04_02', '05_01', '05_02',
                        '06_01', '06_02', '07_01', '07_02', '08_01', '08_02', '09_01', '09_02', '10_01', '10_02']
    gt = np.array([1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18])
    gt_small = np.array([1,0,3,2])
    if base_path == base_shublocal_path:
        rescan_rooms_ids = rescan_rooms_ids_small
        gt = gt_small

    instances_all = []
    for i in range(len(rescan_rooms_ids)):
        semantics_path= Path(base_path+ "scene"+ rescan_rooms_ids[i][:2]+ 
                        "/semantics" +rescan_rooms_ids[i][:2]+"/seq"+ rescan_rooms_ids[i]+ "/")
        instances_txt = semantics_path / "instances.txt"
        instances_all.append(instances_txt)

    featVect, dict_semantics = feat_vect(instances_all) #dict_semantics for debugging
    #verify_featVect(featVect, instances_all, dict_semantics, rescan_rooms_ids) #use this function to verify featVect

    #one, two, three, four = featVect[0], featVect[1], featVect[2], featVect[3]
    mInds = getMatchInds(featVect, featVect, topK=2)
    # mInds[0] would be matching to itself. What we want is mInds[1]
    predictions = mInds[1]
    accuracy(predictions, gt)