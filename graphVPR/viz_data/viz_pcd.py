import open3d as o3d
import os
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat, savemat
import cv2
import pycolmap
import sys

def camera_intrinsics(habitat=True):
        #if habitat=True, those are actual parameters. If false,then use
        # primesense parameters. Doesn't make sense but just for testing purpose.
        fx, fy, cx, cy, width, height = 960, 960, 960.5, 540.5, 1920, 1080
        if habitat is True:
                cam_int = o3d.camera.PinholeCameraIntrinsic(width,height,fx,fy,cx,cy)
        else:
                cam_int = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        #print("Camera intrinsics matrix:")
        #print(cam_int.intrinsic_matrix)
        return cam_int

def scan_viz(scan_r_flat, scan_r_rgb_flat, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan_r_flat)
    if color==True:
        pcd.colors = o3d.utility.Vector3dVector(scan_r_rgb_flat/255.0)
    o3d.visualization.draw_geometries([pcd])

def flatten_scan(scan):
    scan_flat = scan.reshape((scan.shape[0]*scan.shape[1], scan.shape[2]))
    return scan_flat

def pcd_from_mat(r):
    scan_r = loadmat(Path('./data/' + r + '.mat'))["XYZcut"]
    scan_r_rgb = loadmat(Path('./data/' + r + '.mat'))["RGBcut"]
    #dict_sav = {"XYZcut": scan_r_l}
    #savemat(Path('./data/' + r + 'samply.mat'), dict_sav)

    #scan_r = loadmat(Path('./data/' + r + '.mat'))["XYZcut"]
    print((scan_r.shape))

    scan_r_flat = flatten_scan(scan_r)
    scan_r_rgb_flat = flatten_scan(scan_r_rgb)
    scan_viz(scan_r_flat, scan_r_rgb_flat, True)

def pcd_from_depth_usingo3d():
    # returns: pcd, (depth_img or rgb_img first 2 dimensions, i.e. 1080, 1920)
    # TODO-maybe: If pipeline goes wrong, you may have to look at the following
    # 1. depth_scale
    # 2. .transform() #Doing flip
    img_id = "0"
    depth_path = os.path.join('./data/' +img_id + "_depth.png")
    rgb_path = os.path.join('./data/' +img_id + "_rgb.png")
    color_raw = o3d.io.read_image(rgb_path)

    #discard the 4th channel of RGB image
    clr_arr = np.asarray(color_raw)
    clr_arr_new = clr_arr[:,:,:3]
    color_raw_new = o3d.geometry.Image(clr_arr_new.astype(np.uint8))

    depth_raw = o3d.io.read_image(depth_path)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_new, depth_raw,depth_scale=1000.0,depth_trunc=1000.0,convert_rgb_to_intensity=False)

    cam_int = camera_intrinsics(habitat=True)
    pcd_query = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, cam_int)

    #pcd_query.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) #Doing flip as per current understanding, Link for more details: https://www.notion.so/saishubodh/Depth-Images-to-Point-Cloud-Conversion-61ef9de0c99c40ec9cb72ef7d69819ab  #TODO: verify once later again
    print(np.asarray(pcd_query.points).shape)
    sys.exit()
    #o3d.visualization.draw_geometries([pcd_query])
    
    return np.asarray(pcd_query.points), (np.asarray(depth_raw).shape)

def pcd_from_depth():
    # returns: pcd, (depth_img or rgb_img first 2 dimensions, i.e. 1080, 1920)
    # TODO-maybe: If pipeline goes wrong, you may have to look at the following
    # 1. depth_scale
    # 2. .transform() #Doing flip
    img_id = "0"
    depth_path = os.path.join('./data/' +img_id + "_depth.png")
    rgb_path = os.path.join('./data/' +img_id + "_rgb.png")
    color_raw = o3d.io.read_image(rgb_path)

    #discard the 4th channel of RGB image
    clr_arr = np.asarray(color_raw)
    clr_arr_new = clr_arr[:,:,:3]
    color_raw_new = o3d.geometry.Image(clr_arr_new.astype(np.uint8))
    color_np = np.asarray(color_raw_new)
    depth_raw = o3d.io.read_image(depth_path)
    depth_np = np.asarray(depth_raw)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_new, depth_raw,depth_scale=1000.0,depth_trunc=1000.0,convert_rgb_to_intensity=False)
    cam_int = camera_intrinsics(habitat=True)
    pcd_query = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, cam_int)

    #pcd_query.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) #Doing flip as per current understanding, Link for more details: https://www.notion.so/saishubodh/Depth-Images-to-Point-Cloud-Conversion-61ef9de0c99c40ec9cb72ef7d69819ab  #TODO: verify once later again
    print(np.asarray(pcd_query.points).shape)
    sys.exit()
    #o3d.visualization.draw_geometries([pcd_query])
    
    return np.asarray(pcd_query.points), (np.asarray(depth_raw).shape)


def mat_from_pcd(pcd):
    # Input: np array of dimension [width, height, 3], and this will save it as .mat file
    #dict_sav = {"XYZcut": scan_r_l}
    #savemat(Path('./data/' + r + 'samply.mat'), dict_sav)
    pass

if __name__ == '__main__':
    r = 'DUC_cutout_082_30_0.jpg'
    #pcd_from_mat(r)
    pcd, dim = pcd_from_depth()
    #print(pcd.shape)
    #scan_unflat = pcd.reshape((dim[0],dim[1], -1))
    #print(scan_unflat.shape)
