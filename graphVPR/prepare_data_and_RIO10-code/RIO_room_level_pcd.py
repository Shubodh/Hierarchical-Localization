import open3d as o3d
import os
from pathlib import Path
import numpy as np
import cv2
import sys

def pcd_from_depth(rgb_path, depth_path):
    #depth_path = os.path.join('./data/' +img_id + "_depth.png")
    #rgb_path = os.path.join('./data/' +img_id + "_rgb.png")
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


if __name__=='__main__':
    
    input_path = "/home/shubodh/hdd1/Shubodh/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"
    #input_path = "/home/shubodh/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"

    frame_id = "frame-000000"
    rgb_ext = ".color.jpg"
    pose_ext = ".pose.txt"
    depth_ext = ".rendered.depth.png"
    camera_params_file = "camera.yaml"
    pcd_from_depth(rgb_path = input_path + frame_id + rgb_ext, depth_path=input_path+frame_id+depth_ext)