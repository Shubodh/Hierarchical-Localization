import numpy as np
import open3d as o3d
import json
import sys

import matplotlib.pyplot as plt

import cv2

import torch.nn.functional as F
import torch
from torchvision.transforms import ToTensor


def viz_entire_room_file(dataset_dir, downsample=True):
    # room level pcds are never used in this hloc pipeline
    mat_file = (dataset_dir / "scans/DUC1/DUC_scan_024.ptx.mat")
    print(mat_file)

    xyz_file  = loadmat(Path(mat_file))#["XYZcut"]
    #print(xyz_file["A"])
    #print(xyz_file.keys())
    sys.exit()
    rgb_file = loadmat(Path(mat_file))["RGBcut"]

    xyz_sp = (xyz_file.shape)

    xyz_file = (xyz_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))
    rgb_file = (rgb_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_file)
    pcd.colors = o3d.utility.Vector3dVector(rgb_file/255.0)

    

    # Downsampling for quicker visualization. Not needed if less pcds.
    print(f"len(pcd.points): {len(pcd.points)}")
    if downsample == True:
        pcd_final = pcd.voxel_down_sample(voxel_size=0.01)  
        print(f"After downsampling: {len(pcd.points)}")

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    o3d.visualization.draw_geometries([pcd_final, mesh])

def custom_draw_geometry(pcd, coord_mesh, filename, show_coord=True):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()


#    pcd_small = o3d.geometry.PointCloud()
#    pcd_small.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:100])
#    print(np.asarray(pcd.points)[:100])
#    pcd_small.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[:100])
#    vis.add_geometry(pcd_small)

    vis.add_geometry(pcd)
    if show_coord:
        vis.add_geometry(coord_mesh)
    vis.run() # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    #vis.capture_screen_image(filename+".png")
    #print(param.extrinsic)
    vis.destroy_window()

def viz_with_array_inp(xyz_points, rgb_file):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_file)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coord_mesh)
    vis.run()
    vis.destroy_window()


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename, custom_inloc_viewer=False):
    vis = o3d.visualization.Visualizer()
    # vis = o3d.visualization.O3DVisualizer()


    if custom_inloc_viewer==True:

        H = 1200
        W = 1600
        vis.create_window(width=W, height=H)

        vpt_json = json.load(open(filename))
        cx_old = vpt_json['intrinsic']['intrinsic_matrix'][6]
        cy_old = vpt_json['intrinsic']['intrinsic_matrix'][7]
        vpt_json['intrinsic']['intrinsic_matrix'][6] =  cx_old - 0.5
        vpt_json['intrinsic']['intrinsic_matrix'][7] =  cy_old - 0.5
        print("new camera params for o3d viz:")
        print(vpt_json)
        
        json_object = json.dumps(vpt_json, indent = 4)
        with open(filename + "_o3d.json", "w") as p3p_pose_file:
            p3p_pose_file.write(json_object)
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(filename + "_o3d.json")
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param, True)
        print(f"debug A:")
        print(ctr.convert_to_pinhole_camera_parameters().intrinsic)
        sys.exit()
    else:
        H = 1200
        W = 1600
        vis.create_window(width=W, height=H)
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(filename)
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param, True)

    vis.run()

    vis.capture_screen_image(filename+".png")
    #vis.capture_screen_image(filename+".png", do_render=True)
    print(f"visualizer image saved at {filename}.png")
    vis.destroy_window()
