import argparse
import h5py
from pathlib import Path
import cv2
import pycolmap
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R, rotation
import numpy as np
import torch
import open3d as o3d
import sys

import json

from hloc.utils.open3d_helper import custom_draw_geometry, load_view_point
from hloc.utils.camera_projection_helper import synthesize_img_given_viewpoint, load_depth_to_scan
# from hloc.localize_inloc import interpolate_scan

def interpolate_scan(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w-1, h-1]]) * 2 - 1 # To normalize kp values b/w [-1, 1] i.e. [0, 1] * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None] # [1, 3, 1200, 1600]
    kp = torch.from_numpy(kp)[None, None] # None adds extra 1 dimension: kp shape now -> [1,1,X,2], X being number of keypoints for an image
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(
        scan, kp, align_corners=True, mode='bilinear')[0, :, 0] #output shape: [3,no_of_kps]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode='nearest')[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)
    #alid = torch.nonzero(~valid)
    #print(valid[:25], alid)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid

def pose_from_2d3dpair_habitat():
    pass

def pose_from_2d3dpair_inloc(dataset_dir, img_path, feature_file, skip_matches=20):
    height, width = cv2.imread(str(dataset_dir / img_path)).shape[:2]
    cx = .5 * width 
    cy = .5 * height
    focal_length = 4032. * 28. / 36.
    kpr = feature_file[img_path]['keypoints'].__array__()
    scan_r = loadmat(Path(dataset_dir, img_path + '.mat'))["XYZcut"]
    kp3d, valid = interpolate_scan(scan_r, kpr)
    cfg = {
        'model': 'SIMPLE_PINHOLE',
        'width': width,
        'height': height,
        'params': [focal_length, cx, cy]
    }
    ret = pycolmap.absolute_pose_estimation(
        kpr, kp3d, cfg, 48.00)
    ret['cfg'] = cfg

    return ret, kpr, kp3d


# def old_o3d_params():
#     focal_length_o3d = 617.47611289830479
#     cx_o3d = 682.5
#     cy_o3d = 356.0
#         #356.0,
# #    K[0][0], K[1][1] = focal_length, focal_length
# #    K[0][2] = cx
# #    K[1][2] = cy
# #    print("K after")
#     intrinsic_matrix_o3d = list([ 
#         focal_length_o3d,
#         0.0,
#         0.0,
#         0.0,
#         focal_length_o3d,
#         0.0,
#         cx_o3d,
#         cy_o3d,
#         1.0
#     ])

#     o3d_params = False 
#     if o3d_params == True:
#         intrinsic_matrix_1 = intrinsic_matrix_o3d
#         H =713
#         W = 1366


def find_p3p_pose(dataset_dir, features, img_path):
    assert features.exists(), features
    img_full = dataset_dir / img_path
    assert img_full.exists(), img_full
    
    feature_file = h5py.File(features, 'r')
    img_path_str = str(img_path)
    ret, kpq, kp3d = pose_from_2d3dpair_inloc(dataset_dir, img_path_str, feature_file)

    ret_scipy_convention = ret['qvec']
    ret_scipy_convention[0] = ret['qvec'][1]
    ret_scipy_convention[1] = ret['qvec'][2]
    ret_scipy_convention[2] = ret['qvec'][3]
    ret_scipy_convention[3] = ret['qvec'][0]

    rot_matrix = R.from_quat(ret_scipy_convention).as_matrix()
    # print(f"tvec, qvec: {ret['tvec'], ret_scipy_convention}")
    # print(f"rot-matrix: {rot_matrix}")
    extrinsic_matrix = np.hstack((rot_matrix, ret['tvec'].reshape((3,1))))
    extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([[0,0,0,1]])))


    extrinsic_matrix_col_major = list(extrinsic_matrix.T.reshape((16)))

    H = 1200
    W = 1600
    cx = (.5 * W)# - 0.5
    cy = (.5 * H)# - 0.5
    focal_length = 4032. * 28. / 36. 


    intrinsic_matrix_1 = list([ 
        focal_length,
        0.0,
        0.0,
        0.0,
        focal_length,
        0.0,
        cx,
        cy,
        1.0
    ])
    # print(f"MATRICES: exitrinsic: {extrinsic_matrix}")
    # print(f"FLAT: extrinsic, intrinsic: {extrinsic_matrix_col_major} {intrinsic_matrix_1}")


    p3p_pose = {
        "class_name": "PinholeCameraParameters",
        "extrinsic": list(extrinsic_matrix_col_major), 
        "intrinsic": {"height": H,
                        "intrinsic_matrix": intrinsic_matrix_1,
                        "width": W},
        "version_major": 1,
        "version_minor": 0
    }

    return p3p_pose

def merged_pcd_from_mat_files_inloc(mat_files):
    pcds = []

    for mat_file in mat_files:
        mat_file_path = Path(mat_file)
        assert mat_file_path.exists(), mat_file_path
        print(mat_file)

        xyz_file  = loadmat(Path(mat_file))["XYZcut"]
        rgb_file = loadmat(Path(mat_file))["RGBcut"]

        xyz_sp = (xyz_file.shape)

        xyz_file = (xyz_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))
        rgb_file = (rgb_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_file)
        pcd.colors = o3d.utility.Vector3dVector(rgb_file/255.0)

        pcds.append(pcd)
    
    pcd_final = o3d.geometry.PointCloud()
    for pcd_each in pcds:
        pcd_final += pcd_each
    
    return pcd_final
    

def pcd_and_p3pfile_given_p3p_pose(dataset_dir, img_path,room_no, p3p_pose=None,  merge_entire_room=False, save_compute=True, downsample=False):

    # base_path = "/home/shubodh/hdd1/Shubodh/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/"
    base_path = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/"
    if p3p_pose is None: # If the json file has full info (for example, through some external method)
        # filename = base_path + "graphVPR/ideas_SG/place-graphVPR/rand_json/T_" + room_no + "_l1_blue_issue.json"
        filename = base_path + "graphVPR/ideas_SG/place-graphVPR/rand_json/" + room_no + "/" +room_no + "_L1.json"
        # coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        #custom_draw_geometry(pcd_final, coord_mesh, filename, show_coord=True)
    else: # Here dumping p3p_pose (and other instrinsics) info into new json file as we calculated that in this script (and not existing before)
        filename = base_path + "graphVPR/ideas_SG/place-graphVPR/rand_json/p3p_"+ room_no + "/p3p_" + room_no + ".json"
  
        json_object = json.dumps(p3p_pose, indent = 4)
        with open(filename, "w") as p3p_pose_file:
            p3p_pose_file.write(json_object)

    if merge_entire_room:
        room_path_str =str(dataset_dir) + "/" + "cutouts_imageonly/DUC1/" + room_no + "/"
        room_path = Path(room_path_str)  
        print(f"room_path: {room_path}")
        mat_files = sorted(list(room_path.glob('*.jpg.mat')))
        if save_compute:
            mat_files = mat_files[:6]#6
            print("Note: Using only 6 mat files of entire room for computation.")
    else:
        str_path = str(dataset_dir) + "/" + str(img_path) + ".mat"
        mat_files = [Path(str_path)]


    pcd_final = merged_pcd_from_mat_files_inloc(mat_files)

    # Downsampling for quicker visualization. Not needed if less pcds.
    print(f"len(pcd.points): {len(pcd_final.points)}")
    if downsample == True:
        print(f"Before downsampling: {len(pcd_final.points)}")
        pcd_final = pcd_final.voxel_down_sample(voxel_size=0.002) #0.01
        print(f"After downsampling: {len(pcd_final.points)}")

    
    return pcd_final, filename




def main_using_p3p_pose(dataset_dir, features):
    room_no = "024"#024, 025, 005, 084, 010
    #DUC_cutout_024_150_0.jpg" "/DUC_cutout_025_0_0.jpg"
    img_path = Path("cutouts_imageonly/DUC1/" + room_no + "/DUC_cutout_024_150_0.jpg") 

    p3p_pose = find_p3p_pose(dataset_dir, features, img_path)
    pcd_final, p3p_file = pcd_and_p3pfile_given_p3p_pose(dataset_dir, img_path, room_no, p3p_pose)
    load_view_point(pcd_final, p3p_file, custom_inloc_viewer=False)
    synthesize_img_given_viewpoint(pcd_final, p3p_file)

def main_using_o3dviz_pose(dataset_dir, features):
    room_no = "024"#024, 025, 005, 084, 010
    #DUC_cutout_024_150_0.jpg" "/DUC_cutout_025_0_0.jpg"
    img_path = Path("cutouts_imageonly/DUC1/" + room_no + "/DUC_cutout_024_150_0.jpg") 

    p3p_pose = find_p3p_pose(dataset_dir, features, img_path)
    pcd_final, p3p_file = pcd_and_p3pfile_given_p3p_pose(dataset_dir, img_path, room_no, 
                                    p3p_pose=None, merge_entire_room=True)
    load_view_point(pcd_final, p3p_file, custom_inloc_viewer=False)
    synthesize_img_given_viewpoint(pcd_final, p3p_file)

# def main(dataset_dir, features, img_path):
#     temp_all(dataset_dir, features, img_path)
#     # 1. main_using_p3p_pose()
#     # 2. main_using_gt_pose() (for RIO10)
#     # 3. main_using_any_pose() (either selected thru open3d or your convex hull algo)

if __name__ == '__main__':
    # Example arguments:
    # --dataset_dir ./datasets/inloc_small/ # This would be path of where `cutouts_imageonly` resides. 
    # --features ./outputs/inloc_small/feats-superpoint-n4096-r1600.h5 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    args = parser.parse_args()
    main_using_p3p_pose(**args.__dict__)
    main_using_o3dviz_pose(**args.__dict__)