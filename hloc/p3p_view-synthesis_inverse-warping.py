import argparse
import h5py
from pathlib import Path
import cv2
import pycolmap
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R, rotation
import numpy as np

import json

from localize_inloc import interpolate_scan, viz_entire_room_by_registering
from utils.open3d_helper import custom_draw_geometry, load_view_point, synthesize_img_given_viewpoint

def pose_from_2d3dpair_habitat():
    pass

def pose_from_2d3dpair_inloc(dataset_dir, img_path, feature_file, skip_matches):
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

def main(dataset_dir, features, img_path, skip_matches=None):
    
    assert features.exists(), features
    
    feature_file = h5py.File(features, 'r')
    img_path_str = str(img_path)
    ret, kpq, kp3d = pose_from_2d3dpair_inloc(dataset_dir, img_path_str, feature_file, skip_matches)

    ret_scipy_convention = ret['qvec']
    ret_scipy_convention[0] = ret['qvec'][1]
    ret_scipy_convention[1] = ret['qvec'][2]
    ret_scipy_convention[2] = ret['qvec'][3]
    ret_scipy_convention[3] = ret['qvec'][0]

    rot_matrix = R.from_quat(ret_scipy_convention).as_matrix()
    print(f"tvec, qvec: {ret['tvec'], ret_scipy_convention}")
    print(f"rot-matrix: {rot_matrix}")
    extrinsic_matrix = np.hstack((rot_matrix, ret['tvec'].reshape((3,1))))
    extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([[0,0,0,1]])))


    # Debug: Trying inverse matrix
    #print(f"extrinsic_matrix before: {extrinsic_matrix}")
    #R_T = extrinsic_matrix[0:3,0:3].T
    #R_T_times_t = -extrinsic_matrix[0:3,0:3].T @ extrinsic_matrix[0:3,3]
    #extrinsic_matrix[0:3,0:3] = R_T
    ##extrinsic_matrix[0:3,3] = R_T_times_t
    #print(f"extrinsic_matrix after: {extrinsic_matrix}")

    
    extrinsic_matrix_col_major = list(extrinsic_matrix.T.reshape((16)))

    H = 1200
    W = 1600
    #cx = .5 * W 
    #cy = .5 * H
    cx = (.5 * W)# - 0.5
    cy = (.5 * H)# - 0.5
    focal_length = 4032. * 28. / 36. 
    focal_length_o3d = 617.47611289830479
    cx_o3d = 682.5
    cy_o3d = 356.0
        #356.0,
#    K[0][0], K[1][1] = focal_length, focal_length
#    K[0][2] = cx
#    K[1][2] = cy
#    print("K after")


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
    intrinsic_matrix_o3d = list([ 
        focal_length_o3d,
        0.0,
        0.0,
        0.0,
        focal_length_o3d,
        0.0,
        cx_o3d,
        cy_o3d,
        1.0
    ])

    o3d_params = False 
    if o3d_params == True:
        intrinsic_matrix_1 = intrinsic_matrix_o3d
        H =713
        W = 1366
    print(f"MATRICES: exitrinsic: {extrinsic_matrix}")
    print(f"FLAT: extrinsic, intrinsic: {extrinsic_matrix_col_major} {intrinsic_matrix_1}")


    p3p_pose = {
        "class_name": "PinholeCameraParameters",
        "extrinsic": list(extrinsic_matrix_col_major), 
        "intrinsic": {"height": H,
                        "intrinsic_matrix": intrinsic_matrix_1,
                        "width": W},
        "version_major": 1,
        "version_minor": 0
    }

    full_path = "/home/shubodh/hdd1/Shubodh/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/graphVPR/ideas_SG/place-graphVPR/rand_json/"
    #with open(full_path + "p3p_pose.json", "w") as p3p_pose_file:
    #    json.dump(p3p_pose, p3p_pose_file)

    viz_entire_room_by_registering(dataset_dir, r=None, img_path=img_path, p3p_pose=p3p_pose)


if __name__ == '__main__':
    # Example arguments:
    # --dataset_dir ../datasets/inloc_small/ # This would be path of where `cutouts_imageonly` resides. 
    # --img_path cutouts_imageonly/DUC1/024/DUC_cutout_024_300_0.jpg
    # --features sample_data/inloc_data/feats-superpoint-n4096-r1600.h5
    # --skip_matches 20
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--img_path', type=Path, required=True)
    parser.add_argument('--skip_matches', type=int)
    args = parser.parse_args()
    main(**args.__dict__)
