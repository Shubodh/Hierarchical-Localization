import argparse
import h5py
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pycolmap
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R, rotation
import numpy as np
import torch
import open3d as o3d
import sys

import json

from hloc.utils.open3d_helper import custom_draw_geometry, load_view_point, viz_with_array_inp, o3d_convert_depth_frame_to_pointcloud
from hloc.utils.camera_projection_helper import synthesize_img_given_viewpoint, load_depth_to_scan, convert_depth_pixel_to_metric_coordinate, convert_depth_frame_to_pointcloud
from hloc.utils.parsers import parse_pose_file_RIO, parse_camera_file_RIO
from hloc.utils.io import read_image
from hloc.utils.viz import plot_images
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


def find_extrinsics_intrinsics_p3p_inloc(dataset_dir, features, img_path):
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
    
    pcd_final = merged_pcd_from_single_pcds(pcds)
    # pcd_final = o3d.geometry.PointCloud()
    # for pcd_each in pcds:
    #     pcd_final += pcd_each
    
    return pcd_final
    
def merged_pcd_from_single_pcds(single_pcds):
    # single_pcds is list of individual Open3D pcd
    pcd_final = o3d.geometry.PointCloud()
    for pcd_each in single_pcds:
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

    p3p_pose = find_extrinsics_intrinsics_p3p_inloc(dataset_dir, features, img_path)
    pcd_final, p3p_file = pcd_and_p3pfile_given_p3p_pose(dataset_dir, img_path, room_no, p3p_pose)
    load_view_point(pcd_final, p3p_file, custom_inloc_viewer=False)
    synthesize_img_given_viewpoint(pcd_final, p3p_file)

def main_using_o3dviz_pose(dataset_dir, features):
    room_no = "024"#024, 025, 005, 084, 010
    #DUC_cutout_024_150_0.jpg" "/DUC_cutout_025_0_0.jpg"
    img_path = Path("cutouts_imageonly/DUC1/" + room_no + "/DUC_cutout_024_150_0.jpg") 

    p3p_pose = find_extrinsics_intrinsics_p3p_inloc(dataset_dir, features, img_path)
    pcd_final, p3p_file = pcd_and_p3pfile_given_p3p_pose(dataset_dir, img_path, room_no, 
                                    p3p_pose=None, merge_entire_room=True)
    load_view_point(pcd_final, p3p_file, custom_inloc_viewer=False)
    synthesize_img_given_viewpoint(pcd_final, p3p_file)

# def main(dataset_dir, features, img_path):
#     temp_all(dataset_dir, features, img_path)
#     # 1. main_using_p3p_pose()
#     # 2. main_using_gt_pose() (for RIO10)
#     # 3. main_using_any_pose() (either selected thru open3d or your convex hull algo)
def main_exp_inloc(dataset_dir, features):
    room_no = "024"#024, 025, 005, 084, 010
    #DUC_cutout_024_150_0.jpg" "/DUC_cutout_025_0_0.jpg"
    img_path = Path("cutouts_imageonly/DUC1/" + room_no + "/DUC_cutout_024_150_0.jpg") 

    cam_params = find_extrinsics_intrinsics_p3p_inloc(dataset_dir, features, img_path)
    cam_intrinsics = cam_params['intrinsic']['intrinsic_matrix']
    height, width = cam_params['intrinsic']['height'], cam_params['intrinsic']['width']

    # print(cam_intrinsics[0], cam_intrinsics[4], cam_intrinsics[6]-0.5, cam_intrinsics[7]-0.5)

    pcd_final, p3p_file = pcd_and_p3pfile_given_p3p_pose(dataset_dir, img_path, room_no, cam_params)
    # Bring this pcd to egocentric frame and then compare below 2 print statements
    xyz = np.asarray(pcd_final.points)
    extrinsics = np.array(cam_params['extrinsic']).reshape(4,4).T
    xyz_T = xyz.T
    xyz_hom1 = np.vstack((xyz_T, np.ones(xyz_T[0].shape)))
    xyz_hom1 = np.matmul(extrinsics, xyz_hom1) #xyz_hom1.shape: 4 * 11520000
    xyz_final = xyz_hom1.T[:, :3]

    xyz_reshaped = xyz_final.reshape((height, width,3))
    pixel_x, pixel_y = 30, 70
    print(xyz_reshaped[pixel_x, pixel_y])
    depth_in = xyz_reshaped[pixel_x, pixel_y,2]

    cam_intrinsics_dict = {'fx':cam_intrinsics[0] , 'fy':cam_intrinsics[4] , 'cx':cam_intrinsics[6] , 'cy':cam_intrinsics[7] }
    XYZ_m = convert_depth_pixel_to_metric_coordinate(depth_in, pixel_x, pixel_y, cam_intrinsics_dict)
    print(XYZ_m)
    print("CURENT STATUS:Move on to full_depth_frame instead of single pixel, backproject and visualize pcd ")

def rio_main_single_pcd(frame_id, seq_id, dataset_dir, features, debug=False):
    dataset_dir = Path("/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/")
    seq_path = Path("scene01/seq01/seq01_" + seq_id +"/")
    full_prefix_path = dataset_dir / seq_path

    model_path = dataset_dir / Path("scene01/models01/seq01_" + seq_id +"/")

    camera_file = Path(full_prefix_path, 'camera.yaml')
    pose_file   = Path(full_prefix_path, 'frame-{:06d}.pose.txt'.format(frame_id))
    rgb_file    = Path(full_prefix_path, 'frame-{:06d}.color.jpg'.format(frame_id))
    depth_file  = Path(full_prefix_path, 'frame-{:06d}.rendered.depth.png'.format(frame_id) )
    mesh_file   = Path(model_path ,"mesh.obj")


    assert camera_file.exists(), camera_file
    assert   pose_file.exists(), pose_file  
    assert    rgb_file.exists(), rgb_file   
    assert  depth_file.exists(), depth_file 
    assert   mesh_file.exists(), mesh_file
  
    rgb_img = read_image(rgb_file)
    # depth_img = read_image(depth_file, grayscale=True) #This is incorrectly loading the depth image, discretizing the values very randomly.
    depth_raw = o3d.io.read_image(str(depth_file))
    depth_img = np.asarray(depth_raw)


    plot_img_list = [rgb_img, depth_img]
    if debug:
        # print("Showing images")
        # plot_images(plot_img_list)
        # plt.show()
        print("Showing room mesh model now:")
        mesh_model = o3d.io.read_triangle_mesh(str(mesh_file), True)
        o3d.visualization.draw_geometries([mesh_model])
        sys.exit()

    K, img_size =  parse_camera_file_RIO(camera_file) 
    RT, RT_ctow = parse_pose_file_RIO(pose_file)
    RT_wtoc = RT

    # print(f"H & W: {img_size}, \n K:\n{K}, \n tf w to c:\n{RT} \n tf c to w:\n{RT_ctow} ")
    height, width = img_size

    pixel_x, pixel_y = 30, 70
    # print(depth_img[pixel_x:pixel_x+10, pixel_y:pixel_y+10])
    # print("npunique")
    # print(np.unique(depth_img))
    # sys.exit()
    # depth_in = depth_img[pixel_x, pixel_y]

    cam_intrinsics_dict = {'fx':K[0,0] , 'fy':K[1,1] , 'cx':K[0,2] , 'cy':K[1,2] }
    ## XYZ_onepoint = convert_depth_pixel_to_metric_coordinate(depth_in, pixel_x, pixel_y, cam_intrinsics_dict)
    # XYZ_full, RGB_full = convert_depth_frame_to_pointcloud(depth_img, rgb_img, cam_intrinsics_dict)
    # viz_with_array_inp(XYZ_full, RGB_full/255.0)
    print("TODO-Later-2: The above custom code for making pcd has color issue. Possibly some inversion. Correct later.")
    # print(XYZ_onepoint, (XYZ_full.shape))
    o3d_pcd, XYZ_o3d, RGB_o3d = o3d_convert_depth_frame_to_pointcloud(rgb_file, depth_file, cam_intrinsics_dict, img_size)
    if debug:
        viz_with_array_inp(XYZ_o3d, RGB_o3d, coords_bool=True)

    # taking to global frame
    o3d_pcd.transform(RT_ctow)

    return o3d_pcd, XYZ_o3d, RGB_o3d

if __name__ == '__main__':
    # Example arguments:
    # --dataset_dir ./datasets/inloc_small/ # This would be path of where `cutouts_imageonly` resides. 
    # --features ./outputs/inloc_small/feats-superpoint-n4096-r1600.h5 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--debug', dest='debug', default=False, action='store_true') # Just provide "--debug" on command line if you want to debug. Don't set it to anything.
    args = parser.parse_args()

    # main_using_p3p_pose(**args.__dict__)
    # main_using_o3dviz_pose(**args.__dict__)
    # main_exp_inloc(**args.__dict__)

    # CODE FOR SEQ_ID as 01
    seq_id = "01" 
    frame_ids_full = list(np.arange(0, 3000, 200)) #200 # 60
    frame_ids_small = [50, 131, 4318]

    frame_ids = frame_ids_full

    single_pcds = []
    for frame_id in frame_ids:
        o3d_pcd, XYZ_o3d, RGB_o3d = rio_main_single_pcd(frame_id, seq_id, **args.__dict__)
        single_pcds.append(o3d_pcd)
    merged_pcd = merged_pcd_from_single_pcds(single_pcds)
    viz_with_array_inp(np.asarray(merged_pcd.points), np.asarray(merged_pcd.colors), coords_bool=True)

    # CODE FOR SEQ_ID as 02
    seq_id = "02" 
    frame_ids_full = list(np.arange(0, 2000, 200)) #200 # 60
    frame_ids_small = [50, 131, 4318]

    frame_ids = frame_ids_full

    single_pcds = []
    for frame_id in frame_ids:
        o3d_pcd, XYZ_o3d, RGB_o3d = rio_main_single_pcd(frame_id, seq_id, **args.__dict__)
        single_pcds.append(o3d_pcd)
    merged_pcd_2 = merged_pcd_from_single_pcds(single_pcds)
    viz_with_array_inp(np.asarray(merged_pcd_2.points), np.asarray(merged_pcd_2.colors), coords_bool=True)

    # MERGING SEQ_ID 01 and 02
    merged_seq_pcd = merged_pcd_from_single_pcds([merged_pcd, merged_pcd_2])
    viz_with_array_inp(np.asarray(merged_seq_pcd.points), np.asarray(merged_seq_pcd.colors), coords_bool=True)