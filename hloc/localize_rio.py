import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import logging
import pickle
import cv2
import pycolmap
import sys
import png
import re
from scipy.spatial.transform import Rotation as R
import json


from .utils.open3d_helper import custom_draw_geometry, load_view_point, viz_with_array_inp
from .utils.camera_projection_helper import output_global_scan_rio, reestimate_pose_using_3D_features_pcloc #convert_depth_frame_to_pointcloud
from .utils.parsers import parse_retrieval, names_to_pair, parse_pose_file_RIO, parse_camera_file_RIO
from .utils.io import read_image
from .utils.viz import plot_images, plot_matches, save_plot

# sys.path.append('../')
# from p3p_view_synthesis_inverse_warping import viz_entire_room_by_registering


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


def get_scan_pose(dataset_dir, rpath):
    split_image_rpath = rpath.split('/')
    floor_name = split_image_rpath[-3]
    scan_id = split_image_rpath[-2]
    image_name = split_image_rpath[-1]
    building_name = image_name[:3]

    path = Path(
        dataset_dir, 'database/alignments', floor_name,
        f'transformations/{building_name}_trans_{scan_id}.txt')
    with open(path) as f:
        raw_lines = f.readlines()

    P_after_GICP = np.array([
        np.fromstring(raw_lines[7], sep=' '),
        np.fromstring(raw_lines[8], sep=' '),
        np.fromstring(raw_lines[9], sep=' '),
        np.fromstring(raw_lines[10], sep=' ')
    ])

    return P_after_GICP

# def output_global_scan_rio(dataset_dir, r):
#     #dataset_dir: datasets/InLoc_like_RIO10/scene01_synth 
#     #r: database/cutouts/frame-001820.color.jpg
#     full_prefix_path = dataset_dir / r.parents[0]
#     r_stem = r.stem.replace("color", "")

#     camera_file = Path(full_prefix_path, 'camera.yaml')
#     pose_file   = Path(full_prefix_path, r_stem + 'pose.txt')
#     rgb_file    = Path(full_prefix_path, r_stem + 'color.jpg')
#     depth_file  = Path(full_prefix_path, r_stem + 'rendered.depth.png')

#     assert camera_file.exists(), camera_file
#     assert   pose_file.exists(), pose_file  
#     assert    rgb_file.exists(), rgb_file   
#     assert  depth_file.exists(), depth_file 

#     rgb_img = read_image(rgb_file)
#     depth_raw = o3d.io.read_image(str(depth_file))
#     depth_img = np.asarray(depth_raw)

#     K, img_size =  parse_camera_file_RIO(camera_file) 
#     RT, RT_ctow = parse_pose_file_RIO(pose_file)
#     RT_wtoc = RT
#     # print(f"H & W: {img_size}, \n K:\n{K}, \n tf w to c:\n{RT} \n tf c to w:\n{RT_ctow} ")
#     height, width = img_size
#     cam_intrinsics_dict = {'fx':K[0,0] , 'fy':K[1,1] , 'cx':K[0,2] , 'cy':K[1,2] }

#     XYZ, RGB_has_bug = convert_depth_frame_to_pointcloud(rgb_img, depth_img, cam_intrinsics_dict)
#     # RGB_has_bug has issues currently. See the convert_depth_frame_to_pointcloud() for more info. 
#     global_pcd = (RT_ctow[:3, :3] @ XYZ.T) + RT_ctow[:3, 3].reshape((3,1))
#     global_pcd = global_pcd.T
#     global_pcd = (global_pcd.reshape((height, width, 3)))
#     debug = False
#     if debug:
#         viz_with_array_inp(XYZ, RGB_has_bug, coords_bool=True)

#     return global_pcd 

def cam_intrinsics_from_query_img(dataset_dir, q):
    full_prefix_path = dataset_dir / q.parents[0]
    # print(full_prefix_path)

    camera_file = Path(full_prefix_path, 'camera.yaml')
    assert camera_file.exists(), camera_file

    K, img_size =  parse_camera_file_RIO(camera_file) 
    height, width = img_size
    cam_intrinsics_dict = {'fx':K[0,0] , 'fy':K[1,1] , 'cx':K[0,2] , 'cy':K[1,2] }

    fx, fy, cx, cy = cam_intrinsics_dict['fx'],cam_intrinsics_dict['fy'],cam_intrinsics_dict['cx'],cam_intrinsics_dict['cy']
    return fx, fy, cx, cy, height, width

def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    # height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    # #print(width, height)
    # cx = .5 * width 
    # cy = .5 * height
    # focal_length = 4032. * 28. / 36.
    fx, fy, cx, cy, height, width = cam_intrinsics_from_query_img(Path(dataset_dir), Path(q))
    # print(fx, fy, cx, cy, height, width)
    focal_length = fx

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []
    kpq = feature_file[q]['keypoints'].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]['matches0'].__array__()
        v = (m > -1)

        # Uncomment below if code is stopping. Likely because of number of correspondences < threshold.
        #print(f"No of correspondences: {np.count_nonzero(v), q, r}")
        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        # VISUALIZATION DEBUG:
        viz_or_save_plots = False
        if viz_or_save_plots:
            # print(dataset_dir, Path(q).stem, Path(Path(r).stem).stem)
            # print(f"Number of matches: {mkpq.shape[0]}")

            plot_images([read_image(dataset_dir / q), read_image(dataset_dir / r)])
            plot_matches(mkpq, mkpr)
            pref_path = Path("outputs/graphVPR/rio_metric/viz/")
            path_sv =  pref_path / Path(dataset_dir.stem[:7] + "_q-" + Path(Path(q).stem).stem + "_r-" + Path(Path(r).stem).stem +  ".png")
            save_plot(path_sv)
            # print(f"saved correspondences plot at {path_sv}")

            # plt.show()

        # viz_entire_room_by_registering(dataset_dir, r)
        # scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
        scan_r = output_global_scan_rio(Path(dataset_dir), Path(r))
        # Note that width height of query different from reference
        #print(f"DEBUG 1:  width, height - {scan_r.shape, width, height}")
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        #### Tr = get_scan_pose(dataset_dir, r) #Already in global frame. This was needed for InLoc to take it from room -> global (there were 3 there: local, room, global)
        #### mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    if len(all_mkpq) == 0:
        T_w2c = np.array([[1.0, 0.0, 0.0, 1000.0],
                        [0.0, 1.0, 0.0, 1000.0],
                        [0.0, 0.0, 1.0, 1000.0],
                        [0.0, 0.0, 0.0, 1.0]])
        qx_c, qy_c, qz_c, qw_c = R.from_matrix(T_w2c[0:3,0:3]).as_quat()
        tx_c, ty_c, tz_c = T_w2c[0:3,3]
        ret = {'success': False}
        ret['qvec'] = np.array([qw_c, qx_c, qy_c, qz_c])
        ret['tvec'] = np.array([tx_c, ty_c, tz_c])
        cfg = {
            'model': 'PINHOLE', # PINHOLE, Also note: Try OPENCV uses distortion as well
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy]
        }
        ret['cfg'] = cfg
        #print(ret)
        #sys.exit()
    else:
        all_mkpq = np.concatenate(all_mkpq, 0)
        all_mkpr = np.concatenate(all_mkpr, 0)
        all_mkp3d = np.concatenate(all_mkp3d, 0)
        all_indices = np.concatenate(all_indices, 0)

        # cfg = {
        #     'model': 'SIMPLE_PINHOLE',
        #     'width': width,
        #     'height': height,
        #     'params': [focal_length, cx, cy]
        # }

        #NOTE-3: using focal_length fx, fy currently. Also NOT using distortion params. (pycolmap allows it) Look at 'OPENCV' model in https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
        cfg = {
            'model': 'PINHOLE', # PINHOLE, Also note: Try OPENCV uses distortion as well
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy]
        }
        ret = pycolmap.absolute_pose_estimation(
            all_mkpq, all_mkp3d, cfg, 48.00)
        ret['cfg'] = cfg
        # print('hi bro')
        # print(ret)
        # print(all_mkpq.shape, all_mkpr.shape, all_mkp3d.shape, all_indices.shape, num_matches)
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches


def pose_from_cluster_tf_idea_simple(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    """
    tf_idea_simple Idea: Apply tf to ref images to go back
    Steps: 1. NetVLAD top40 -> Apply -1m tf (or places - center to pose idea) -> these are places
    2. With 3D features for whole world beforehand, project 3D features on places
    3. now matching query <> places
    """
    # height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    # #print(width, height)
    # cx = .5 * width 
    # cy = .5 * height
    # focal_length = 4032. * 28. / 36.
    fx, fy, cx, cy, height, width = cam_intrinsics_from_query_img(Path(dataset_dir), Path(q))
    # print(fx, fy, cx, cy, height, width)
    focal_length = fx

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []
    kpq = feature_file[q]['keypoints'].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        print("debuggg")
        print(r)
        sys.exit()
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]['matches0'].__array__()
        v = (m > -1)

        # Uncomment below if code is stopping. Likely because of number of correspondences < threshold.
        #print(f"No of correspondences: {np.count_nonzero(v), q, r}")
        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        # VISUALIZATION DEBUG:
        viz_or_save_plots = False
        if viz_or_save_plots:
            # print(dataset_dir, Path(q).stem, Path(Path(r).stem).stem)
            # print(f"Number of matches: {mkpq.shape[0]}")

            plot_images([read_image(dataset_dir / q), read_image(dataset_dir / r)])
            plot_matches(mkpq, mkpr)
            pref_path = Path("outputs/graphVPR/rio_metric/viz/")
            path_sv =  pref_path / Path(dataset_dir.stem[:7] + "_q-" + Path(Path(q).stem).stem + "_r-" + Path(Path(r).stem).stem +  ".png")
            save_plot(path_sv)
            # print(f"saved correspondences plot at {path_sv}")

            # plt.show()

        # viz_entire_room_by_registering(dataset_dir, r)
        # scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
        scan_r = output_global_scan_rio(Path(dataset_dir), Path(r))
        # Note that width height of query different from reference
        #print(f"DEBUG 1:  width, height - {scan_r.shape, width, height}")
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        #### Tr = get_scan_pose(dataset_dir, r) #Already in global frame. This was needed for InLoc to take it from room -> global (there were 3 there: local, room, global)
        #### mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    if len(all_mkpq) == 0:
        T_w2c = np.array([[1.0, 0.0, 0.0, 1000.0],
                        [0.0, 1.0, 0.0, 1000.0],
                        [0.0, 0.0, 1.0, 1000.0],
                        [0.0, 0.0, 0.0, 1.0]])
        qx_c, qy_c, qz_c, qw_c = R.from_matrix(T_w2c[0:3,0:3]).as_quat()
        tx_c, ty_c, tz_c = T_w2c[0:3,3]
        ret = {'success': False}
        ret['qvec'] = np.array([qw_c, qx_c, qy_c, qz_c])
        ret['tvec'] = np.array([tx_c, ty_c, tz_c])
        cfg = {
            'model': 'PINHOLE', # PINHOLE, Also note: Try OPENCV uses distortion as well
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy]
        }
        ret['cfg'] = cfg
        #print(ret)
        #sys.exit()
    else:
        all_mkpq = np.concatenate(all_mkpq, 0)
        all_mkpr = np.concatenate(all_mkpr, 0)
        all_mkp3d = np.concatenate(all_mkp3d, 0)
        all_indices = np.concatenate(all_indices, 0)

        # cfg = {
        #     'model': 'SIMPLE_PINHOLE',
        #     'width': width,
        #     'height': height,
        #     'params': [focal_length, cx, cy]
        # }

        #NOTE-3: using focal_length fx, fy currently. Also NOT using distortion params. (pycolmap allows it) Look at 'OPENCV' model in https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
        cfg = {
            'model': 'PINHOLE', # PINHOLE, Also note: Try OPENCV uses distortion as well
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy]
        }
        ret = pycolmap.absolute_pose_estimation(
            all_mkpq, all_mkp3d, cfg, 48.00)
        ret['cfg'] = cfg
        # print('hi bro')
        # print(ret)
        # print(all_mkpq.shape, all_mkpr.shape, all_mkp3d.shape, all_indices.shape, num_matches)
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches


def main(dataset_dir, retrieval, features, matches, results, scene_id, refine_pcloc=False,
         skip_matches=None):

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logging.info('Starting localization...')
    for q in tqdm(queries):
        db = retrieval_dict[q]
        #ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
        #    dataset_dir, q, db, feature_file, match_file, skip_matches)
        ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster_tf_idea_simple(
            dataset_dir, q, db, feature_file, match_file, skip_matches)


        # print(ret)
        # refine_pcloc = False
        on_ada = True
        if refine_pcloc:
            fx, fy, cx, cy, height, width = cam_intrinsics_from_query_img(Path(dataset_dir), Path(q))
            camera_parm = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            #ret, mkpq, mkpr, mkp3d, indices, num_matches = reestimate_pose_using_3D_features_pcloc(
                # ret['qvec'], ret['tvec'])#, dataset_dir, q, db, feature_file, match_file, skip_matches)
            ret_new, mkpq, mkpr, mkp3d, indices, num_matches = reestimate_pose_using_3D_features_pcloc(
                dataset_dir, q, ret['qvec'], ret['tvec'], on_ada, scene_id, camera_parm, height, width)#, dataset_dir, q, db, feature_file, match_file, skip_matches)
            if ret_new['success']:
                # print(ret_new['success'])
                ret = ret_new
            #else: ret = ret


        # print(mkpq.shape, mkpr.shape, mkp3d.shape, indices.shape, num_matches)
        #sys.exit()
        #print(ret)

        poses[q] = (ret['qvec'], ret['tvec']) #pycolmap's quaternion convention is: w x y z
        logs['loc'][q] = {
            'db': db,
            'PnP_ret': ret,
            'keypoints_query': mkpq,
            'keypoints_db': mkpr,
            '3d_points': mkp3d,
            'indices_db': indices,
            'num_matches': num_matches,
        }

    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in queries:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split("/")[-1]
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')

def main_only_kp(dataset_dir, retrieval, features, matches, results,
         skip_matches=None):
    '''Only extract keypoints and matches. Ignore poses, 3D points etc.
    If you want all, see `main` function''' 

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

#    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logging.info('Starting localization...')
    for q in tqdm(queries):
        db = retrieval_dict[q]
        mkpq, mkpr, indices, num_matches = kp_from_cluster(
            dataset_dir, q, db, feature_file, match_file, skip_matches)

 #       poses[q] = (ret['qvec'], ret['tvec'])
        logs['loc'][q] = {
            'db': db,
#            'PnP_ret': ret,
            'keypoints_query': mkpq,
            'keypoints_db': mkpr,
#            '3d_points': mkp3d,
            'indices_db': indices,
            'num_matches': num_matches,
        }

#    logging.info(f'Writing everythig EXCEPT poses to {results}...')
#    with open(results, 'w') as f:
#        for q in queries:
#            qvec, tvec = poses[q]
#            qvec = ' '.join(map(str, qvec))
#            tvec = ' '.join(map(str, tvec))
#            name = q.split("/")[-1]
#            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--skip_matches', type=int)
    args = parser.parse_args()
    main(**args.__dict__)
