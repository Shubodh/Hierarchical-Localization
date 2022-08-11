#!/usr/bin/env python
# coding: utf-8
import h5py
from pathlib import Path
from pprint import pformat
import sys
from matplotlib import cm
import cv2
from tqdm import tqdm
import random
import numpy as np
import pickle
import time
import argparse
from scipy.spatial.transform import Rotation as R

# sys.path.append(str(Path(__file__).parent / '..')) #to import hloc
from hloc import extract_features, match_features, localize_rio, visualization

#import hloc
#print("carefully inspect which hloc it is, whether the docker one or normal modified one.")
#print(hloc)


from hloc.utils.parsers import parse_retrieval, names_to_pair, parse_pose_file_RIO, parse_camera_file_RIO

from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text, save_plot)
from hloc.utils.io import read_image
from hloc.utils.camera_projection_helper import output_global_scan_rio, reestimate_pose_using_3D_features_pcloc, moveback_tf_simple_given_pose #convert_depth_frame_to_pointcloud

# from .utils.io import read_image
# from .utils.parsers import parse_retrieval, names_to_pair, parse_pose_file_RIO, parse_camera_file_RIO
# from .utils.viz import plot_images, plot_matches, save_plot

import matplotlib.pyplot as plt

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

def ret_from_file(dataset_dir, r):
    fx, fy, cx, cy, height, width = cam_intrinsics_from_query_img(Path(dataset_dir), Path(r))
    ret = {}
    full_prefix_path = dataset_dir / Path(r).parents[0]
    r_stem = Path(r).stem.replace("color", "")
    pose_file  = Path(full_prefix_path, r_stem + 'pose.txt')
    assert  pose_file.exists(), pose_file 
    T_w2c, T_c2w = parse_pose_file_RIO(pose_file)

    qx_c, qy_c, qz_c, qw_c = R.from_matrix(T_w2c[0:3,0:3]).as_quat()
    tx_c, ty_c, tz_c = T_w2c[0:3,3]
    ret['qvec'] = np.array([qw_c, qx_c, qy_c, qz_c])
    ret['tvec'] = np.array([tx_c, ty_c, tz_c])
    cfg = {
        'model': 'PINHOLE', # PINHOLE, Also note: Try OPENCV uses distortion as well
        'width': width,
        'height': height,
        'params': [fx, fy, cx, cy]
    }
    ret['cfg'] = cfg

    return ret


def viz_every_query_by_3d_proj(save_pref_path, scene_id, dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    fx, fy, cx, cy, height, width = cam_intrinsics_from_query_img(Path(dataset_dir), Path(q))
    camera_parm = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    for i, r in enumerate(retrieved):
        on_ada = False
        viz_or_save_plots = True 

        r_ret = ret_from_file(dataset_dir, r)
        ret_new, mkpq, mkpr, mkp3d, indices, num_matches = reestimate_pose_using_3D_features_pcloc(
            dataset_dir, q, r_ret['qvec'], r_ret['tvec'], on_ada, scene_id, camera_parm, height, width)#, dataset_dir, q, db, feature_file, match_file, skip_matches)
        if ret_new['success']:
            print(ret_new['success'])
            print("success")
            # ret = ret_new
        else: 
            print("failure")
            sys.exit()
        # VISUALIZATION DEBUG:
        if viz_or_save_plots:
            print(f"Number of matches: {mkpq.shape[0]}, {q, r}")

            plot_images([read_image(dataset_dir / q), read_image(dataset_dir / r)])
            plot_matches(mkpq, mkpr)
            # pref_path = Path("outputs/graphVPR/rio_metric/viz_3d_expt/")
            pref_path = Path(save_pref_path)
            pref_name = "_3d_proj" + "_num_matches_" + str(mkpq.shape[0])
            path_sv =  pref_path / Path(dataset_dir.stem[:7] + "_q-" + Path(Path(q).stem).stem + "_r-" + Path(Path(r).stem).stem + pref_name+  ".png")
            save_plot(path_sv)
            print(f"saved correspondences plot at {path_sv}")

def viz_every_query(save_pref_path, dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
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
        # print(f"Number of matches: {mkpq.shape[0]}")

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        print(dataset_dir, Path(q).stem, Path(Path(r).stem).stem)
        print(f"Number of matches: {mkpq.shape[0]}, {q, r}")
        plot_images([read_image(dataset_dir / q), read_image(dataset_dir / r)])
        plot_matches(mkpq, mkpr)
        # viz_type = "projection_3d"
        # pref_path = Path("outputs/graphVPR/rio_metric/viz/" + viz_type + "/")
        # pref_path = Path("outputs/graphVPR/rio_metric/viz_3d_expt/")
        # pref_path = Path("outputs/graphVPR/rio_metric/viz_3d_expt/")
        pref_path = Path(save_pref_path)
        # pref_name = "_rend_ref"
        pref_name = "_ref" + "_num_matches_" + str(mkpq.shape[0])
        path_sv =  pref_path / Path(dataset_dir.stem[:7] + "_q-" + Path(Path(q).stem).stem + "_r-" + Path(Path(r).stem).stem + pref_name+  ".png")
        save_plot(path_sv)
        print(f"saved correspondences plot at {path_sv}")
        # sys.exit()

        # plt.show()


def main_viz_images_full_func(dataset_dir, retrieval, features, matches, results, scene_id, refine_pcloc=False,
         skip_matches=None):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    poses = {}
    # logs = {
    #     'features': features,
    #     'matches': matches,
    #     'retrieval': retrieval,
    #     'loc': {},
    # }
    for q in tqdm(queries):
        db = retrieval_dict[q]
        pref_path = Path("outputs/graphVPR/rio_metric/viz_3d_expt/")
        viz_every_query(pref_path, dataset_dir, q, db, feature_file, match_file, skip_matches)
        # viz_every_query_rend(dataset_dir, q, db, feature_file, match_file, skip_matches)

        viz_every_query_by_3d_proj(pref_path, scene_id, dataset_dir, q, db, feature_file, match_file, skip=None)




if __name__ == '__main__':
    '''
    This file takes
    INPUTS: txt file with (queries <> refs) for which you want plots of matches to be saved.    
    OUTPUT: Just saves the plots at particular location. (currently: outputs/graphVPR/rio_metric/viz_3d_expt/)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True) #example argument: --scene_id 05
    args = parser.parse_args()
    given_scene_id = str(args.scene_id)

    # dataset = Path('datasets/InLoc_like_RIO10/scene01_synth/')  # change this if your dataset is somewhere else
    # dataset = Path('datasets/InLoc_like_RIO10/scene'+ given_scene_id + '_viz/')  # change this if your dataset is somewhere else
    expt_name = "_small_for_3dproj_pkl" #"3d_expt"
    # dataset = Path('datasets/InLoc_like_RIO10/scene'+ given_scene_id + '_viz' + '_'+ expt_name + '/')  # change this if your dataset is somewhere else
    dataset = Path('datasets/InLoc_like_RIO10/scene'+ given_scene_id + expt_name +  '/')  # change this if your dataset is somewhere else

    expt_name_pname = "3d_expt_2" #""
    pairs = Path('pairs/graphVPR/rio_metric/') #'pairs/inloc/'
    # loc_pairs = pairs / 'bruteforce40_samply.txt'#_tiny_0 #_cheating  # bruteforce40_samply.txt #_tiny_0 # top 40 retrieved by NetVLAD #-minustop3rooms
    # loc_pairs = pairs / Path('bruteforce40_samply_viz_scene' +given_scene_id+ '.txt')#_tiny_0 #_cheating  # bruteforce40_samply.txt #_tiny_0 # top 40 retrieved by NetVLAD #-minustop3rooms
    loc_pairs = pairs / Path('bruteforce40_samply_viz_'+ expt_name_pname + '_' +'scene' +given_scene_id+ '.txt')#_tiny_0 #_cheating  # bruteforce40_samply.txt #_tiny_0 # top 40 retrieved by NetVLAD #-minustop3rooms

    outputs = Path('outputs/graphVPR/rio_metric/tiny/')  # where everything will be saved

    # Set config
    dt_time = 'dt040722-t0210'
    custom_info = '' #PINHOLE_cam
    feature_name  = 'superpoint_inloc'  # sift, superpoint_inloc, d2net-ss, netvlad
    matcher_name  = 'superglue' # NN-mutual, superglue
    skip_no = 20
    refine_pcloc =  True

    results = outputs / Path('RIO_hloc_LOCAL_TINY_' + custom_info + '_' + feature_name +'+' + matcher_name + '_skip' + str(skip_no) + '_' + dt_time + '.txt')  # the result file
    print(f"Starting localization on {dt_time}")

    # list the standard configurations available
    # print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    # print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


    # pick one of the configurations for extraction and matching
    # you can also simply write your own here!
    feature_conf = extract_features.confs[feature_name] # superpoint_inloc, d2net-ss, netvlad
    matcher_conf = match_features.confs[matcher_name] # superglue


    # ## Extract local features for database and query images
    feature_path = extract_features.main(feature_conf, dataset, outputs)
    print(feature_path)


    # ## Match the query images
    # Here we assume that the localization pairs are already computed using image retrieval (NetVLAD). To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.
    match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)
    print(match_path)

    # ## Localize!
    # Perform hierarchical localization using the precomputed retrieval and matches. Different from when localizing with Aachen, here we do not need a 3D SfM model here: the dataset already has 3D lidar scans. The file `InLoc_hloc_superpoint+superglue_netvlad40.txt` will contain the estimated query poses.

    main_viz_images_full_func( 
        dataset, loc_pairs, feature_path, match_path, results, given_scene_id, refine_pcloc,
        skip_matches=skip_no) #20. 10 is giving error currently, for 1 query, unable to find any matches > 20  # skip database images with too few matches
    # localize_rio.main(
    #     dataset, loc_pairs, feature_path, match_path, results, given_scene_id, refine_pcloc,
    #     skip_matches=skip_no) #20. 10 is giving error currently, for 1 query, unable to find any matches > 20  # skip database images with too few matches


    # ## Visualization
    # We parse the localization logs and for each query image plot matches and inliers with a few database images.
    #visualization.visualize_loc(results, dataset, n=1, top_k_db=1, seed=2)