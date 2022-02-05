import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import h5py
from scipy.io import loadmat
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


from .utils.parsers import parse_retrieval, names_to_pair
from .utils.open3d_helper import custom_draw_geometry, load_view_point
from .utils.camera_projection_helper import load_depth_to_scan

sys.path.append('../')
from p3p_view_synthesis_inverse_warping import viz_entire_room_by_registering

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



def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    #print(width, height)
    cx = .5 * width 
    cy = .5 * height
    focal_length = 4032. * 28. / 36.

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

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)


        viz_entire_room_by_registering(dataset_dir, r)
        scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
        # Note that width height of query different from reference
        #print(f"DEBUG 1:  width, height - {scan_r.shape, width, height}")
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        Tr = get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    cfg = {
        'model': 'SIMPLE_PINHOLE',
        'width': width,
        'height': height,
        'params': [focal_length, cx, cy]
    }
    ret = pycolmap.absolute_pose_estimation(
        all_mkpq, all_mkp3d, cfg, 48.00)
    ret['cfg'] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches



def pose_from_cluster_mp3d(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    fx, fy, cx, cy, width, height = 960, 960, 960.5, 540.5, 1920, 1080
    focal_length = fx
    #cx = .5 * width #TO-CHECK-1: Should cx be exactly half of width? Above it's not.
    #cy = .5 * height
    #focal_length = 4032. * 28. / 36.


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

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        #viz_entire_room_by_registering(dataset_dir, r)
        #scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
        depth_base_path = Path('/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/x-view-scratch/data_collection/x-view/mp3d/')
        r_split = re.split('/|.png', str(r))
        im_split = re.split('_', r_split[2])
        env_name, room_name, img_id = im_split[3], im_split[4], im_split[0]
        depth_fin_path = env_name + "/rooms/" + room_name + "/raw_data/" +  img_id + "_depth.png"
        depth_full_path = depth_base_path / depth_fin_path
        #samply_dep = Path('/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/datasets/graphVPR/room_level_localization_small/0_mp3d_8WUmhLawc2A/references/bathroom1/8_rgb_mp3d_8WUmhLawc2A_bathroom1_depth.png')
        file_json = depth_base_path / (env_name + "/rooms/" + room_name + "/poses_cleaned.json")
        with open(file_json, 'r') as f:
            poses = json.load(f)
        # pose_local_to_global_full
        rot = np.array(poses['rotation'][int(img_id)]).astype(np.float64)
        pos = np.array(poses['position'][int(img_id)]).astype(np.float64)
        #print("DeBuG")
        #pos, rot = lines[int(img_id)*2], lines[int(img_id)*2 + 1] #rot in quat format: x, y, z, w
        #print(im_split, pos, rot)
        #sys.exit()
        scan_r = load_depth_to_scan(depth_full_path, pos, rot)
        #print(f"DEBUG 3 {scan_r.shape}")
        #sys.exit()
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        #Tr = get_scan_pose(dataset_dir, r)
        #mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    cfg = {
        'model': 'SIMPLE_PINHOLE',
        'width': width,
        'height': height,
        'params': [focal_length, cx, cy]
    }
    ret = pycolmap.absolute_pose_estimation(
        all_mkpq, all_mkp3d, cfg, 48.00)
    ret['cfg'] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches

def kp_from_cluster(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    cx = .5 * width
    cy = .5 * height
    #focal_length = 4032. * 28. / 36.

    all_mkpq = []
    all_mkpr = []
    #all_mkp3d = []
    all_indices = []
    kpq = feature_file[q]['keypoints'].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]['matches0'].__array__()
        v = (m > -1)

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        #print(f"valid shape {valid.shape}, mkpq {kpq.shape} {kpr.shape}")
        #Tr = get_scan_pose(dataset_dir, r)
        #mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        #all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    #all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    #cfg = {
    #    'model': 'SIMPLE_PINHOLE',
    #    'width': width,
    #    'height': height,
    #    'params': [focal_length, cx, cy]
    #}
    #ret = pycolmap.absolute_pose_estimation(
    #    all_mkpq, all_mkp3d, cfg, 48.00)
    #ret['cfg'] = cfg
    return all_mkpq, all_mkpr, all_indices, num_matches


def main(dataset_dir, retrieval, features, matches, results,
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
        ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
            dataset_dir, q, db, feature_file, match_file, skip_matches)

        poses[q] = (ret['qvec'], ret['tvec'])
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
