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

from .utils.parsers import parse_retrieval, names_to_pair


def interpolate_scan(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w-1, h-1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(
        scan, kp, align_corners=True, mode='bilinear')[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode='nearest')[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

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

def viz_cloud(mkp3d):
    print(f"mkp3d.shape {mkp3d.shape}")
    sys.exit()


def viz_entire_room(dataset_dir, r):
    # Load all the .jpg.mat files aka scan points of a particular room and visualize them
    room_path = (dataset_dir / "cutouts_imageonly/DUC1/024/")
    mat_files = sorted(list(room_path.glob('*.jpg.mat')))

    mat_files_small = mat_files[:6]#, mat_files[3]]

    pcds = []

    for mat_file in mat_files_small:
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

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    o3d.visualization.draw_geometries([pcd_final, mesh])
    
    sys.exit()
#    scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
#    mkp3d, valid = interpolate_scan(scan_r, mkpr)
#    Tr = get_scan_pose(dataset_dir, r)
#    mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T
#    print("DEBUG")



def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
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

        viz_entire_room(dataset_dir, r)
        scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]
        #print(f"mkpr, scan_r: {mkpr.shape} {scan_r.shape}")
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        Tr = get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T
        viz_cloud(mkp3d)

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
    #logging.info('Starting localization...')
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
