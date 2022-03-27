from pathlib import Path
import logging
import numpy as np
from collections import defaultdict
import yaml


def parse_image_list(path, with_intrinsics=False):
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            if with_intrinsics:
                camera_model, width, height, *params = data
                params = np.array(params, float)
                info = (camera_model, int(width), int(height), params)
                images.append((name, info))
            else:
                images.append(name)

    assert len(images) > 0
    logging.info(f'Imported {len(images)} images from {path.name}')
    return images

def parse_poses_from_file(path):
    poses = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            poses.append((name, data))

    assert len(poses) > 0
    logging.info(f'Imported {len(poses)} poses from {path.name}')
    return poses

def parse_image_lists(paths, with_intrinsics=False):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics)
    return images


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def parse_pose_file_RIO(pose_file):
    #Reading the individual pose file txt (RIO10 format: 4*4 matrix in 4 lines)
    with open(pose_file,'r') as f:
        pose_lines = f.readlines()
    # for row in pose_lines:
    #     print(row)
    pose_lines = [line.strip() for line in pose_lines]
    pose_lines = [line.split(' ') for line in pose_lines]
    pose_vals = [float(i) for line in pose_lines for i in line]
    RT_mat = np.array(pose_vals)
    RT_ctow = RT_mat.reshape((4,4))
    # NOTE: This RT is from  camera coordinate system to the world coordinate 
    # We want world to camera

    RT_wtoc = np.copy(RT_ctow)
    RT_wtoc[0:3,0:3] = RT_ctow[0:3,0:3].T
    RT_wtoc[0:3,3] = - RT_ctow[0:3,0:3].T @ RT_ctow[0:3,3]

    # print("DEBUG")
    # print(RT, RT_wtoc)
    RT_final = RT_wtoc

    # param.extrinsic = RT_final 
    # print(param.extrinsic)
    return RT_final, RT_ctow

def parse_camera_file_RIO(camera_file):
    with open(camera_file, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
        
    intrinsics = yaml_file['camera_intrinsics']
    img_size = (intrinsics['height'],intrinsics['width']) #H,W
    model = intrinsics['model']
    K = np.zeros((3,3))
    K[0,0] = model[0]
    K[1,1] = model[1]
    K[0,2] = model[2]
    K[1,2] = model[3]
    K[2,2] = 1
    # print("camera model:" ,model)
    # print("img size", img_size)
    # print(K)
    # #Set intrinsics here itself:
    # param = o3d.camera.PinholeCameraParameters()
    # intrinsic = param.intrinsic.set_intrinsics(width = img_size[1],
    #                                                 height = img_size[0],
    #                                                 fx = model[0],
    #                                                 fy = model[1],
    #                                                 cx = model[2],
    #                                                 cy = model[3])
    # # param.intrinsic = intrinsic
    # # print(img_size)
    # #print(param.intrinsic.intrinsic_matrix)
    return K, img_size