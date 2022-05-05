import cv2
import glob
import argparse
import numpy as np
import open3d as o3d
import json
import sys
import os
import pickle
import torch
import pycolmap
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.utils import read_image as read_image_sg
        
sys.path.append('../../') #TODO-Later: Not a permanent solution, should fix imports later.
# from hloc.localize_rio import output_global_scan_rio
from hloc.utils.io import read_image as read_image_hloc
from hloc.utils.parsers import parse_pose_file_RIO, parse_camera_file_RIO
from hloc.utils.open3d_helper import viz_with_array_inp

from hloc.utils.refinements import Refinement_extended, Refinement_base

def get_depth_at_pixel(depth_frame, pixel_x, pixel_y):
    """
    Get the depth value at the desired image point

    Parameters:
    -----------
    depth_frame 	 : rs.frame()
                           The depth frame containing the depth information of the image coordinate
    pixel_x 	  	 	 : double
                           The x value of the image coordinate
    pixel_y 	  	 	 : double
                            The y value of the image coordinate

    Return:
    ----------
    depth value at the desired pixel

    """
    return depth_frame.as_depth_frame().get_distance(round(pixel_x), round(pixel_y))



def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, cam_intrinsics):
    """
    Convert the depth and image point information to metric coordinates

    Parameters:
    -----------
    depth 	 	 	 : double
                           The depth value of the image point
    pixel_x 	  	 	 : double
                           The x value of the image coordinate
    pixel_y 	  	 	 : double
                            The y value of the image coordinate
    cam_intrinsics : dict of camera intrinsic matrix: keys are 'fx', 'fy', 'cx', 'cy'
                        of the imager in whose coordinate system the depth_frame is computed

    Return:
    ----------
    X : double
        The x value in meters
    Y : double
        The y value in meters
    Z : double
        The z value in meters

    """
    fx, fy, cx, cy = cam_intrinsics['fx'], cam_intrinsics['fy'], cam_intrinsics['cx'], cam_intrinsics['cy']
    X = ((pixel_x - cx)*depth)/fx 
    Y = ((pixel_y - cy)*depth)/fy 
    return X, Y, depth



def convert_depth_frame_to_pointcloud(rgb_img,depth_image,  cam_intrinsics):
    """
    Convert the depthmap to a 3D point cloud

    Parameters:
    -----------
    depth_image 	 	 : numpy depth array 
                           The depth_frame containing the depth map
    rgb_image            : numpy corresponding rgb array 
    cam_intrinsics : dict of camera intrinsic matrix: keys are 'fx', 'fy', 'cx', 'cy'
                        of the imager in whose coordinate system the depth_frame is computed

    Return:
    ----------
    xyz : array
        The xyz values of the pointcloud in meters
    rgb : array
        The corresponding rgb values of the pointcloud 

    """
    
    fx, fy, cx, cy = cam_intrinsics['fx'], cam_intrinsics['fy'], cam_intrinsics['cx'], cam_intrinsics['cy']
    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - cx)/fx
    y = (v.flatten() - cy)/fy

    z = depth_image.flatten() / 1000 #Values of depth images are typically in millimetres. 
    x = np.multiply(x,z)
    y = np.multiply(y,z)

    rgb_img = rgb_img.reshape(-1, 3)

    # print("Your doubt of how to represent points with no depth values is here: Take a look")
    # print(x.shape)
    # x = x[np.nonzero(z)]
    # y = y[np.nonzero(z)]
    # z = z[np.nonzero(z)]
    # print(x.shape)
    # sys.exit()

    rgb  = rgb_img[np.nonzero(z)] 
    #NOTE-1: rgb currently has bug. Perhaps the order is wrong.
    # But not using it anywhere in the pipeline, so not fixing it.
    # RELEVANT: See NOTE-2 in p3p_view_synthesis_inverse_warping.py. Perhaps same issue.

    # print(x.shape, rgb.shape, rgb_img.shape)
    # sys.exit()

    xyz = np.zeros((x.shape[0], 3))
    xyz[:,0] = x
    xyz[:,1] = y
    xyz[:,2] = z
    # print(xyz.shape, xyz[:10], x[:10], y[:10], z[:10])
    # sys.exit()

    return xyz, rgb

def output_global_scan_rio(dataset_dir, r):
    #dataset_dir: datasets/InLoc_like_RIO10/scene01_synth 
    #r: database/cutouts/frame-001820.color.jpg
    full_prefix_path = dataset_dir / r.parents[0]
    r_stem = r.stem.replace("color", "")

    camera_file = Path(full_prefix_path, 'camera.yaml')
    pose_file   = Path(full_prefix_path, r_stem + 'pose.txt')
    rgb_file    = Path(full_prefix_path, r_stem + 'color.jpg')
    depth_file  = Path(full_prefix_path, r_stem + 'rendered.depth.png')

    assert camera_file.exists(), camera_file
    assert   pose_file.exists(), pose_file  
    assert    rgb_file.exists(), rgb_file   
    assert  depth_file.exists(), depth_file 

    rgb_img = read_image_hloc(rgb_file)
    depth_raw = o3d.io.read_image(str(depth_file))
    depth_img = np.asarray(depth_raw)

    K, img_size =  parse_camera_file_RIO(camera_file) 
    RT, RT_ctow = parse_pose_file_RIO(pose_file)
    RT_wtoc = RT
    # print(f"H & W: {img_size}, \n K:\n{K}, \n tf w to c:\n{RT} \n tf c to w:\n{RT_ctow} ")
    height, width = img_size
    cam_intrinsics_dict = {'fx':K[0,0] , 'fy':K[1,1] , 'cx':K[0,2] , 'cy':K[1,2] }

    XYZ, RGB_has_bug = convert_depth_frame_to_pointcloud(rgb_img, depth_img, cam_intrinsics_dict)
    # RGB_has_bug has issues currently. See the convert_depth_frame_to_pointcloud() for more info. 
    global_pcd = (RT_ctow[:3, :3] @ XYZ.T) + RT_ctow[:3, 3].reshape((3,1))
    global_pcd = global_pcd.T
    global_pcd = (global_pcd.reshape((height, width, 3)))
    debug = False
    if debug:
        viz_with_array_inp(XYZ, RGB_has_bug, coords_bool=True)

    return global_pcd 

def load_depth_to_scan(depth_path, pos, rot):
    # 1. Converts depth to scan
    # 2. bring it to global frame from perspective frame
    rotation_matrix = R.from_quat(rot).as_matrix()
    position = pos.reshape((3, 1))
    T = np.concatenate((rotation_matrix, position), axis=1)
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)

    fx, fy, cx, cy, width, height = 960, 960, 960.5, 540.5, 1920, 1080

    K = np.array([
    [fx, 0., cx, 0.],
    [0., fy, cy, 0.],
    [0., 0.,  1, 0],
    [0., 0., 0, 1]])

    xs = (np.linspace(0, width, width))
    ys = (np.linspace(0, height, height))
    xs, ys = np.meshgrid(xs, ys)

    r=png.Reader(filename= depth_path )
    pngdata_depth = r.asDirect()
    depth = np.vstack(map(np.uint16, list(pngdata_depth[2])))
    depth = depth/100.0

    xys =  np.vstack(((xs) * depth, (ys) * depth, depth, np.ones(depth.shape)))# -depth or depth in 3rd arg?
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    xyz = T @ xy_c0
    xyz = xyz.T[:,:3]

    xyz = xyz.reshape((height, width, 3))
    #print(f"DEBUG X: {xyz.shape}")
    return xyz


def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
    """
    Convert the world coordinate to a 2D image coordinate

    Parameters:
    -----------
    pointcloud 	 	 : numpy array with shape 3xN

    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

    Return:
    ----------
    x : array
        The x coordinate in image
    y : array
        The y coordiante in image

    """

    assert (pointcloud.shape[0] == 3)
    x_ = pointcloud[0,:]
    y_ = pointcloud[1,:]
    z_ = pointcloud[2,:]

    m = x_[np.nonzero(z_)]/z_[np.nonzero(z_)]
    n = y_[np.nonzero(z_)]/z_[np.nonzero(z_)]

    x = m*camera_intrinsics.fx + camera_intrinsics.cx
    y = n*camera_intrinsics.fy + camera_intrinsics.cy

    return x, y

def get_clipped_pointcloud(pointcloud, boundary):
    """
    Get the clipped pointcloud withing the X and Y bounds specified in the boundary
    
    Parameters:
    -----------
    pointcloud 	 	 : array
                           The input pointcloud which needs to be clipped
    boundary      : array
                                        The X and Y bounds 
    
    Return:
    ----------
    pointcloud : array
        The clipped pointcloud
    
    """
    assert (pointcloud.shape[0]>=2)
    pointcloud = pointcloud[:,np.logical_and(pointcloud[0,:]<boundary[1], pointcloud[0,:]>boundary[0])]
    pointcloud = pointcloud[:,np.logical_and(pointcloud[1,:]<boundary[3], pointcloud[1,:]>boundary[2])]
    return pointcloud

def convert_superglue_db_format(img_tensor, pred_q, pred_kpts, pred_desc, pred_score, device):


    pred_feat = {'keypoints': [torch.from_numpy(pred_kpts).to(device)],
                 'descriptors': [torch.from_numpy(pred_desc).to(device)],
                 'scores': [torch.from_numpy(pred_score).to(device)]}
    pred = {}
    pred = {**pred, **{k + '0': v for k, v in pred_q.items()}}
    pred = {**pred, **{k + '1': v for k, v in pred_feat.items()}}

    data = {'image0': img_tensor, 'image1': img_tensor, **pred}

    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])

    return data

def reestimate_pose_using_3D_features(dataset_dir, q, qvec, tvec, on_ada, scene_id, camera_parm, height, width):#, pcfeat_pth, camera_parm, max_keypoints):
    max_keypoints = 3000  #SuperPoint's default is -1. In hloc, we're using -1. In PCLoc, 3000.
    if on_ada:
        pcfeat_base_pth = "/scratch/saishubodh/PCLoc_saved/ICCV_TEST/pc_feats/"
        pc_idx = "pcfeat_scene" + str(scene_id) + ".pkl"
        pcfeat_pth = pcfeat_base_pth + pc_idx
        assert Path(pcfeat_pth).exists(), Path(pcfeat_pth)
        pcfeat_pth = [pcfeat_pth]
    else: #local_shub
        pcfeat_base_pth = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/outputs/rio/ICCV_TEST/pc_feats/"
        pc_idx = "pcfeat_scene" + str(scene_id) + ".pkl"
        pcfeat_pth = pcfeat_base_pth + pc_idx
        assert Path(pcfeat_pth).exists(), Path(pcfeat_pth)
        pcfeat_pth = [pcfeat_pth]

    # qvec, tvec i get from hloc pipeline is wtoc or w2c.
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Running inference on device \"{}\"'.format(device))
    config = {'superpoint': {'weights': 'indoor', # IMPORTANT: TO-CHECK-Later: hloc by default uses indoor weights, but PCLoc uses outdoor.
                            #'descriptor_dim': 256,
                            #'keypoint_encoder': [32, 64, 128, 256],
                            #'GNN_layers': ['self', 'cross'] * 9,
                            'sinkhorn_iterations': 100,# TO-CHECK-Later: hloc by default uses 100, but PCLoc uses 20.
                            'match_threshold': 0.2,
                             }}
    superglue = SuperGlue(config.get('superglue', {})).eval().to(device)
    refinement = Refinement_base(superglue) # Later: Add Refinement_extended, Extended PC (Div matching)

    RT_wtoc = np.zeros((4,4))
    RT_wtoc[3,3] = 1 
    qw, qx, qy, qz  = qvec 
    RT_wtoc[0:3,0:3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    RT_wtoc[0:3, 3] = tvec
    # RT_ctow = np.copy(RT_wtoc)
    # RT_ctow = np.linalg.inv(RT_wtoc)

    pred_pose = RT_wtoc


    # print(">> Pose Correction using the pre-estimated poses...")
    # refine_poses = []
    # num_inliers = []
    # for idx in tqdm(topk_inliers):
    #     topk_idx = clustered_frames[idx]
    #     pred_pose = pred_poses[idx]
    #     pcfeat_pth = pcfeat_list[topk_idx // 36]

    pred_kpts, pred_desc, pred_score, pred_xyz = scan2imgfeat_projection(pred_pose, pcfeat_pth, camera_parm, height, width, num_kpts=max_keypoints)
    # image0, inp0, scales0 = read_image_sg(cutout_pth[:-4], device, [1200], 0, False)
    # TO-CHECK-Later: (In case bad results) Look at 3rd param `resize` below. In PCLoc for InLoc, they use 1600.
    query_pth = str(dataset_dir / q)
    image0, inp0, scales0 = read_image_sg(query_pth, device, [-1], 0, False)
    # print(cutout_pth, "\n", image0, "\n", inp0, "\n", scales0)
    sg_config = {'superpoint': {'nms_radius': 4,
                             'keypoint_threshold': 0.005,
                             'max_keypoints': 3000  #SuperPoint's default is -1. In hloc, we're using -1. In PCLoc, 3000.
                             }}

    superpoint = SuperPoint(sg_config.get('superpoint', {})).eval().to(device)
    pred0 = superpoint({'image': inp0})

    data = convert_superglue_db_format(inp0, pred0, pred_kpts, pred_desc, pred_score, device)
    mkpts0, mkpts_xyz = refinement(data, pred_xyz) #mkpts0 is query keypoints

    fx, fy, cx, cy = camera_parm[0,0], camera_parm[1,1], camera_parm[0,2], camera_parm[1,2]
    cfg = {
        'model': 'PINHOLE', # PINHOLE, Also note: Try OPENCV uses distortion as well
        'width': width,
        'height': height,
        'params': [fx, fy, cx, cy]
    }
    ret = pycolmap.absolute_pose_estimation(
        mkpts0, mkpts_xyz, cfg, 48.00)
    ret['cfg'] = cfg

    # print("after")
    # print(mkpts0.shape, pred_kpts.shape, mkpts_xyz.shape, mkpts0.shape[0])
    # TO-CHECK-Later: Logs will be written WRONG. return ing wrong things.
    return ret, mkpts0, mkpts0, mkpts_xyz, mkpts0[:,0], mkpts0.shape[0]
    # return ret, all_mkpq,  all_mkpr, all_mkp3d, all_indices, num_matches
    # TO-CHECK-Later: Better way to code the above is to exclusively handle failure cases as below commented code:
#    if len(mkpts0) > 3:
#        result, inliers = do_pnp(mkpts0, mkpts_xyz, camera_parm, 0.00, reproj_error=args.reproj_err)
#        # cfg = {
#        #     'model': 'PINHOLE', # PINHOLE, Also note: Try OPENCV uses distortion as well
#        #     'width': width,
#        #     'height': height,
#        #     'params': [fx, fy, cx, cy]
#        # }
#
#    else:
#        result = loc_failure
#        T_w2c = pred_pose
#        result = LocResult(False, result.num_inliers, result.inlier_ratio, T_w2c)
#
#    if result.success:
#        T_c2w = result.T
#        T_w2c = np.linalg.inv(T_c2w)
#
#    else:
#        T_w2c = pred_pose
#
#    refine_poses.append(T_w2c)
#    num_inliers.append(result.num_inliers)


def scan2imgfeat_projection(pose, feat_pth, camera_parm, height, width, num_kpts=4096):
    # print("TO-CHECK-Later: Left this function as it is currently, inspect it later if results aren't good.")
    with open(feat_pth[0], 'rb') as handle:
        scan_feat = pickle.load(handle)
        scan_pts = scan_feat['ptcloud']
        scan_desc = scan_feat['descriptors']
        scan_score = scan_feat['scores']

    # img_size = camera_parm[:2, 2] * 2 # uv coordinate
    # H = int(img_size[1])
    # W = int(img_size[0])
    H, W = height, width
    # print(H, W)


    proj_mat = np.matmul(camera_parm, pose[:3, :])
    H_xyz = np.concatenate((scan_pts.T, np.ones((1, len(scan_pts)))), axis=0)

    uv = np.matmul(proj_mat, H_xyz)
    uv_norm = uv[2, :]
    uv = np.divide(uv[:2, :], uv_norm)

    front_idx = uv_norm > 0
    uv = uv[:, front_idx]
    desc = scan_desc[:, front_idx]
    score = scan_score[front_idx]
    ptcloud = scan_pts[front_idx, :]


    visible_idx = (uv[0, :] >= 0) & (uv[1, :] >= 0) & (uv[0, :] < W) & (uv[1,:] < H)
    uv = uv[:, visible_idx]
    desc = desc[:, visible_idx]
    score = score[visible_idx]
    ptcloud = ptcloud[visible_idx, :]

    score_idx = np.argsort(score)[::-1][:num_kpts]
    uv = uv[:, score_idx]
    desc = desc[:, score_idx]
    score = score[score_idx]
    ptcloud = ptcloud[score_idx, :]

    kpts = np.floor(uv.T).astype(np.float32)

    return kpts, desc, score, ptcloud

def synthesize_img_given_viewpoint_long(pcd, viewpoint_json):
    # colors = colors * 255
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    
    H = 1200
    W = 1600
    vpt_json = json.load(open(viewpoint_json))
    extrinsics = np.array(vpt_json['extrinsic']).reshape(4,4).T
    K = np.array(vpt_json['intrinsic']['intrinsic_matrix']).reshape(3,3).T
    print("K")
    print(K)

    rvecs = np.zeros(3)
    cv2.Rodrigues(extrinsics[0:3,0:3], rvecs)
    tvecs = np.zeros(3)
    tvecs = extrinsics[0:3,3] # - extrinsics[0:3,0:3].T @ extrinsics[0:3,3]

    print(f"rvecs, tvecs: {rvecs, tvecs}")
    dist = np.zeros(5)
    print("Starting cv2.project:")
    xyz_T = xyz.T
    xyz_hom1 = np.vstack((xyz_T, np.ones(xyz_T[0].shape)))
    K_hom = np.vstack((K, np.zeros(K[0].shape)))
    K_hom = np.hstack((K_hom, np.array([[0,0,0,1]]).T))

    #print("1. X_G visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))
    xyz_hom1 = np.matmul(extrinsics, xyz_hom1) #xyz_hom1.shape: 4 * 11520000
    #print("2. X_L visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))

    #print("3. X_L_corr visualization") # #tf_ex = np.array([[1,0,0],[0,-1,0],[0,0,-1]]), then np.vstack((tf_ex, np.zeros(tf_ex[0].shape))), then np.hstack((tf_ex_hom, np.array([[0,0,0,1]]).T)), then xyz_hom1 = np.matmul(tf_ex_hom, xyz_hom1)
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))

    xy_img = np.matmul(K_hom, xyz_hom1)
    #print(np.nanmax(xy_img[0:2,:]), np.nanmin(xy_img[0:2,:]))
    xy_img = xy_img[0:2,:] / xy_img[2:3,:] 
    #print(xy_imgcv.shape, xy_img.shape)
    xy_imgcv = np.array(xy_img.T, dtype = np.int_) #TODO: Relook, instead of int_, do upper or lower bound explicitly instead, something like: np.floor(uv.T).astype(np.float32)

    print("Done cv2.project:")

    # GETTING SAME RESULTS USING OPENCV TOO: see below old_code_dump_from_synthesize_img_given_viewpoint()
    synth_img = np.ones((H, W, 3))  * 255

    for i in range(colors.shape[0]):
        # Be careful here: For xy_imgcv, (x,y) means x right first then y down.
        # Whereas for numpy array, (x, y) means x down first then y right.

        # 1. Ignore points with negative depth, i.e. ones behind the camera. 
        if xyz_hom1[2,i] > 0: # Make sure the xyz you're checking are in ego frame
            # 2. projected pixel must be between  [{0,W},{0,H}]
            if (xy_imgcv[i,0] >= 0) & (xy_imgcv[i,0] < W):
                if (xy_imgcv[i,1] >= 0) &  (xy_imgcv[i,1] < H):
                    #print(xy_imgcv[i], i)
                    synth_img[xy_imgcv[i,1], xy_imgcv[i,0]] = colors[i] #


    img = o3d.geometry.Image((synth_img).astype(np.uint8))
    #o3d.visualization.draw_geometries([img])
    o3d.io.write_image(viewpoint_json + "synth.jpg", img)
    print(f"image written to {viewpoint_json}synth.jpg")

def synthesize_img_given_viewpoint_short(pcd, viewpoint_json):
    """ WARNING: This function yet to be tested.
    Wrote this taking inspiration from scan2imgfeat_projection() from PCLoc repo (see its utils.py).
    Much cleaner and you can see the steps one by one compared to original function.
    """
    # colors = colors * 255
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    
    H = 1200
    W = 1600
    vpt_json = json.load(open(viewpoint_json))
    extrinsics = np.array(vpt_json['extrinsic']).reshape(4,4).T
    K = np.array(vpt_json['intrinsic']['intrinsic_matrix']).reshape(3,3).T
    print("K")
    print(K)

    rvecs = np.zeros(3)
    cv2.Rodrigues(extrinsics[0:3,0:3], rvecs)
    tvecs = np.zeros(3)
    tvecs = extrinsics[0:3,3] # - extrinsics[0:3,0:3].T @ extrinsics[0:3,3]

    print(f"rvecs, tvecs: {rvecs, tvecs}")
    dist = np.zeros(5)
    print("Starting cv2.project:")
    xyz_T = xyz.T
    xyz_hom1 = np.vstack((xyz_T, np.ones(xyz_T[0].shape)))
    K_hom = np.vstack((K, np.zeros(K[0].shape)))
    K_hom = np.hstack((K_hom, np.array([[0,0,0,1]]).T))

    #print("1. X_G visualization") #print("2. X_L visualization")#print("3. X_L_corr visualization") # #tf_ex = np.array([[1,0,0],[0,-1,0],[0,0,-1]]), then np.vstack((tf_ex, np.zeros(tf_ex[0].shape))), then np.hstack((tf_ex_hom, np.array([[0,0,0,1]]).T)), then xyz_hom1 = np.matmul(tf_ex_hom, xyz_hom1)
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))

    # MAIN PART OF THIS FUNCTION STARTS HERE: (For direct OpenCV function, see below old_code_dump_from_synthesize_img_given_viewpoint())
    # 1. Camera projection and normalization
    xyz_hom1 = np.matmul(extrinsics, xyz_hom1) #xyz_hom1.shape: 4 * 11520000
    xy_img = np.matmul(K_hom, xyz_hom1)
    xy_norm = xy_img[2, :]
    xy_img = np.divide(xy_img[0:2,:], xy_norm)

    # 2. Ignore points with negative depth, i.e. ones behind the camera. 
    positive_depth_points = xy_norm > 0
    xy_img = xy_img[:, positive_depth_points]
    colors = colors[:, positive_depth_points]

    # 3. projected pixel must be between [{0,W},{0,H}]
    within_boundary = (xy_img[0,:] >= 0) & (xy_img[1,:] >= 0) & (xy_img[0,:] < W) & (xy_img[1,:] < H)
    xy_img = xy_img[:, within_boundary]
    colors = colors[:, within_boundary]

    xy_imgcv = np.array(xy_img.T, dtype = np.int_) #TODO: Relook, instead of int_, do upper or lower bound explicitly instead, something like: np.floor(uv.T).astype(np.float32)

    print("Done cv2.project:")
    synth_img = np.ones((H, W, 3))  * 255
    for i in range(colors.shape[0]):
        synth_img[xy_imgcv[i,1], xy_imgcv[i,0]] = colors[i] #
        # Be careful here: For xy_imgcv, (x,y) means x right first then y down.
        # Whereas for numpy array, (x, y) means x down first then y right.

    img = o3d.geometry.Image((synth_img).astype(np.uint8))
    #o3d.visualization.draw_geometries([img])
    o3d.io.write_image(viewpoint_json + "synth.jpg", img)
    print(f"image written to {viewpoint_json}synth.jpg")


#def old_code_dump_from_synthesize_img_given_viewpoint():
#    cx = .5 * W 
#    cy = .5 * H
#    focal_length = 4032. * 28. / 36.
#    K[0][0], K[1][1] = focal_length, focal_length
#    K[0][2] = cx
#    K[1][2] = cy
#    print("K after")
#    print(K)
    #H = int(vpt_json['intrinsic']['height'])
    #W = int(vpt_json['intrinsic']['width'])
    #cv2.Rodrigues(extrinsics[0:3,0:3].T, rvecs)
    #tvecs = - extrinsics[0:3,0:3].T @ extrinsics[0:3,3]

    # GETTING SAME RESULTS USING OPENCV TOO:
    #xy_imgcv, jac = cv2.projectPoints(xyz, rvecs, tvecs, K, dist)
    #xy_imgcv = np.array(xy_imgcv.reshape(xy_imgcv.shape[0], 2), dtype=np.int_)

    #print(xy_imgcv.shape, xy_imgcv_n.shape)
    #print(np.max(xy_imgcv), np.min(xy_imgcv), xy_imgcv)

#    W_valid = (xy_imgcv[:,0] >= 0) &  (xy_imgcv[:,0] < W)
#    H_valid = (xy_imgcv[:,1] >= 0) &  (xy_imgcv[:,1] < H)
#    #print(xy_imgcv[:,0].shape,"hi", np.nanmax(xy_imgcv, axis=0), "hii", xy_imgcv[0:10])
#    #print(xy_imgcv.shape)
#    final_valid = (H_valid  & W_valid)
#    #print(xy_imgcv[final_valid])
#    print(np.nanmax(xy_imgcv[final_valid], axis=0))
#    #print(np.argwhere(final_valid==False))
#
    # 2. habitat script grid_sample method

#    colors = np.array(np.asarray(pcd.colors) , dtype=np.single)
#    colors_re = colors.reshape((H,W,3))
#
#    xyz_T = xyz.T
#    xyz_hom0 = np.vstack((xyz_T, np.ones(xyz_T[0].shape)))
#    K_hom = np.vstack((K, np.zeros(K[0].shape)))
#    K_hom = np.hstack((K_hom, np.array([[0,0,0,1]]).T))
#
#    xyz_hom1 = np.matmul(extrinsics, xyz_hom0)
#    xy_img = np.matmul(K_hom, xyz_hom1)
#    xy_img = xy_img[0:2,:] / xy_img[2:3,:] #TXDX: Check if minus - should be there before xy_img[2:3,:].
#    #xy_img[1] *= -1
#
##    print(xy_img.shape, "bii", np.nanmax(xy_img, axis=1),  xy_img[:, 0:10])
#
#    print(xy_img)
#    sampler = torch.Tensor(xy_img).view(2, H, W).permute(1,2,0).unsqueeze(0)
#    print("HI", sampler)
#    #print(colors_re.shape, xy_img.shape)
#    # Create generated image
#    img2_tensor = ToTensor()(colors_re).unsqueeze(0)
#    img2_warped = F.grid_sample(img2_tensor, sampler)
#    
#    #print(img2_tensor[:,:,10], img2_warped[:,:,10])
#
#    # Visualise
#    #plt.figure(figsize=(10,10))
#    #ax1 = plt.subplot(221)
#    #ax1.imshow(img2_tensor.squeeze().permute(1,2,0))
#    #ax1.set_title("View 2", fontsize='large')
#    #ax1.axis('off')
#    #ax1 = plt.subplot(222)
#    #plt.imshow(img2_warped.squeeze().permute(1,2,0))
#    #ax1.set_title("View 2 warped into View 1 \n according to the estimated transformation", fontsize='large')
#    #ax1.axis('off')
#
def backprojection_to_3D_features_and_save_rio(save_dir, db_dir, scene_id):
    # local_feat_dir = os.path.join(save_dir, 'local_feats')
    # if not os.path.exists(local_feat_dir): os.makedirs(local_feat_dir)
    pc_feat_dir = os.path.join(save_dir, 'pc_feats')
    if not os.path.exists(pc_feat_dir): os.makedirs(pc_feat_dir)

    print(">> [Database] Local Feature Generation...")
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {'superpoint': {'nms_radius': 4,
                             'keypoint_threshold': 0.005,
                             'max_keypoints': 3000  #SuperPoint's default is -1. In hloc, we're using -1. In PCLoc, 3000.
                             }}

    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)

    feat_idx = 0
    cutout_list_rgb = glob.glob(os.path.join(db_dir, 'database/cutouts/*.color.jpg'))
    cutout_list_rgb.sort()

    scan_desc = []
    scan_score = []
    scan_xyz = []

    for cutout_pth in tqdm(cutout_list_rgb):

        # image0, inp0, scales0 = read_image_sg(cutout_pth[:-4], device, [1200], 0, False)
        # TO-CHECK-Later: (In case bad results) Look at 3rd param `resize` below. In PCLoc for InLoc, they use 1600.
        image0, inp0, scales0 = read_image_sg(cutout_pth, device, [-1], 0, False)
        # print(cutout_pth, "\n", image0, "\n", inp0, "\n", scales0)

        # scan_data = io.loadmat(cutout_pth)
        # xyz = scan_data['XYZcut']
        r = Path(*Path(cutout_pth).parts[-3:])
        xyz = output_global_scan_rio(Path(db_dir), Path(r)) #equivalent to # scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]

        pred = superpoint({'image': inp0})

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        keypoints = (pred['keypoints'] * scales0).astype(int)
        kpts_xyz = xyz[keypoints[:, 1], keypoints[:, 0], :]
        H_kpts = np.concatenate((kpts_xyz.T, np.ones((1, len(kpts_xyz)))), axis=0)
        # align_xyz = np.matmul(P_after, H_kpts) #For InLoc, not relevant for RIO10
        align_xyz = H_kpts
        align_xyz = np.divide(align_xyz[:3, :], align_xyz[3, :]).T

        nan_idx = ~np.isnan(align_xyz).all(axis=1)

        rm_keypoints = keypoints[nan_idx, :]
        rm_descrptors = pred['descriptors'][:, nan_idx]
        rm_scores = pred['scores'][nan_idx]
        rm_xyz = align_xyz[nan_idx, :]


        feat_cutout = dict()
        feat_cutout['keypoints'] = rm_keypoints
        feat_cutout['scores'] = rm_scores
        feat_cutout['descriptors'] = rm_descrptors
        feat_cutout['pts_xyz'] = rm_xyz

        # Uncommmet bcz not saving local features. Just saving PCLoc 3D features.
        # save_cutout_feat_fname = os.path.join(local_feat_dir, 'local_feat_{:05}.pkl'.format(feat_idx))
        # with open(save_cutout_feat_fname, 'wb') as handle:
        #     pickle.dump(feat_cutout, handle, protocol=pickle.HIGHEST_PROTOCOL)
        feat_idx += 1

        scan_desc.append(rm_descrptors)
        scan_score.append(rm_scores)
        scan_xyz.append(rm_xyz)

    total_score = np.concatenate(scan_score, 0)
    total_desc = np.concatenate(scan_desc, 1)
    total_xyz = np.concatenate(scan_xyz, 0)

    pc_feat = dict()
    pc_feat['ptcloud'] = total_xyz
    pc_feat['descriptors'] = total_desc
    pc_feat['scores'] = total_score

    pc_idx = "pcfeat_scene" + str(scene_id) + ".pkl"
    # save_pcfeat_fname = os.path.join(pc_feat_dir, 'pcfeat_{:05}.pkl'.format(pc_idx))
    save_pcfeat_fname = os.path.join(pc_feat_dir, pc_idx)
    with open(save_pcfeat_fname, 'wb') as handle:
        pickle.dump(pc_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # pc_idx += 1

    print(">> Save Local Feature and PC Feature Completed...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # local shub #db_dir should ls = database/ query/
    # --db_dir /media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/datasets/InLoc_like_RIO10/scene01/ --save_dir /media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/outputs/rio/ICCV_TEST/

    # on ADA #db_dir should ls = database/ query/
    # --db_dir /data/InLoc_like_RIO10/sampling10/scene01_JUST/ --save_dir /data/InLoc_dataset/outputs/rio/ICCV_TEST --scene_id 01
    parser.add_argument('--db_dir',   help='Path to Inloc dataset', required=True)
    parser.add_argument('--save_dir', help='Path to save database features (Output)', required=True)
    parser.add_argument('--scene_id', help='scene_id, ex: 01', required=True)
    args = parser.parse_args()
    backprojection_to_3D_features_and_save_rio(args.save_dir, args.db_dir, args.scene_id)
