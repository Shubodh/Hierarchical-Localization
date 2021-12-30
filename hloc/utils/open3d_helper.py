import numpy as np
import open3d as o3d
import json
import sys

import matplotlib.pyplot as plt

import cv2

import torch.nn.functional as F
import torch
from torchvision.transforms import ToTensor

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
        vis.create_window()
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(filename)
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()

    vis.capture_screen_image(filename+".png")
    #vis.capture_screen_image(filename+".png", do_render=True)
    print(f"visualizer image saved at {filename}.png")
    vis.destroy_window()


def synthesize_img_given_viewpoint(pcd, viewpoint_json):
    H = 1200
    W = 1600
    vpt_json = json.load(open(viewpoint_json))
    extrinsics = np.array(vpt_json['extrinsic']).reshape(4,4).T
    K = np.array(vpt_json['intrinsic']['intrinsic_matrix']).reshape(3,3).T
    print("K")
    print(K)
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

    xyz = np.asarray(pcd.points)
    

#    print(f"xyz, extrinsics {xyz.shape}{extrinsics.shape}")
#    xyz_T = np.matmul(extrinsics[0:3,0:3], xyz.T)
#    xyz = xyz_T.T
    rvecs = np.zeros(3)
    cv2.Rodrigues(extrinsics[0:3,0:3], rvecs)
    #cv2.Rodrigues(extrinsics[0:3,0:3].T, rvecs)
    tvecs = np.zeros(3)
    tvecs = extrinsics[0:3,3]
    #tvecs = - extrinsics[0:3,0:3].T @ extrinsics[0:3,3]


    print(f"rvecs, tvecs: {rvecs, tvecs}")
    dist = np.zeros(5)
    print("Starting cv2.project:")
    xyz_T = xyz.T
    xyz_hom1 = np.vstack((xyz_T, np.ones(xyz_T[0].shape)))
    K_hom = np.vstack((K, np.zeros(K[0].shape)))
    K_hom = np.hstack((K_hom, np.array([[0,0,0,1]]).T))

#

    #print("1. X_G visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))
    xyz_hom1 = np.matmul(extrinsics, xyz_hom1) #xyz_hom1.shape: 4 * 11520000
    #print("2. X_L visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))

    #tf_ex = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    #tf_ex_hom = np.vstack((tf_ex, np.zeros(tf_ex[0].shape)))
    #tf_ex_hom = np.hstack((tf_ex_hom, np.array([[0,0,0,1]]).T))
    #xyz_hom1 = np.matmul(tf_ex_hom, xyz_hom1)
    #print("3. X_L_corr visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))

    xy_img = np.matmul(K_hom, xyz_hom1)
    #print(np.nanmax(xy_img[0:2,:]), np.nanmin(xy_img[0:2,:]))
    xy_img = xy_img[0:2,:] / xy_img[2:3,:] #TODO: Check if minus - should be there before xy_img[2:3,:].
    #print(xy_imgcv.shape, xy_img.shape)
    xy_imgcv = np.array(xy_img.T, dtype = np.int_)

    print("Done cv2.project:")

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
#
    pcd_colors = np.asarray(pcd.colors) * 255
#    #print(pcd_colors.shape, pcd_colors[:10]*255)
#
    synth_img = np.ones((H, W, 3))  * 255
#    print(synth_img.shape)

#    colors_re = pcd_colors.reshape((H,W,3))
#    colors_re = colors_re.T
#    pcd_colors = colors_re.reshape((H*W, 3))
    #print(f"xy_imgcv.shape, synth_img.shape, pcd_colors.shape: {xy_imgcv.shape}, {synth_img.shape}, {pcd_colors.shape}")
    #synth_img(xy_imgcv[:]) 
    for i in range(pcd_colors.shape[0]):
        # Be careful here: For xy_imgcv, (x,y) means x right first then y down.
        # Whereas for numpy array, (x, y) means x down first then y right.

        # 1. Ignore points with negative depth, i.e. ones behind the camera. 
        if xyz_hom1[2,i] > 0: # Make sure the xyz you're checking are in ego frame
            # 2. projected pixel must be between  [{0,W},{0,H}]
            if (xy_imgcv[i,0] >= 0) & (xy_imgcv[i,0] < W):
                if (xy_imgcv[i,1] >= 0) &  (xy_imgcv[i,1] < H):
                    #print(xy_imgcv[i], i)
                    synth_img[xy_imgcv[i,1], xy_imgcv[i,0]] = pcd_colors[i] #



    img = o3d.geometry.Image((synth_img).astype(np.uint8))
    #o3d.visualization.draw_geometries([img])
    o3d.io.write_image(viewpoint_json + "synth.jpg", img)
    print(f"image written to {viewpoint_json}synth.jpg")


    # 2. habitat script grid_sample method

#    pcd_colors = np.array(np.asarray(pcd.colors) , dtype=np.single)
#    colors_re = pcd_colors.reshape((H,W,3))
#
#    xyz_T = xyz.T
#    xyz_hom0 = np.vstack((xyz_T, np.ones(xyz_T[0].shape)))
#    K_hom = np.vstack((K, np.zeros(K[0].shape)))
#    K_hom = np.hstack((K_hom, np.array([[0,0,0,1]]).T))
#
#    xyz_hom1 = np.matmul(extrinsics, xyz_hom0)
#    xy_img = np.matmul(K_hom, xyz_hom1)
#    xy_img = xy_img[0:2,:] / xy_img[2:3,:] #TODO: Check if minus - should be there before xy_img[2:3,:].
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
