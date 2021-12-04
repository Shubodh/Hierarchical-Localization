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
    vis.add_geometry(pcd)
    #print(np.asarray(pcd.points)[:100])
    #sys.exit()
    if show_coord:
        vis.add_geometry(coord_mesh)
    vis.run() # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    #vis.capture_screen_image(filename+".png")
    #print(param.extrinsic)
    vis.destroy_window()

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    print(param.extrinsic)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.capture_screen_image(filename+".png")
    vis.destroy_window()

def synthesize_img_given_viewpoint(pcd, viewpoint_json):
    # IMP NOTE: CURRENT `pcd` is not yet FULL ROOM PCD, it's just first 5 depths merged...
    # Step 1: From json file, load extrinsic tf and pre-multiply this with pcd
    # Step 2: From json, load instrincic and ,,,,, with previous step output
    # Just do both in 1 step finally
    #load_view_point(pcd, viewpoint_json)
    vpt_json = json.load(open(viewpoint_json))
    extrinsics = np.array(vpt_json['extrinsic']).reshape(4,4).T
    K = np.array(vpt_json['intrinsic']['intrinsic_matrix']).reshape(3,3).T
    #H = int(vpt_json['intrinsic']['height'])
    #W = int(vpt_json['intrinsic']['width'])
    H = 1200
    W = 1600

    xyz = np.asarray(pcd.points)

#    print(f"xyz, extrinsics {xyz.shape}{extrinsics.shape}")
#    xyz_T = np.matmul(extrinsics[0:3,0:3], xyz.T)
#    xyz = xyz_T.T
    rvecs = np.zeros(3)
    cv2.Rodrigues(extrinsics[0:3,0:3], rvecs)
    tvecs = np.zeros(3)
    tvecs = extrinsics[0:3,3]

    dist = np.zeros(5)
    xy_imgcv, jac = cv2.projectPoints(xyz, rvecs, tvecs, K, dist)
    xy_imgcv = np.array(xy_imgcv.reshape(xy_imgcv.shape[0], 2), dtype=np.int_)
    print(np.max(xy_imgcv), np.min(xy_imgcv), xy_imgcv)

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
    print(f"xy_imgcv.shape, synth_img.shape, pcd_colors.shape: {xy_imgcv.shape}, {synth_img.shape}, {pcd_colors.shape}")
    #synth_img(xy_imgcv[:]) 
    total = 0
    total_if = 0
    for i in range(pcd_colors.shape[0]):
        total+=1
        if (xy_imgcv[i,0] >= 0) & (xy_imgcv[i,0] < H):
            if (xy_imgcv[i,1] >= 0) &  (xy_imgcv[i,1] < W):
                #print(xy_imgcv[i], i)
                synth_img[H - 1 - xy_imgcv[i,0], W - 1 - xy_imgcv[i,1]] = pcd_colors[i]
                total_if+=1
    print(f"Total number of full pcd_colors iterations & if conditions pass: {total, total_if}")


    img = o3d.geometry.Image((synth_img).astype(np.uint8))
    #o3d.visualization.draw_geometries([img])
    o3d.io.write_image(viewpoint_json + "synth.jpg", img)


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
