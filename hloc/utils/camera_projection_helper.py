import cv2
import numpy as np
import open3d as o3d
import json

# from open3d_helper import viz_with_array_inp


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



def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
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
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

    Return:
    ----------
    X : double
        The x value in meters
    Y : double
        The y value in meters
    Z : double
        The z value in meters

    """
    X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
    Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth
    return X, Y, depth



def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics ):
    """
    Convert the depthmap to a 3D point cloud

    Parameters:
    -----------
    depth_frame 	 	 : rs.frame()
                           The depth_frame containing the depth map
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

    Return:
    ----------
    x : array
        The x values of the pointcloud in meters
    y : array
        The y values of the pointcloud in meters
    z : array
        The z values of the pointcloud in meters

    """
    
    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

    z = depth_image.flatten() / 1000;
    x = np.multiply(x,z)
    y = np.multiply(y,z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z

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

    x = m*camera_intrinsics.fx + camera_intrinsics.ppx
    y = n*camera_intrinsics.fy + camera_intrinsics.ppy

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



def synthesize_img_given_viewpoint(pcd, viewpoint_json):
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
    xy_img = xy_img[0:2,:] / xy_img[2:3,:] #TODO: Check if minus - should be there before xy_img[2:3,:].
    #print(xy_imgcv.shape, xy_img.shape)
    xy_imgcv = np.array(xy_img.T, dtype = np.int_)

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
#