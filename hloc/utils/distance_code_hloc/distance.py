import os
import numpy as np
import sys
from pathlib import Path
from shutil import copy2

from evaluation import get_both_errors, get_errors


def metric_dist(A,B):
    """
    A and B are poses
    numpy 4x4 pose matrices

    returns l2 norm of T vector in pose matrix
    """
    A_T = A[0:3,3]
    B_T = B[0:3,3]
    #norm of X is sq. rt. (x1^2 + x2^2 + x3^2)
    return np.linalg.norm(A_T - B_T)




def read_pose(file_name):
    #Reading the individual pose file txt (RIO10 format: 4*4 matrix in 4 lines)
    with open(file_name,'r') as f:
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


def calculate_distances(target_file, ref_files, distance_func):
    """
    Takes target_file and ref_files without the final extension
    See main function for how target and ref files are prepped
    returns a np array of dimension of ref_files
    each element is a distance calculated by distance_func
    """
    distances = []
    _,target_pose = read_pose(target_file + '.pose.txt')
    for i in sorted(ref_files):

        _,ref_pose = read_pose(i + '.pose.txt')

        print(i)
        samply_path_to_copy = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/hloc/utils/distance_code_hloc/samply_visualization/"
        copy2(Path(i+'.color.jpg'), Path(samply_path_to_copy))

        rot_error, trans_error = distance_func(target_pose, ref_pose)
        distances.append(trans_error)
        # print("Distance between "+i+" and "+target_file+" is: ",distance_func(target_pose, ref_pose))
    return np.array(distances)

def images_in_x_radius(target_file, ref_files, radius,distance_func):
    """
    returns list of files within x radius
    """

    distances = calculate_distances(target_file, ref_files, distance_func)
    ref_numpy = np.array(ref_files)

    ref_in_x = ref_numpy[distances < radius]
    final_files = [i + '.color.jpg' for i in ref_in_x]
    return final_files

def images_in_x_radius_and_t_angle(target_file, ref_files, radius,distance_func):
    """
    returns list of files within x radius
    """

    distances = calculate_distances(target_file, ref_files, distance_func)
    ref_numpy = np.array(ref_files)

    ref_in_x = ref_numpy[distances < radius]
    final_files = [i + '.color.jpg' for i in ref_in_x]
    return final_files

def closest_x_images(target_file, ref_files,num_matches,distance_func):
    """
    Returns top num_matches of files according to distance_func
    Final array dimension is of size
    """
    distances = calculate_distances(target_file, ref_files, distance_func)
    ref_numpy = np.array(ref_files)

    idx = np.argsort(distances)
    ref_sorted = ref_numpy[idx]
    ref_top = ref_sorted[0:num_matches]
    final_files = [i + '.color.jpg' for i in ref_top]
    return final_files



if __name__ == '__main__':
    shub_local = '/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/'
    dataset_path = shub_local + 'scene01'
    seq_num = 'seq01'
    sub_seq_num = '01'

    ref_files_dataset_path = os.path.join(dataset_path, seq_num,seq_num+'_'+sub_seq_num)

    #prepare nums files
    nums = []
    for file in os.listdir(ref_files_dataset_path):
        if '.pose.txt' in file:
            file_name = os.path.join(ref_files_dataset_path,file[:-9])
            nums.append(file_name)

    #Task 2
    # i. Output names of images in x metres (say x=5) radius.
    radius = 1.3#5
    distance_func = get_both_errors
    image_list = images_in_x_radius(nums[0], nums[1:50], radius,distance_func)
    print(f"For this {nums[0]}: output names of images in {radius} radius")
    print(image_list, len(image_list))
    sys.exit()

    #ii. Output names of images in x metres (say x=5) radius AND within 90° angle along yaw axis.
    radius = 1#5
    distance_func = get_both_errors
    image_list = images_in_x_radius_and_t_angle(nums[0], nums[1:50], radius,distance_func)
    print(f"For this {nums[0]}: output names of images in {radius} radius")
    print(image_list, len(image_list))

    #iii. Output 40 closest images in terms of metric distance.
    num_matches = 40
    distance_func = get_both_errors
    closest_x_files = closest_x_images(nums[0], nums[1:1000],num_matches,distance_func)
    print(closest_x_files)