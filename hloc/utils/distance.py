import os
import numpy as np
import sys
from pathlib import Path
from shutil import copy2

sys.path.append('../../')
from hloc.utils.evaluation import get_both_errors


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

def calculate_distances(target_file, ref_files):
    """
    Takes target_file and ref_files without the final extension
    See main function for how target and ref files are prepped
    returns a np array of dimension of ref_files
    """
    distances = []
    angles = []
    _,target_pose = read_pose(target_file + '.pose.txt')
    for i in sorted(ref_files):

        _,ref_pose = read_pose(i + '.pose.txt')

        # print(i)
        # samply_path_to_copy = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/hloc/utils/distance_code_hloc/samply_visualization/"
        # copy2(Path(i+'.color.jpg'), Path(samply_path_to_copy))

        rot_error, trans_error = get_both_errors(target_pose, ref_pose)
        distances.append(trans_error)
        angles.append(rot_error)
    return np.array(distances), np.array(angles)

def images_in_x_radius(target_file, ref_files, radius):
    """
    returns list of files within x radius
    """

    distances, _ = calculate_distances(target_file, ref_files)
    # ref_numpy = np.array(ref_files)

    ref_in_x = ref_numpy[distances < radius]
    return ref_in_x

def images_in_x_radius_and_t_angle(target_file, ref_files, radius, angle_threshold=20.0):
    """
    returns list of files within x radius and
    angle_threshold in degrees
    """

    distances, angles = calculate_distances(target_file, ref_files)
    # ref_numpy = np.array(ref_files)

    ref_in_x = ref_files[angles < angle_threshold]
    distances = distances[angles < angle_threshold]
    # print(ref_in_x.shape, ref_numpy.shape, angles.shape, distances.shape, angle_threshold, radius)
    ref_in_x = ref_in_x[distances < radius]
    # final_files = [i + '.color.jpg' for i in ref_in_x]
    return ref_in_x

def closest_x_images(target_file, ref_files, num_matches):
    """
    Returns top num_matches of files 
    Final array dimension is of size
    """
    distances, _ = calculate_distances(target_file, ref_files)
    # ref_numpy = np.array(ref_files)

    idx = np.argsort(distances)
    ref_sorted = ref_files[idx]
    ref_top = ref_sorted[0:num_matches]
    # final_files = [i + '.color.jpg' for i in ref_top]
    return ref_top 

def bin_based_shortlisting(target_file, ref_files, iter_no, final_set):
    """ Call like : iter_no=0, final_set=set()
    Explanation:
    Given a query pose and list of ref poses, shortlist 40 ref poses "closest" to query poses.
    "closest" is difficult to define, therefore we take a bin based approach.
    Basically, if `bins = [(0.05, 5), (0.25,10), (2, 10), ....]`, 
    we first shortlist imgs using images_in_x_radius_and_t_angle() in bins[0], 
    then if <40 found, then include bins[1] and so on till 40 are found.
    If > 40 found, use closest_x_images() for finding 40 exactly.
    """
    # Remember that image_list will include bins[0] + .. + bins[i], not just bins[i].
    bins = [(0.25, 5.0), (0.5,10), (1,10), (2, 10), (3,15), (5,20), (7,30), (10,60), (10, 160), (10, 200)]

    # print("DEBUGGING HERE")
    # bins = [(10, 60)]
    radius, angle_threshold = bins[iter_no]
    image_list = images_in_x_radius_and_t_angle(target_file, ref_files, radius, angle_threshold)

    # print(len(image_list), "hi", len(ref_files))
    # print(f"iteration {i}, target_file {target_file}, image_list {image_list}, radius {radius}  and angle {angle_threshold}")

    # PRINT THIS
    # print(f"len-image_list {len(image_list)}, iteration {iter_no}, target_file {target_file}, radius {radius}  and angle {angle_threshold}")

    #if len(image_list) + len(final_set) >= 40:
    if len(image_list) >= 40: # image_set will already include final_set. image_set = image_set[i-1] + final_set[i]
        # Below is wrong: it will give redundant images
        # closest_list = closest_x_images(target_file, image_list, num_matches=40-len(final_set))

        # Below is right: You are removing common images and then giving to closest_x_..() function.
        new_list = np.array(list((set(image_list)).difference(final_set)))
        closest_list = closest_x_images(target_file, new_list, num_matches=40-len(final_set))
        final_set.update(closest_list)
        final_list = list(final_set)

        # PRINT THIS
        print(f"For this query {target_file}, ended at {bins[iter_no]}, len(final_list): {len(final_list)}")
        # print(f"The fight is done. Ended this at {bins[iter_no]}, len(final_list): {len(final_list)}, final_list: {final_list}")
        return final_list 
    else:
        final_set.update(image_list)
        iter_no += 1
        return bin_based_shortlisting(target_file, ref_files, iter_no, final_set)



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

    target_file, ref_files = nums[0], np.array(nums[1:50])
    final_set = bin_based_shortlisting(target_file, ref_files)
    sys.exit()

    separate_func = False
    if separate_func:
        # i. Output names of images in x metres (say x=5) radius.
        # radius = 1.3#5
        # image_list = images_in_x_radius(nums[0], nums[1:50], radius)
        # print(f"For this {nums[0]}: output names of images in {radius} radius")
        # print(image_list, len(image_list))

        #ii. Output names of images in x metres (say x=5) radius AND within 90° angle along yaw axis.
        radius = 1.3#5
        angle_threshold = 60.0
        image_list = images_in_x_radius_and_t_angle(nums[0], nums[1:50], radius, angle_threshold)
        print(f"For this {nums[0]}: output names of images in {radius} radius and {angle_threshold} angle")
        print(image_list, len(image_list))
        sys.exit()

        #iii. Output 40 closest images in terms of metric distance.
        num_matches = 40
        closest_x_files = closest_x_images(nums[0], nums[1:1000],num_matches)
        print(closest_x_files)