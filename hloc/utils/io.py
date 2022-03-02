import cv2
import h5py
import numpy as np
import logging
from pathlib import Path
import sys
import png
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append('../../') #TODO-Later: Not a permanent solution, should fix imports later.
sys.path.append('../')
from hloc.utils.parsers import parse_poses_from_file
from hloc.utils.viz import plot_images, plot_images_simple
# from .parsers import parse_poses_from_file
# from parsers import parse_poses_from_file

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image

def read_depth_image_given_depth_path(path):
    depth_file = Path(path)
    assert  depth_file.exists(), depth_file 
    
    depth_image = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        raise ValueError(f'Cannot read image {path}.')

    return depth_image

def read_depth_image_given_colorimg_path(dataset_dir, r):
    """
    This function replaces color img path with depth img for same img id.
    dataset_dir: Path(datasets/InLoc_like_RIO10/scene01_synth)
    r: Path(database/cutouts/frame-001820.color.jpg)
    """
    full_prefix_path = dataset_dir / r.parents[0]
    r_stem = r.stem.replace("color", "")
    depth_file  = Path(full_prefix_path, r_stem + 'rendered.depth.png')
    assert  depth_file.exists(), depth_file 
    
    depth_image = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        raise ValueError(f'Cannot read image {path}.')

    # Using Open3D
    #print("DEBUG DePTH") 
    #depth_raw = o3d.io.read_image(str(depth_file))
    #depth_img = np.asarray(depth_raw)
    #print(np.array_equal(depth_img, depth_image))

    return depth_image

def save_depth_image(np_array, save_path):
    """
    Given numpy array of depth in millimetres, save the png image at save_path.
    save_path must end with "_depth.png" or "rendered.depth.png"
    """
    # If np_array in metres, do np_array*1000
    depth_mm = (np_array).astype(np.uint16)

    # with open(raw_data_folder + str(obs_id) + "_depth.png", 'wb') as fdep:
    with open(save_path, 'wb') as fdep:
        writer = png.Writer(width=depth_mm.shape[1], height=depth_mm.shape[0], bitdepth=16, greyscale=True)
        depth_gray2list = depth_mm.tolist()
        writer.write(fdep, depth_gray2list)
    print(f"Saved depth image at {save_path}")

def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))

def write_results_to_file(path, img_poses_list):
    # img_poses_list: [(img_name_1, pose_1), (_2, _2)..]. pose_1 is [qw, qx, qy, qz]
    with open(path, 'w') as f:
        for name, pose in img_poses_list:
            qvec, tvec = pose[:4], pose[4:]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            # name = q.split("/")[-1]
            f.write(f'{name} {qvec} {tvec}\n')
    logging.info(f'Written {len(img_poses_list)} poses to {path.name}')

def convert_pose_file_format_wtoc_to_ctow(pose_path):
    # file_pose = Path("outputs/rio/full/RIO_hloc_d2net-ss+NN-mutual_skip10_dt160222-t0411.txt")
    pose_path = Path(pose_path)
    poses = parse_poses_from_file(pose_path)
    img_poses_list_final = []
    for file, pose in poses:
        RT_wtoc = np.zeros((4,4))
        RT_wtoc[3,3] = 1 

        assert len(pose) == 7
        qw, qx, qy, qz  = pose[0:4]
        RT_wtoc[0:3,0:3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        RT_wtoc[0:3, 3] = pose[4:]
        RT_ctow = np.copy(RT_wtoc)
        RT_ctow[0:3,0:3] = RT_wtoc[0:3,0:3].T
        RT_ctow[0:3,3] = - RT_wtoc[0:3,0:3].T @ RT_wtoc[0:3,3]

        qx_c, qy_c, qz_c, qw_c = R.from_matrix(RT_ctow[0:3,0:3]).as_quat()
        tx_c, ty_c, tz_c = RT_ctow[0:3,3]

        pose_c = [qw_c, qx_c, qy_c, qz_c, tx_c, ty_c, tz_c]
        img_poses_list_final.append((file, pose_c))

    write_pose_path = Path(str(pose_path.parents[0] / pose_path.stem) + "_corrected_frame.txt")
    # write_pose_path = Path("outputs/rio/full/RIO_hloc_d2net-ss+NN-mutual_skip10_dt160222-t0411_corrected_frame.txt")
    write_results_to_file(write_pose_path, img_poses_list_final)

def main_check_read_write_depth():
    dataset_dir = "../../datasets/InLoc_like_RIO10/scene01_synth"
    r = "database/cutouts/frame-004269.color.jpg" #frame-001820.color.jpg #frame-004269.color.jpg

    depth_image = read_depth_image_given_colorimg_path(Path(dataset_dir), Path(r))
    save_path = "./temp_dir/hi.rendered.depth.png"
    save_depth_image(depth_image, save_path)

    #1. read saved depth image
    depth_image_2 = read_depth_image_given_depth_path(save_path)
    #2. Compare current and original depth, np.array_equal 
    # print(np.array_equal(depth_image, depth_image_2))
    # plot_images_simple(read_image(Path(dataset_dir) / Path(r)), depth_image_2)
    # plt.show()
    # print(np.unique(depth_image_2))

if __name__ == "__main__":
    # main_dummy()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_path', type=Path, required=True)
    args = parser.parse_args()
    convert_pose_file_format_wtoc_to_ctow(**args.__dict__)

    # main_check_read_write_depth()
    # pose_path = Path("outputs/rio/full/RIO_hloc_d2net-ss+NN-mutual_skip10_dt160222-t0411.txt")
    # convert_pose_file_format_wtoc_to_ctow(pose_path)
