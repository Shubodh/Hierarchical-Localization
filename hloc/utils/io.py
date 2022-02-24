import cv2
import h5py
import logging

from .parsers import parse_poses_from_file

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

def read_depth_image_given_colorimg_path(dataset_dir, r):
    """
    This function replaces color img path with depth img for same img id.
    dataset_dir: datasets/InLoc_like_RIO10/scene01_synth 
    r: database/cutouts/frame-001820.color.jpg
    """
    full_prefix_path = dataset_dir / r.parents[0]
    r_stem = r.stem.replace("color", "")
    depth_file  = Path(full_prefix_path, r_stem + 'rendered.depth.png')
    assert  depth_file.exists(), depth_file 

    # Using Open3D
    depth_raw = o3d.io.read_image(str(depth_file))
    depth_img = np.asarray(depth_raw)
    
    # Using cv2 ? matplotlib?

    return depth_img

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

def convert_pose_file_format_wtoc_to_ctow(path):
    file_pose = Path("outputs/rio/full/RIO_hloc_d2net-ss+NN-mutual_skip10_dt160222-t0411.txt")
    #file_pose = path
    poses = parse_poses_from_file(file_pose)
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


    write_pose_path = Path("outputs/rio/full/RIO_hloc_d2net-ss+NN-mutual_skip10_dt160222-t0411_corrected_frame.txt")
    write_results_to_file(write_pose_path, img_poses_list_final)

def main_dummy():
    pass

if __name__ == "__main__":
    main_dummy()
    dataset_dir = "datasets/InLoc_like_RIO10/scene01_synth"
    r = "database/cutouts/frame-001820.color.jpg"
    depth_img = read_depth_image_given_colorimg_path(Path(dataset_dir), Path(r))