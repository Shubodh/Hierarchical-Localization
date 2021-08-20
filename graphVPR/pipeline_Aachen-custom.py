import h5py
from pathlib import Path
from pprint import pformat
import sys
from matplotlib import cm
import random
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / '..')) #to import hloc
from hloc import extract_features, match_features, pairs_from_covisibility
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization


from hloc.utils.read_write_model import read_model
from hloc.utils.parsers import parse_image_lists, parse_retrieval, names_to_pair

from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from hloc.utils.io import read_image


def main(dataset_dir, retrieval, features):


if __name__ == '__main__':
    dataset = Path('../datasets/aachen_small/')  # change this if your dataset is somewhere else
    images = dataset / 'images/images_upright/'

    pairs = Path('../pairs/aachen/')
    sfm_pairs = pairs / 'pairs-db-covis20-small.txt'  # top 20 most covisible in SIFT model
    loc_pairs = pairs / 'pairs-query-netvlad20-small.txt'  # top 20/50 retrieved by NetVLAD

    outputs = Path('../outputs/aachen_small/')  # where everything will be saved
    reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
    results = outputs / 'Aachen_small_hloc_superpoint+superglue_netvlad20-small.txt'  # the result file

    features = Path('../outputs/aachen_small/feats-superpoint-n4096-r1024.h5')

    main(images, loc_pairs, features)
