#!/usr/bin/env python
# coding: utf-8
import h5py
from pathlib import Path
from pprint import pformat
import sys
from matplotlib import cm
import cv2
from tqdm import tqdm
import random
import numpy as np
import pickle
import time

#sys.path.append(str(Path(__file__).parent / '..')) #to import hloc
from hloc import extract_features, match_features, localize_rio, visualization

#import hloc
#print("carefully inspect which hloc it is, whether the docker one or normal modified one.")
#print(hloc)


from hloc.utils.parsers import parse_retrieval, names_to_pair

from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from hloc.utils.io import read_image

import matplotlib.pyplot as plt



# # Pipeline for indoor localization

# ## Setup
# Here we declare the paths to the dataset, image pairs, and we choose the feature extractor and the matcher. You need to download the [InLoc dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/inloc/`, or change the path.

# dataset = Path('/data/InLoc_dataset/')  # change this if your dataset is somewhere else
dataset = Path('datasets/InLoc_like_RIO10/scene01/')  # change this if your dataset is somewhere else

pairs = Path('pairs/graphVPR/rio_metric/') #'pairs/inloc/'
loc_pairs = pairs / 'bruteforce40_samply.txt'  # bruteforce40_samply.txt # top 40 retrieved by NetVLAD #-minustop3rooms

outputs = Path('outputs/rio/')  # where everything will be saved

# Set config
dt_time = 'dt150222-t0112'
feature_name  = 'd2net-ss'  # sift, superpoint_inloc
matcher_name  = 'superglue' # NN-mutual
skip_no = 10

results = outputs / Path('RIO_hloc_' + feature_name +'+' + matcher_name + '_skip' + str(skip_no) + '_' + dt_time + '.txt')  # the result file
print(f"Starting localization on {dt_time}")

# list the standard configurations available
# print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
# print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


# pick one of the configurations for extraction and matching
# you can also simply write your own here!
feature_conf = extract_features.confs[feature_name] # superpoint_inloc, d2net-ss, netvlad
matcher_conf = match_features.confs[matcher_name] # superglue


# ## Extract local features for database and query images
feature_path = extract_features.main(feature_conf, dataset, outputs)


# ## Match the query images
# Here we assume that the localization pairs are already computed using image retrieval (NetVLAD). To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.
match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)

# ## Localize!
# Perform hierarchical localization using the precomputed retrieval and matches. Different from when localizing with Aachen, here we do not need a 3D SfM model here: the dataset already has 3D lidar scans. The file `InLoc_hloc_superpoint+superglue_netvlad40.txt` will contain the estimated query poses.

localize_rio.main(
    dataset, loc_pairs, feature_path, match_path, results,
    skip_matches=skip_no) #20. 10 is giving error currently, for 1 query, unable to find any matches > 20  # skip database images with too few matches


# ## Visualization
# We parse the localization logs and for each query image plot matches and inliers with a few database images.
#visualization.visualize_loc(results, dataset, n=1, top_k_db=1, seed=2)
