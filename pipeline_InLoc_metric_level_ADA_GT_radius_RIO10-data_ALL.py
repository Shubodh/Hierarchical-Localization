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
from hloc import extract_features_rio, match_features, localize_rio, visualization

#import hloc
#print("carefully inspect which hloc it is, whether the docker one or normal modified one.")
#print(hloc)


from hloc.utils.parsers import parse_retrieval, names_to_pair

from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from hloc.utils.io import read_image

import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True) #example argument: --scene_id 05
    parser.add_argument('--scene_type', type=str, required=True, help="Example: ROI_with_QOI") #example argument: --scene_type ROI_with_QOI
    parser.add_argument('--date', type=str, required=True, help="Format: dt100622") #example argument: --scene_type ROI_with_QOI
    parser.add_argument('--time', type=str, required=True, help="Format: t0201") #example argument: --scene_type ROI_with_QOI
    args = parser.parse_args()

    given_scene_id = str(args.scene_id)
    scene_type = str(args.scene_type) #ROI_with_QOI or ROI_and_ARRI_with_QOI_and_AQRI
    date = str(args.date)
    date_netvlad = date #'dt100622' # date
    time = str(args.time)
    dt_time = date +'-' + time 

    #date = 'dt100622'
    #dt_time = date +'-' + 't1111'



    # # Pipeline for indoor localization

    # ## Setup
    # Here we declare the paths to the dataset, image pairs, and we choose the feature extractor and the matcher. You need to download the [InLoc dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/inloc/`, or change the path.

    # and_only_just = "_JUST" # ""  # _ONLY_PLACES means only places, "" means only scene0X, "_AND_PLACES" means both places + scene0X.# _and_places # _only_places

    dataset = Path('/data/InLoc_like_RIO10/sampling10/scene'+ given_scene_id + "_" + scene_type + '/')  # change this if your dataset is somewhere else
    assert dataset.exists(), dataset 
    #dataset = Path('/data/InLoc_like_RIO10/sampling10/scene01_and_places/') #scene01_only_places  # change this if your dataset is somewhere else

    #pairs = Path('pairs/graphVPR/rio_metric/') #'pairs/inloc/'
    pairs = Path('pairs/graphVPR/rio_metric/scene' + given_scene_id + '/')
    netvlad_no = "40" # "100" # 
    loc_pairs = pairs / Path('netvlad'+netvlad_no +"_scene_GT_radius_"+scene_type +"_sampling10_" + date_netvlad +  '.txt') #netvlad40_FOR-scene01_and_places_sampling10_dt070322.txt  #netvlad40_FOR-scene01_only_places_dt070322.txt  # 'netvlad40_dt140222.txt' # top 40 retrieved by NetVLAD #-minustop3rooms
    assert loc_pairs.exists(), loc_pairs

    output_end ='scene' + given_scene_id + "_" + scene_type +  '/' #'scene' + given_scene_id + '_and_places/' #'scene01_and_places' #'scene01_just/'
    # output_end ='scene' + given_scene_id + and_only_just +  '/' #'scene' + given_scene_id + '_and_places/' #'scene01_and_places' #'scene01_just/'
    outputs = Path('/data/InLoc_dataset/outputs/rio/' + output_end)  # where everything will be saved

    # Set config
    feature_name  = 'superpoint_inloc'  # sift, superpoint_inloc, d2net-ss, netvlad
    matcher_name  = 'superglue' # NN-mutual, superglue
    skip_no = 3 #20 #10 # USed 40 for all scenes excluding 9th one for d2net. For superglue, even 10 stops. PCLoc uses 3 lol. so using 3 now.
    refine_pcloc = False

    testing_type = scene_type + "_" + 'GT_radius_scene'+ given_scene_id + '_sampling10_netvlad' + netvlad_no + '_' #'scene01_sampling10_' #'scene01_only_places_' #'scene01_and_places_'

    #results = outputs / Path('RIO_hloc_superpoint+superglue_skip10_' + dt_time + '.txt')  # the result file
    results = outputs / Path(testing_type + 'RIO_hloc_' + feature_name +'+' + matcher_name + '_skip' + str(skip_no) + '_' + dt_time + '.txt')  # the result file
    print(f"Starting localization on {dt_time}")

    # list the standard configurations available
    # print(f'Configs for feature extractors:\n{pformat(extract_features_rio.confs)}')
    # print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


    # pick one of the configurations for extraction and matching
    # you can also simply write your own here!
    feature_conf = extract_features_rio.confs[feature_name] # superpoint_inloc, d2net-ss, netvlad
    matcher_conf = match_features.confs[matcher_name] # superglue


    # ## Extract local features for database and query images
    feature_path = extract_features_rio.main(feature_conf, dataset, outputs)
    print(feature_path)

    match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)
    print(match_path)
    # 
    localize_rio.main(
        dataset, loc_pairs, feature_path, match_path, results,given_scene_id, refine_pcloc,
        skip_matches=skip_no) #20. 10 is giving error currently, for 1 query, unable to find any matches > 20  # skip database images with too few matches

    #visualization.visualize_loc(results, dataset, n=1, top_k_db=1, seed=2)

    # DESCRIPTIONS OF ABOVE STEPS

    # ## Match the query images
    # Here we assume that the localization pairs are already computed using image retrieval (NetVLAD). To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.

    # ## Localize!
    # Perform hierarchical localization using the precomputed retrieval and matches. Different from when localizing with Aachen, here we do not need a 3D SfM model here: the dataset already has 3D lidar scans. The file `InLoc_hloc_superpoint+superglue_netvlad40.txt` will contain the estimated query poses.



    # ## Visualization
    # We parse the localization logs and for each query image plot matches and inliers with a few database images.
