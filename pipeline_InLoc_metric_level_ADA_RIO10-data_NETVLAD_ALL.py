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
    args = parser.parse_args()
    given_scene_id = str(args.scene_id)
    scene_type = str(args.scene_type) #ROI_with_QOI or ROI_and_ARRI_with_QOI_and_AQRI

    # # Pipeline for indoor localization

    # ## Setup
    # Here we declare the paths to the dataset, image pairs, and we choose the feature extractor and the matcher. You need to download the [InLoc dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/inloc/`, or change the path.

    # and_only_just = "_JUST" # ""  # _ONLY_PLACES means only places, "" means only scene0X, "_AND_PLACES" means both places + scene0X.# _and_places # _only_places
    #date = 'dt050622'

    dataset = Path('/data/InLoc_like_RIO10/sampling10/scene'+ given_scene_id + "_" + scene_type + '/')  # change this if your dataset is somewhere else
    assert dataset.exists(), dataset 
    #dataset = Path('/data/InLoc_like_RIO10/sampling10/scene01_and_places/') #scene01_only_places  # change this if your dataset is somewhere else

    #pairs = Path('pairs/graphVPR/rio_metric/') #'pairs/inloc/'
    #loc_pairs = pairs / Path('netvlad40_FOR-scene'+ given_scene_id + '_sampling10_' + date +  '.txt') #netvlad40_FOR-scene01_and_places_sampling10_dt070322.txt  #netvlad40_FOR-scene01_only_places_dt070322.txt  # 'netvlad40_dt140222.txt' # top 40 retrieved by NetVLAD #-minustop3rooms

    output_end ='scene' + given_scene_id + "_" + scene_type +  '/' #'scene' + given_scene_id + '_and_places/' #'scene01_and_places' #'scene01_just/'
    outputs = Path('/data/InLoc_dataset/outputs/rio/' + output_end)  # where everything will be saved
    outputs.mkdir(parents=True, exist_ok=True)

    # Set config
    #dt_time = date +'-t1110'
    feature_name  = 'netvlad'  # sift, superpoint_inloc, d2net-ss, netvlad
    # matcher_name  = 'NN-mutual' # NN-mutual, superglue
    #skip_no = 20

    #testing_type ='scene'+ given_scene_id + '_sampling10_' #'scene01_sampling10_' #'scene01_only_places_' #'scene01_and_places_'

    #results = outputs / Path('RIO_hloc_superpoint+superglue_skip10_' + dt_time + '.txt')  # the result file
    #results = outputs / Path(testing_type + 'RIO_hloc_' + feature_name +'+' + matcher_name + '_skip' + str(skip_no) + '_' + dt_time + '.txt')  # the result file
    #print(f"Starting localization on {dt_time}")

    # list the standard configurations available
    # print(f'Configs for feature extractors:\n{pformat(extract_features_rio.confs)}')
    # print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


    # pick one of the configurations for extraction and matching
    # you can also simply write your own here!
    feature_conf = extract_features_rio.confs[feature_name] # superpoint_inloc, d2net-ss, netvlad
    # matcher_conf = match_features.confs[matcher_name] # superglue


    # ## Extract local features for database and query images
    feature_path = extract_features_rio.main(feature_conf, dataset, outputs)
    print(feature_path)
    print(outputs)
    print("\n")