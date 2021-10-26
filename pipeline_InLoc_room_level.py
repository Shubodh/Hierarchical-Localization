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
from hloc import extract_features, match_features, localize_inloc, visualization

#import hloc
print("carefully inspect which hloc it is, whether the docker one or normal modified one.")
#print(hloc)


from hloc.utils.parsers import parse_retrieval, names_to_pair

from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from hloc.utils.io import read_image

import matplotlib.pyplot as plt


def main(dataset, pairs, outputs, loc_pairs, results):

    ## list the standard configurations available
    #print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    #print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    # pick one of the configurations for extraction and matching
    # you can also simply write your own here!
    feature_conf = extract_features.confs['superpoint_inloc'] #superpoint_inloc, d2net-ss, netvlad
    matcher_conf = match_features.confs['superglue'] # NN-mutual,  NN-ratio, NN-superpoint, superglue

    # ## Extract local features for database and query images
    feature_path = extract_features.main(feature_conf, dataset, outputs)
    print(feature_path)
    #time.sleep(20)

    # ## Match the query images
    # Here we assume that the localization pairs are already computed using image retrieval (NetVLAD). To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.
    match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)
    print(match_path)
    #exit()
    #time.sleep(45)
    #time.sleep(10)
            

    # ## Localize!
    # Perform hierarchical localization using the precomputed retrieval and matches. Different from when localizing with Aachen, here we do not need a 3D SfM model here: the dataset already has 3D lidar scans. The file `InLoc_hloc_superpoint+superglue_netvlad40.txt` will contain the estimated query poses.
    localize_inloc.main(
        dataset, loc_pairs, feature_path, match_path, results,
        skip_matches=20)#20  # skip database images with too few matches

    ## ## Visualization
    ## We parse the localization logs and for each query image plot matches and inliers with a few database images.
    visualization.visualize_loc_kp(results, dataset, n=1, top_k_db=1, seed=2)

def kp_from_cluster(dataset_dir, q, retrieved, feature_file, match_file):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    cx = .5 * width
    cy = .5 * height
    #focal_length = 4032. * 28. / 36.

    all_mkpq = []
    all_mkpr = []
    #all_mkp3d = []
    all_indices = []
    kpq = feature_file[q]['keypoints'].__array__()
    #print(len(retrieved))
    #exit()

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        q_image = read_image(dataset_dir / q)
        r_image = read_image(dataset_dir / r)
        print(r_image.shape)
        exit()
        m = match_file[pair]['matches0'].__array__()
        v = (m > -1)

        mkpq, mkpr = kpq[v], kpr[m[v]]

        #print(mkpq.shape, mkpr.shape)
        plot_images([q_image, r_image])
        plot_matches(mkpq, mkpr)
        plt.show()
        exit()
        #v = (m > -1)
#print(q, kpq.shape)
#    num_matches = 0
#
#    for i, r in enumerate(retrieved):
#        kpr = feature_file[r]['keypoints'].__array__()
#        pair = names_to_pair(q, r)

# Experimentation
def main_experiment(dataset_dir, retrieval, features, matches):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    #print(retrieval_dict)
    queries = list(retrieval_dict.keys())

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    i = 0
    for q in tqdm(queries):
        db = retrieval_dict[q]
        kp_from_cluster(dataset_dir, q, db, feature_file, match_file)
        if i == 5:
            exit()
        i +=1


if __name__ == '__main__':
    # ## Setup
    # Here we declare the paths to the dataset, image pairs, and we choose the feature extractor and the matcher. You need to download the [InLoc dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/inloc/`, or change the path.
    scene_names = [
    '8WUmhLawc2A',
    'EDJbREhghzL',
    'i5noydFURQK',
    'jh4fc5c5qoQ',
    'mJXqzFtmKg4',
    'qoiz87JEwZ2',
    'RPmz2sHmrrY',
    'S9hNv5qa7GM',
    'ULsKaCPVFJR',
    'VzqfbhrpDEA',
    'wc2JMjhGNzB',
    'WYY7iVyf5p8',
    'X7HyMhZNoso',
    'YFuZgdQ5vWj',
    'yqstnuAEVhm'
    ]
    
    folder_names = [
    '0_mp3d_8WUmhLawc2A',
    '1_mp3d_EDJbREhghzL',
    '2_mp3d_i5noydFURQK',
    '3_mp3d_jh4fc5c5qoQ',
    '4_mp3d_mJXqzFtmKg4',
    '5_mp3d_qoiz87JEwZ2',
    '6_mp3d_RPmz2sHmrrY',
    '7_mp3d_S9hNv5qa7GM',
    '8_mp3d_ULsKaCPVFJR',
    '9_mp3d_VzqfbhrpDEA',
    '10_mp3d_wc2JMjhGNzB',
    '11_mp3d_WYY7iVyf5p8',
    '12_mp3d_X7HyMhZNoso',
    '13_mp3d_YFuZgdQ5vWj',
    '14_mp3d_yqstnuAEVhm'
    ]

    txt_suffix = '.txt'
    h5_suffix = '.h5'


    debug=True
    if debug==True:
        scene_names = [
        '8WUmhLawc2A'
        ]
        
        folder_names = [
        '0_mp3d_8WUmhLawc2A'
        ]

    retrieval_name = ["SP_SG_bruteforce", "hist-top3r-1i", "netvlad-top40", "netvlad-top3"]
    #so retrieval_name[0] is bruteforce, i.e. has ALL pairs, for every query, every ref would exist in that pair txt file.
    # change this if your dataset is somewhere else
    for scene_name, folder_name in tqdm(zip(scene_names, folder_names)):
        print(f"currently folder: {folder_name}")
        dataset = Path('./datasets/graphVPR/room_level_localization_small/' + folder_name + '/')
        pairs = Path('./pairs/graphVPR/room_level_localization_small/')
        #loc_pairs = pairs / (retrieval_name[0]+ '/'+'pairs-' + folder_name + txt_suffix)  #'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
        loc_pairs = pairs / (retrieval_name[2]+ '/'+ scene_name + txt_suffix)  
        #outputs = Path('../outputs/graphVPR/room_level_localization_small/'+ retrieval_name[0]+ '/' + folder_name + '/')  # where everything will be saved
        outputs = Path('./outputs/graphVPR/room_level_localization_small/' +retrieval_name[2]+ '/'+ scene_name + '/')  # where everything will be saved
        #results = outputs /  '_hloc_superpoint+superglue_NOTnetvlad40.txt'  # the result file
        results = outputs / '_hloc_superpoint+superglue_NOTnetvlad40.txt'  # the result file

        main(dataset, pairs, outputs, loc_pairs, results)
#
#    scene_name = '8WUmhLawc2A'
#    dataset = Path('../datasets/graphVPR/dummy-old/mp3d_'+scene_name+'_small/')
#    features = Path('../outputs/graphVPR/mp3d_'+scene_name+'_small/feats-superpoint-n4096-r1600.h5')
#    matches = Path('../outputs/graphVPR/mp3d_'+scene_name+'_small/feats-superpoint-n4096-r1600_matches-superglue_pairs-query-mp3d_'+scene_name+'_small.h5')
#    loc_pairs = Path('../pairs/graphVPR/mp3d_'+scene_name+'_small/pairs-query-mp3d_8WUmhLawc2A_small.txt')
#    main_experiment(dataset, loc_pairs, features, matches)
