import h5py
from pathlib import Path
from pprint import pformat
import sys
import re

sys.path.append(str(Path(__file__).parent / '..')) #to import hloc
from hloc import extract_features, match_features, localize_inloc, visualization

from matplotlib import cm
import random
import numpy as np
import pickle

from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from hloc.utils.io import read_image

def main(scene_name, folder_name, num_lines_dict, num_rooms_dict):

    # Set scene_name & matching method name
    feat_method = 'superpoint-n4096-r1600' #sift,  superpoint-n4096-r1600, d2net-ss
    match_method = 'superglue' #superglue, NN-mutual
    # Set whether you want output for 2 queries or 2nd query or 1st query (small)
    #query_2 = '_only2query'; queries_2 = '_2queries'; small = '_small'
    h5_suffix = '.h5'
    # Next line might need to be changed for more queries, just do *2 for (1-easy + 1-difficult) aka _2queries
#    if h5_suffix==queries_2:
#        num_lines_dict = {'8WUmhLawc2A': 992*2, 'EDJbREhghzL':1312*2, 'i5noydFURQK':882*2} 
#    else:
#        num_lines_dict = {'8WUmhLawc2A': 992, 'EDJbREhghzL':1312, 'i5noydFURQK':882} 


    #path = str(Path('outputs/inloc_small/feats-superpoint-n4096-r1600.h5'))
    #path = str(Path('outputs/inloc_small/feats-superpoint-n4096-r1600_matches-superglue_pairs-query-netvlad40-custom-shub-small.h5'))
    #path = str(Path('outputs/graphVPR/feats-superpoint-n4096-r1600_matches-superglue_pairs-query-graphVPR-samply.h5'))
    path = str(Path('../outputs/graphVPR/room_level_localization_small/' + folder_name+ '/\
feats-'+feat_method+ '_matches-' +match_method+ '_pairs-' +folder_name+h5_suffix))

    num_lines = num_lines_dict[folder_name]
    num_rooms = num_rooms_dict[folder_name]

    with h5py.File(path, 'r') as hfile:
        all_list = []
        all_dict = {}
        key_num = 0
        for key in hfile.keys():
            dset = hfile[key]
            matches0 = dset['matches0']
            m0_np = np.array(matches0)
            #print(f"m0_np.shape {m0_np.shape}")
            #print(m0_np[:30])
            all_list.append(m0_np)
            all_dict[key_num] = key 
            key_num +=1
        
    #    print(len(all_dict))
    #    exit()
        result_list = []
        total = 0
        num_queries_total = num_rooms * 2
        for outer in range(num_queries_total):
            best_num=0
            best_key = ''
            ite = 0
            for ite in range(int(num_lines/num_queries_total)):
                temp = (np.unique(all_list[total]).shape[0])
                if temp > best_num:
                    best_num = temp
                    best_key = all_dict[total]
                total +=1
            #print(f'The best pair is {best_key} with num of matches {best_num}')
            best_key_split = re.split('.png|-' ,best_key)
            #print(best_key_split)
            result_list.append(best_key_split[1]== best_key_split[4])
        int_list = list(map(int, result_list))
        len_total = len(int_list)
        easy_list = int_list[:int(len_total/2)]; diff_list = int_list[int(len_total/2):]
        easy_score, diff_score =  (sum(easy_list)/len(easy_list) * 100), (sum(diff_list)/len(diff_list) * 100)
        return easy_score, diff_score
        #exit()
    #    #print(hfile.keys())
    #    dset = hfile['/query-iphone7-IMG_0801.JPG_cutouts_imageonly-DUC1-089-DUC_cutout_089_120_0.jpg']
    #    #dset = hfile['/query-iphone7-IMG_0801.JPG_cutouts_imageonly-DUC1-024-DUC_cutout_024_0_0.jpg']
    #    matches0 = dset['matches0']
    #    m0_np = np.array(matches0)
    #    print(np.unique(m0_np).shape[0])

        #matching_scores0 = dset['matching_scores0']
        #ms0_np = np.array(matching_scores0)
        #print(ms0_np.shape)

        #for k in dset:
            #print(k)
    #    grp = hfile['query']
    #    for k,v in grp.items():
    #        for k2, v2 in v.items():
    #            for k3, v3 in v2.items():
    #                print(k3, v3.__array__())
        #grp = hfile.create_group(image_name)
        #grp.create_dataset('global_descriptor', data=desc)

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

    num_lines_dict = {
    '0_mp3d_8WUmhLawc2A': 1984,
    '1_mp3d_EDJbREhghzL': 2624,
    '2_mp3d_i5noydFURQK': 1764,                                      
    '3_mp3d_jh4fc5c5qoQ': 940,              
    '4_mp3d_mJXqzFtmKg4': 3240,
    '5_mp3d_qoiz87JEwZ2': 3294,                                      
    '6_mp3d_RPmz2sHmrrY': 1560,            
    '7_mp3d_S9hNv5qa7GM': 3204,
    '8_mp3d_ULsKaCPVFJR': 960,                                        
    '9_mp3d_VzqfbhrpDEA': 3564,                                       
    '10_mp3d_wc2JMjhGNzB': 5568,                                      
    '11_mp3d_WYY7iVyf5p8': 720,            
    '12_mp3d_X7HyMhZNoso': 1708,                                      
    '13_mp3d_YFuZgdQ5vWj': 2304,                                     
    '14_mp3d_yqstnuAEVhm': 1296 
    }

    num_rooms_dict = {
    '0_mp3d_8WUmhLawc2A': 8,
    '1_mp3d_EDJbREhghzL': 8,
    '2_mp3d_i5noydFURQK': 7,
    '3_mp3d_jh4fc5c5qoQ': 5,
    '4_mp3d_mJXqzFtmKg4': 9,
    '5_mp3d_qoiz87JEwZ2': 9,
    '6_mp3d_RPmz2sHmrrY': 6,
    '7_mp3d_S9hNv5qa7GM': 9,
    '8_mp3d_ULsKaCPVFJR': 5,
    '9_mp3d_VzqfbhrpDEA': 9,
    '10_mp3d_wc2JMjhGNzB': 12,
    '11_mp3d_WYY7iVyf5p8': 5,
    '12_mp3d_X7HyMhZNoso': 7,
    '13_mp3d_YFuZgdQ5vWj': 8,
    '14_mp3d_yqstnuAEVhm': 6
    }

    debug=False
    if debug==True:
        scene_names = [
        '8WUmhLawc2A'
        ]
        
        folder_names = [
        '0_mp3d_8WUmhLawc2A'
        ]

    scene_name = scene_names[0]; folder_name = folder_names[0]
    scores_easy = []
    scores_diff = []
    for folder_name, scene_name in zip(folder_names, scene_names):
        score_easy, score_diff = main(scene_name, folder_name, num_lines_dict, num_rooms_dict)
        print(f"Score for {folder_name} is easy: {score_easy}; difficult {score_diff}")
        scores_easy.append(score_easy)
        scores_diff.append(score_diff)
    ave_easy = sum(scores_easy)/len(scores_easy)
    ave_diff = sum(scores_diff)/len(scores_diff)
    print(f"FINAL easy score: {scores_easy}, AVERAGE: {ave_easy} \n")
    print(f"FINAL diff score: {scores_diff}, AVERAGE: {ave_diff}")