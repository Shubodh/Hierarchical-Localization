import h5py
from pathlib import Path
from pprint import pformat
import sys
import re

sys.path.append(str(Path(__file__).parent / '..')) #to import hloc
from hloc import extract_features, match_features, localize_inloc, visualization

from matplotlib import cm
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle

from hloc.utils.read_write_model import read_images_binary, read_points3D_binary
from hloc.utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from hloc.utils.io import read_image

def main(scene_name, folder_name, num_lines_dict, num_rooms_dict, matches):
    num_lines = num_lines_dict[folder_name]
    num_rooms = num_rooms_dict[folder_name]

    with h5py.File(matches, 'r') as hfile:
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
            if best_key_split[1]!= best_key_split[4]:
                print("incorrect: ", best_key_split)

        int_list = list(map(int, result_list))
        len_total = len(int_list)
        easy_list = int_list[:int(len_total/2)]; diff_list = int_list[int(len_total/2):]
        easy_score, diff_score =  (sum(easy_list)/len(easy_list) * 100), (sum(diff_list)/len(diff_list) * 100)
        return easy_score, diff_score

def main_viz(scene_name, folder_name, num_lines_dict, num_rooms_dict, matches, features):
    data_path = Path('../datasets/graphVPR/room_level_localization_small/') / folder_name
    assert data_path.exists(), data_path
    assert features.exists(), features
    assert matches.exists(), matches

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    num_lines = num_lines_dict[folder_name]
    num_rooms = num_rooms_dict[folder_name]

    with h5py.File(matches, 'r') as hfile:
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
            #if best_key_split[1]!= best_key_split[4]:
            if best_key_split[1]!= best_key_split[4]:
                #print("incorrect: ", best_key_split)
                feat_ind = best_key_split[0] + '/' + best_key_split[1] + '/' + best_key_split[2] + '.png'
                ref_ind = best_key_split[3].replace("_", "") + '/' + best_key_split[4] + '/' + best_key_split[5] + '.png'
    
                kpq = feature_file[feat_ind]['keypoints'].__array__()
                kpr = feature_file[ref_ind]['keypoints'].__array__()
                m = match_file[best_key]['matches0'].__array__()
                v = (m > -1)

                mkpq, mkpr = kpq[v], kpr[m[v]]

                q_image = read_image(data_path / feat_ind)
                r_image = read_image(data_path / ref_ind)

                #print(mkpq.shape, mkpr.shape)
                plot_images([q_image, r_image])
                plot_matches(mkpq, mkpr)
                plt.show()

        #int_list = list(map(int, result_list))
        #len_total = len(int_list)
        #easy_list = int_list[:int(len_total/2)]; diff_list = int_list[int(len_total/2):]
        #easy_score, diff_score =  (sum(easy_list)/len(easy_list) * 100), (sum(diff_list)/len(diff_list) * 100)

def print_results(retrieval_name, scene_names, folder_names, num_lines_dict, num_rooms_dict):
    feat_method = 'superpoint-n4096-r1600' #sift,  superpoint-n4096-r1600, d2net-ss
    match_method = 'superglue' #superglue, NN-mutual
    h5_suffix = '.h5'


    #scene_name = scene_names[0]; folder_name = folder_names[0]
    scores_easy = []
    scores_diff = []
    for folder_name, scene_name in zip(folder_names, scene_names):
#        matches = str(Path('../outputs/graphVPR/room_level_localization_small/' +retrieval_name[0]+ '/' + folder_name+ '/\
#feats-'+feat_method+ '_matches-' +match_method+ '_pairs-' +folder_name+h5_suffix))
        matches = str(Path('../outputs/graphVPR/room_level_localization_small/' +retrieval_name[1]+ '/' + scene_name+ '/\
feats-'+feat_method+ '_matches-' +match_method+ '_' +scene_name+h5_suffix))
        score_easy, score_diff = main(scene_name, folder_name, num_lines_dict, num_rooms_dict, matches)
        print(f"Score for {folder_name} is easy: {score_easy}; difficult {score_diff}")
        scores_easy.append(score_easy)
        scores_diff.append(score_diff)
    ave_easy = sum(scores_easy)/len(scores_easy)
    ave_diff = sum(scores_diff)/len(scores_diff)
    print(f"FINAL easy score: {scores_easy}, AVERAGE: {ave_easy} \n")
    print(f"FINAL diff score: {scores_diff}, AVERAGE: {ave_diff}")

def viz_results(scene_names, folder_names, num_lines_dict, num_rooms_dict):
    feat_method = 'superpoint-n4096-r1600' #sift,  superpoint-n4096-r1600, d2net-ss
    match_method = 'superglue' #superglue, NN-mutual
    h5_suffix = '.h5'

    for folder_name, scene_name in zip(folder_names, scene_names):
        matches = Path('../outputs/graphVPR/room_level_localization_small/' + folder_name+ '/\
feats-'+feat_method+ '_matches-' +match_method+ '_pairs-' +folder_name+h5_suffix)
        features = Path('../outputs/graphVPR/room_level_localization_small/'+folder_name+'/\
feats-'+feat_method + h5_suffix)
        main_viz(scene_name, folder_name, num_lines_dict, num_rooms_dict, matches, features)

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

    num_lines_dict_top3r = {
    '0_mp3d_8WUmhLawc2A': 48,
    '1_mp3d_EDJbREhghzL': 48,
    '2_mp3d_i5noydFURQK': 42,                                      
    '3_mp3d_jh4fc5c5qoQ': 30,              
    '4_mp3d_mJXqzFtmKg4': 54,
    '5_mp3d_qoiz87JEwZ2': 54,                                      
    '6_mp3d_RPmz2sHmrrY': 36,            
    '7_mp3d_S9hNv5qa7GM': 54,
    '8_mp3d_ULsKaCPVFJR': 30,                                        
    '9_mp3d_VzqfbhrpDEA': 54,                                       
    '10_mp3d_wc2JMjhGNzB': 72,                                      
    '11_mp3d_WYY7iVyf5p8': 30,            
    '12_mp3d_X7HyMhZNoso': 42,                                      
    '13_mp3d_YFuZgdQ5vWj': 48,                                     
    '14_mp3d_yqstnuAEVhm': 36 
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


    retrieval_name = ["SP_SG_bruteforce", "hist-top3r-1i"]
    print_results(retrieval_name,scene_names, folder_names, num_lines_dict_top3r, num_rooms_dict)
    #viz_results(scene_names, folder_names, num_lines_dict, num_rooms_dict)