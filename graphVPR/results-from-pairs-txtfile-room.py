import h5py
from pathlib import Path
from pprint import pformat
import sys
import re

from matplotlib import cm
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
from collections import defaultdict

from tqdm import tqdm


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            q, r = p.split(' ')
            retrieval[q].append(r)
    return dict(retrieval)

def accuracy_room(dict_retrieval):
    result_list = []
    for query in dict_retrieval:
        query_split = re.split('/' ,query)
        val_split1_full = []
        for potential_ref in dict_retrieval[query]:
            val_split = re.split('/', potential_ref)
            val_split1_full.append(val_split[1])
        # Note that here the re.split('/') is different from building, Here it just gives room name,
        # for building, it gave scene name + room name. However, room name is sufficient here
        # You are just doing room level localization here,
        # i.e. within 1 scene matching, so just checking if room name is matching is sufficient to know
        # if it's correctly matched
        if query_split[1] in val_split1_full:
            result_list.append(1.0)
        else:
            result_list.append(0.0)
    score = (sum(result_list)/len(result_list) * 100)
    return score

if __name__ == '__main__':
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
    retrieval_name = ["bruteforce", "hist-top3r-1i", "netvlad-top40", "netvlad-top5","netvlad-top3", "netvlad-top1"]

    score_list = []
    
    #so retrieval_name[0] is bruteforce, i.e. has ALL pairs, for every query, every ref would exist in that pair txt file.
    # change this if your dataset is somewhere else
    for scene_name, folder_name in tqdm(zip(scene_names, folder_names)):
        print(f"currently folder: {folder_name}")
        dataset = Path('../datasets/graphVPR/room_level_localization_small/' + folder_name + '/')
        pairs = Path('../pairs/graphVPR/room_level_localization_small/')
        #loc_pairs = pairs / (retrieval_name[0]+ '/'+'pairs-' + folder_name + txt_suffix)  #'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
        loc_pairs = pairs / (retrieval_name[5]+ '/'+ scene_name + txt_suffix)  
        print(f"loaded loc_pair {loc_pairs}")
        dict_retrieval  = parse_retrieval(loc_pairs)
        score = accuracy_room(dict_retrieval)
        print(f"score for scene {folder_name} is {score}")
        score_list.append(score)
    
    all_score = (sum(score_list)/len(score_list) )
    print(f"the final score is {all_score}")
    print("NOTE THAT CURRENTLY COMBINING DIFFICULT AND EASY SCORES")