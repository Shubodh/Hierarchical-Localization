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

def accuracy(dict_retrieval):
    result_list = []
    for query in dict_retrieval:
        query_split = re.split('/' ,query)
        val_split1_full = []
        for potential_ref in dict_retrieval[query]:
            val_split = re.split('/', potential_ref)
            val_split1_full.append(val_split[1])
        if query_split[1] in val_split1_full:
            result_list.append(1.0)
        else:
            result_list.append(0.0)
    score = (sum(result_list)/len(result_list) * 100)
    print(f"the score is {score}")

if __name__ == '__main__':
    txt_suffix = '.txt'
    h5_suffix = '.h5'
    split_names = ['building_level_small_split1', 'building_level_small_split2']
    split_names_DUC = ['DUC1_graphVPRsplit', 'DUC2_graphVPRsplit']
    split_names = split_names_DUC

    retrieval_name = ["bruteforce", "hist-top3r-1i", "netvlad-top40", "netvlad-top5","netvlad-top3", "netvlad-top1"]
    #so retrieval_name[0] is bruteforce, i.e. has ALL pairs, for every query, every ref would exist in that pair txt file.
    # change this if your dataset is somewhere else
    for split in split_names:
        print(f"currently split: {split}")
        pairs = Path('../pairs/graphVPR/' + split + '/')
        loc_pairs = pairs / (retrieval_name[2] + txt_suffix)  
        print(f"loaded loc_pair {loc_pairs}")
        dict_retrieval  = parse_retrieval(loc_pairs)
        accuracy(dict_retrieval)