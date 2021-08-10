import h5py
from pathlib import Path
import numpy as np


# DUmmy experiment: 
# 1. Replacing 4_rgb-8WUmhLawc2A-bathroom1.png query with random image.

# Set scene_name & matching method name
feat_method = 'd2net-ss' #sift,  superpoint-n4096-r1600, d2net-ss
match_method = 'NN-mutual' #superglue
scene_name = 'i5noydFURQK' #'8WUmhLawc2A', 'EDJbREhghzL', 'i5noydFURQK', 'jh4fc5c5qoQ', 'mJXqzFtmKg4'
# Set whether you want output for 2 queries or 2nd query or 1st query (small)
query_2 = '_only2query.h5'; queries_2 = '_2queries.h5'; small = '_small.h5'
h5_suffix = small 
# Next line might need to be changed for more queries, just do *2 for (1-easy + 1-difficult) aka _2queries
if h5_suffix==queries_2:
    num_lines_dict = {'8WUmhLawc2A': 992*2, 'EDJbREhghzL':1312*2, 'i5noydFURQK':882*2} 
else:
    num_lines_dict = {'8WUmhLawc2A': 992, 'EDJbREhghzL':1312, 'i5noydFURQK':882} 

num_rooms_dict = {'8WUmhLawc2A': 8, 'EDJbREhghzL':8, 'i5noydFURQK':7, 'jh4fc5c5qoQ':5, 'mJXqzFtmKg4':9}

#path = str(Path('outputs/inloc_small/feats-superpoint-n4096-r1600.h5'))
#path = str(Path('outputs/inloc_small/feats-superpoint-n4096-r1600_matches-superglue_pairs-query-netvlad40-custom-shub-small.h5'))
#path = str(Path('outputs/graphVPR/feats-superpoint-n4096-r1600_matches-superglue_pairs-query-graphVPR-samply.h5'))
path = str(Path('../outputs/graphVPR/mp3d_' + scene_name +'_small/\
feats-'+feat_method+ '_matches-' +match_method+ '_pairs-query-mp3d_' +scene_name+h5_suffix))


num_lines = num_lines_dict[scene_name]
num_rooms = num_rooms_dict[scene_name]


with h5py.File(path, 'r') as hfile:
    all_list = []
    all_dict = {}
    key_num = 0
    for key in hfile.keys():
        dset = hfile[key]
        matches0 = dset['matches0']
        m0_np = np.array(matches0)
        print(f"m0_np.shape {m0_np.shape}")
        print(m0_np[:30])
        all_list.append(m0_np)
        all_dict[key_num] = key 
        key_num +=1
    
#    print(all_dict)
#    exit()
    
    total = 0
    for outer in range(num_rooms):
        best_num=0
        best_key = ''
        ite = 0
        for ite in range(int(num_lines/num_rooms)):
            temp = (np.unique(all_list[total]).shape[0])
            if temp > best_num:
                best_num = temp
                best_key = all_dict[total]
            total +=1
        print(f'The best pair is {best_key} with num of matches {best_num}')
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
