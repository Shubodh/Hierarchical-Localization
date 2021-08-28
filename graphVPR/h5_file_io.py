import h5py
from pathlib import Path


features = Path('../outputs/graphVPR/room_level_localization_small/hist-top3r-1i/8WUmhLawc2A/global-feats-netvlad.h5')
with h5py.File(features, 'r') as hfile:
    all_list = []
    all_dict = {}
    key_num = 0
    for key in hfile.keys():
        dset = hfile['references/bathroom1/9_rgb_mp3d_8WUmhLawc2A_bathroom1.png']['global_descriptor'].__array__()
        print(dset, '\n')
#        for key2 in dset:
#            dset2 = dset[key2]
#            print(key2)
#        #matches0 = dset['matches0']