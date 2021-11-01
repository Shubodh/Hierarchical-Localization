# netvlad40-minustopKrooms
# The idea here is: (Doesn't need GT query room IDs for this.)
# Do NetVLAD40 even now and output dict of room_ids from these 40 images (dict 
# -> {building_room_id:freq of that room}, example {DUC1_002:6}), and choose only images of top3 freq
# rooms and ignore the rest of images. So instead of top40 images, it will be top34 or so.

import sys
from pathlib import Path
import re
sys.path.append(str(Path(__file__).parent / '../..')) #to import hloc
#import hloc
from hloc.utils.parsers import parse_retrieval, names_to_pair


if __name__ == '__main__':
    # SET BELOW
    pairs = Path('../../pairs/inloc/')
    loc_pairs = pairs / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
    output_pairs = pairs / 'pairs-query-netvlad40-minustop5rooms.txt'
    topK_rooms = 5 #top3 gives 27 images, top4 gives 33

    retrieval_dict = parse_retrieval(loc_pairs)
    queries = list(retrieval_dict.keys())
    pairs_output_txt = []

    #queries = [queries[0]] #FOR DEBUG
    for key_1 in queries:
        room_freq_dict = {}
        for r in retrieval_dict[key_1]:
            #print(r)
            r_split = re.split('/|.jpg', str(r))
            building_room_id = (r_split[2]+"_"+ r_split[3])
            room_freq_dict.setdefault(building_room_id, [])
            room_freq_dict[building_room_id].append(1)

        room_freq = {}
        for key, value_list in room_freq_dict.items():
            room_freq.setdefault(key, [])
            room_freq[key] = sum(value_list)
        #print(room_freq)

        top_K_ids = sorted(room_freq, key=room_freq.get, reverse=True)[:topK_rooms]

        retrieval_dict_new = {}
        for r in retrieval_dict[key_1]:
            #print(r)
            r_split = re.split('/|.jpg', str(r))
            building_room_id = (r_split[2]+"_"+ r_split[3])
            if building_room_id in top_K_ids: 
                retrieval_dict_new.setdefault(key_1, [])
                retrieval_dict_new[key_1].append(r)

                pair = (key_1, r)
                pairs_output_txt.append(pair)

        #print(len(retrieval_dict_new[key_1]))
    #print(pairs_output_txt)
#    with open(output_pairs, 'w') as f:
#        f.write('\n'.join(' '.join([i, j]) for i, j in pairs_output_txt))
