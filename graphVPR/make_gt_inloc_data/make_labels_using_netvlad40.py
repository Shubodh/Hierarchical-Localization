# netvlad40-minustopKrooms
# The idea here is: (Doesn't need GT query room IDs for this.)
# Do NetVLAD40 even now and output dict of room_ids from these 40 images (dict 
# -> {building_room_id:freq of that room}, example {DUC1_002:6}), and choose only images of top3 freq
# rooms and ignore the rest of images. So instead of top40 images, it will be top34 or so.

import sys
from pathlib import Path
import re
sys.path.append(str(Path(__file__).parent / '../..')) #to import hloc
from hloc.utils.parsers import parse_retrieval, names_to_pair


if __name__ == '__main__':
    # SET BELOW
    pairs = Path('../../pairs/inloc/')
    curr_dir = Path('./')
    loc_pairs = pairs / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
    topK_rooms = 1 #top3 gives 27 images, top4 gives 33
    out_end = 'labels_netvlad40-top' + str(topK_rooms) + '.txt'
    output_pairs = curr_dir / out_end

    retrieval_dict = parse_retrieval(loc_pairs)
    queries = list(retrieval_dict.keys())
    pairs_output_txt = []

    #queries = [queries[0]] #FOR DEBUG
    for query in queries:
        room_freq_dict = {}
        for r in retrieval_dict[query]:
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

        query_split = re.split('/', str(query))
        for top_K_id in top_K_ids:
            r_split = re.split('_', str(top_K_id))
            out_txt = (r_split[0] + "/" + r_split[1] + "/" + query_split[2])
            pairs_output_txt.append(out_txt)

    pairs_sorted_dict = {}
    for pair_out in pairs_output_txt:
        pair_out_split = re.split('_|.JPG', str(pair_out))
        pairs_sorted_dict.setdefault(pair_out_split[1], [])
        pairs_sorted_dict[pair_out_split[1]] = pair_out

    pairs_sorted_dict = dict(sorted(pairs_sorted_dict.items()))

        #print(len(retrieval_dict_new[query]))
    print(len(pairs_sorted_dict))
    with open(output_pairs, 'w') as f:
        f.write('\n'.join(j for i, j in pairs_sorted_dict.items()))
        print(f"written to {output_pairs}")
