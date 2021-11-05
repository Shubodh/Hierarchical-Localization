# The idea here is: (Doesn't need GT query room IDs for this.)
# Similar to top3plus_room_discard.py but here, in addition to discarding images from 
# non-top3, we also consider ALL the room images from top3 rooms. To be more precise, every 3rd image, i.e. 0.jpg (ignoring -30 & 30).

import sys
from pathlib import Path
import re
sys.path.append(str(Path(__file__).parent / '../..')) #to import hloc
#import hloc
from hloc.utils.parsers import parse_retrieval, names_to_pair


if __name__ == '__main__':
    pairs = Path('../../pairs/inloc/')
    loc_pairs = pairs / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
    output_pairs = pairs / 'pairs-query-top4room_allimagesfromroom.txt'
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

        topK_rooms = 4 #top3 gives 27 images, top4 gives 33
        top_K_ids = sorted(room_freq, key=room_freq.get, reverse=True)[:topK_rooms]
        #print(top_K_ids) #['DUC1_023', 'DUC1_024', 'DUC1_022', 'DUC1_021']

        retrieval_dict_new = {}
        for building_room_id in top_K_ids:
            #print(r)
            r_split = re.split('_', str(building_room_id))
            #/data/InLoc_dataset/database/cutouts/DUC1/024/DUC_cutout_024_150_0.jpg
            base_path = "/data/InLoc_dataset/"
            middle_path = str("database/cutouts/" + r_split[0] + "/" + r_split[1])
            total_path = Path(base_path + middle_path)
            jpg_only0_files = list(total_path.glob('*_0.jpg'))

            for jpg_each in jpg_only0_files:

                retrieval_dict_new.setdefault(query, [])
                jpg_split = re.split('/data/InLoc_dataset/', str(jpg_each))
                retrieval_dict_new[query].append(jpg_split[1])

                pair = (query, jpg_split[1])
                pairs_output_txt.append(pair)

        #print(len(retrieval_dict_new[query]))
    print(pairs_output_txt)
    with open(output_pairs, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs_output_txt))
