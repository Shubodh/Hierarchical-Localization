# The idea here is: (Doesn't need GT query room IDs for this.)
# Similar to top3plus_room_discard.py but here, in addition to discarding images from 
# non-top3, we also consider ALL the room images from top3 rooms.

import sys
from pathlib import Path
import re
sys.path.append(str(Path(__file__).parent / '../..')) #to import hloc
#import hloc
from hloc.utils.parsers import parse_retrieval, names_to_pair


if __name__ == '__main__':
    pairs = Path('../../pairs/inloc/')
    loc_pairs = pairs / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
    output_pairs = pairs / 'pairs-query-netvlad40-minustopKrooms.txt'
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

        retrieval_dict_new = {}
        for r in retrieval_dict[query]:
            #print(r)
            r_split = re.split('/|.jpg', str(r))
            base_path = Path(r).parents[0]
            print(base_path)
            sys.exit()
            rgb_files = list(room_rgb.glob('*_0.jpg'))
            building_room_id = (r_split[2]+"_"+ r_split[3])
            room_rgb =  room / 'raw_data/'
            query_list = []
            if building_room_id in top_K_ids: 
                retrieval_dict_new.setdefault(query, [])
                retrieval_dict_new[query].append(r)

                pair = (query, r)
                pairs_output_txt.append(pair)

        #print(len(retrieval_dict_new[query]))
    print(pairs_output_txt)
    #with open(output_pairs, 'w') as f:
    #    f.write('\n'.join(' '.join([i, j]) for i, j in pairs_output_txt))