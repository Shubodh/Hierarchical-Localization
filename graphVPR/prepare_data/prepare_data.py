import argparse
import glob
import os
from pathlib import Path
from shutil import copy2 #copy2 because metadata may be useful

def prep_building_level_localization(scene_names, folder_names, split_name):
    # Arguments need to be properly changed later.
    # This function takes INPUT from our graphVPR-mp3d standard dataset (See this https://iiitaphyd-my.sharepoint.com/personal/shubodh_sai_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshubodh%5Fsai%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2Fdata%2Freplica%2Dhabitat%2Fdata%5Fcollection%2Fx%2Dview%2Fmp3d)
    # and OUTPUTs the images in such a way needed for hloc code. 
    # You need to edit code (paths, scene_names etc) before `for` loop begins below 
    # depending on what/where you want. That's all. # This is that for loop - (`for scene_id, scene in enumerate(scene_names):`).

    # NOTE THAT this code will replace files if already present in output_path, but not entire folders.
    # So for example, if abc.png exists and you are writing same abc.png, it will replace old one.
    # BUT, if abcd.png exists in output path but you are NOT writing abcd.png, abcd.png won't get deleted.
    # so BEST PRACTICE is to: Just delete output directories if any files are already present and generate 
    # freshly everytime. Code takes just a min to run.
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--input_data', '-in', type=str, required=True)
#    parser.add_argument('--output_path', '-out', type=str, required=True)  
#    parser.add_argument('--query_small', '-q', type=bool, required=True)  
#    args = parser.parse_args()
#
#    inp = args.input_data
#    out = args.output_path
    # Be careful with "/" at the end of path
    inp = '/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/x-view-scratch/data_collection/x-view/mp3d/'
    out = '../../datasets/graphVPR/' + split_name + '/'
    env_name = '_mp3d_' # or _replica_
    query_ids_small = list(map(str, [0, 4])); query_ids_all = list(map(str, list(range(8))))
    query_ids_final = query_ids_small #if query_small=True

    for scene, folder in zip(scene_names, folder_names):
        print(f"Extracting for folder {folder} and room: ")
        input_path_a = Path(inp + scene + '/rooms/')
        assert input_path_a.exists(), input_path_a
        input_rooms = (list(input_path_a.iterdir()))
        for room_no, room in enumerate(input_rooms):
            print(f"    room {room.stem} ")
            room_rgb =  room / 'raw_data/'
            query_list = []
            rgb_files = list(room_rgb.glob('*_rgb.png'))
            print(f"           no of rgb files found in this room: {len(rgb_files)}")
            for rgb in rgb_files:
                rgb_stem = rgb.stem
                rgb_no = rgb_stem.replace("_rgb", "")
                if rgb_no in query_ids_all:
                    if rgb_no in query_ids_final:
                        #out_folder = out + 'query/' + str(scene_id) + env_name + scene + '/rooms/' + room.stem + '/'
                        out_folder = out + 'query/' + folder + '_' + room.stem + '/'
                        Path(out_folder).mkdir(parents=True, exist_ok=True)
                        out_path = Path(out_folder + str(rgb_stem + env_name + scene + '_' + room.stem + '.png')) #appending file name to output folder path
                        copy2(rgb, out_path)
                        
                else:
                    #out_folder = out + 'references/' + str(scene_id) + env_name + scene + '/rooms/' + room.stem + '/'
                    out_folder = out + 'references/' + folder + '_' + room.stem + '/'
                    Path(out_folder).mkdir(parents=True, exist_ok=True)
                    out_path = Path(out_folder + str(rgb_stem + env_name + scene + '_' + room.stem + '.png')) #appending file name to output folder path
                    copy2(rgb, out_path)

def write_pair_txt_room_level_localization(folder_name, scene_names, output_txt):
    # writes pairs txt files given folder names : in other words brute force pairs
    inp = '/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/datasets/graphVPR/room_level_localization_small/'

    ref_path = Path(inp + folder_name + '/' + 'references/')
    query_path = Path(inp + folder_name + '/' + 'query/')
    
    assert ref_path.exists(), ref_path
    assert query_path.exists(), query_path

    ref_rooms_path = (list(ref_path.iterdir()))
    query_rooms_path =  (list(query_path.iterdir()))
    print(f"{folder_name}: {len(ref_rooms_path)}")
#
#
    def list_imgs(rooms_path, ref):
        pref = 'references/' if ref==True else 'query/'
        fill_list = []
        for room in rooms_path:
            images = (list(room.iterdir()))
            for image in images:
                fill_list.append(pref + room.stem + '/' + image.name)
            
        return fill_list

    query_list = list_imgs(query_rooms_path, ref=False)
    ref_list = list_imgs(ref_rooms_path, ref=True)
#
    pairs = []
    for query in query_list:
        for ref in ref_list:
            pair = (query, ref)
            pairs.append(pair)

    with open(output_txt, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
    num_lines = len(ref_list) * len(query_list)
#    print(f"No of lines written to {output_txt} is {num_lines}")

def write_pair_txt_building_level_localization(split, output_txt):
    # writes pairs txt files given split names
    inp = '/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/datasets/graphVPR/'

    ref_path = Path(inp + split + '/' + 'references/')
    query_path = Path(inp + split + '/' + 'query/')
    
    assert ref_path.exists(), ref_path
    assert query_path.exists(), query_path

    ref_rooms_path = (list(ref_path.iterdir()))
    query_rooms_path =  (list(query_path.iterdir()))
    print(f"{split}: {len(ref_rooms_path)}")
#
#
    def list_imgs(rooms_path, ref):
        pref = 'references/' if ref==True else 'query/'
        fill_list = []
        for room in rooms_path:
            images = (list(room.iterdir()))
            for image in images:
                fill_list.append(pref + room.stem + '/' + image.name)
            
        return fill_list

    query_list = list_imgs(query_rooms_path, ref=False)
    ref_list = list_imgs(ref_rooms_path, ref=True)
#
    pairs = []
    for query in query_list:
        for ref in ref_list:
            pair = (query, ref)
            pairs.append(pair)

    with open(output_txt, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
    num_lines = len(ref_list) * len(query_list)
    print(f"No of lines written to {output_txt} is {num_lines}")


if __name__ == '__main__':
    scene_names = [
   "8WUmhLawc2A",
   "EDJbREhghzL",
   "i5noydFURQK",
   "jh4fc5c5qoQ",
   "mJXqzFtmKg4",
   "qoiz87JEwZ2",
   "RPmz2sHmrrY",
   "S9hNv5qa7GM",
   "ULsKaCPVFJR",
   "VzqfbhrpDEA",
   "wc2JMjhGNzB",
   "WYY7iVyf5p8",
   "X7HyMhZNoso",
   "YFuZgdQ5vWj",
   "yqstnuAEVhm"
]
    scene_names_split1 = [
   "8WUmhLawc2A",
   "EDJbREhghzL",
   "i5noydFURQK",
   "jh4fc5c5qoQ",
   "mJXqzFtmKg4",
   "qoiz87JEwZ2",
   "RPmz2sHmrrY",
   "S9hNv5qa7GM"
]
    scene_names_split2 = [
   "ULsKaCPVFJR",
   "VzqfbhrpDEA",
   "wc2JMjhGNzB",
   "WYY7iVyf5p8",
   "X7HyMhZNoso",
   "YFuZgdQ5vWj",
   "yqstnuAEVhm"
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
    folder_names_split1 = [
    '0_mp3d_8WUmhLawc2A',
    '1_mp3d_EDJbREhghzL',
    '2_mp3d_i5noydFURQK',
    '3_mp3d_jh4fc5c5qoQ',
    '4_mp3d_mJXqzFtmKg4',
    '5_mp3d_qoiz87JEwZ2',
    '6_mp3d_RPmz2sHmrrY',
    '7_mp3d_S9hNv5qa7GM',
    ]
    folder_names_split2 = [
    '8_mp3d_ULsKaCPVFJR',
    '9_mp3d_VzqfbhrpDEA',
    '10_mp3d_wc2JMjhGNzB',
    '11_mp3d_WYY7iVyf5p8',
    '12_mp3d_X7HyMhZNoso',
    '13_mp3d_YFuZgdQ5vWj',
    '14_mp3d_yqstnuAEVhm'
    ]

    Path('../../pairs/graphVPR/room_level_localization_small/').mkdir(exist_ok=True)
    output_txts = [
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-0_mp3d_8WUmhLawc2A.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-1_mp3d_EDJbREhghzL.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-2_mp3d_i5noydFURQK.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-3_mp3d_jh4fc5c5qoQ.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-4_mp3d_mJXqzFtmKg4.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-5_mp3d_qoiz87JEwZ2.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-6_mp3d_RPmz2sHmrrY.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-7_mp3d_S9hNv5qa7GM.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-8_mp3d_ULsKaCPVFJR.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-9_mp3d_VzqfbhrpDEA.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-10_mp3d_wc2JMjhGNzB.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-11_mp3d_WYY7iVyf5p8.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-12_mp3d_X7HyMhZNoso.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-13_mp3d_YFuZgdQ5vWj.txt',
    '../../pairs/graphVPR/room_level_localization_small/SP_SG_bruteforce/pairs-14_mp3d_yqstnuAEVhm.txt'
    ]

    output_txt_split1 = '../../pairs/graphVPR/building_level_small_split1/bruteforce.txt'
    output_txt_split2 = '../../pairs/graphVPR/building_level_small_split2/bruteforce.txt'

    Path("../../pairs/graphVPR/building_level_small_split1/").mkdir(parents=True, exist_ok=True)
    Path("../../pairs/graphVPR/building_level_small_split2/").mkdir(parents=True, exist_ok=True)

    # 1st function: Prepare building level localization data from your mp3d. Currently still doing room level one manually, but 
    # will hardly take 2-3 lines of code TODO
    split_name1 = 'building_level_small_split1'
    split_name2='building_level_small_split2'
    #prep_building_level_localization(scene_names_split1, folder_names_split1, split_name1)
    #prep_building_level_localization(scene_names_split2, folder_names_split2, split_name2)

    # 2nd function: Write pair txt files for  room level, in other words brute force pairs
    #print('writing pair texts in current folder itself')
    #for folder_name, output_txt in zip(folder_names, output_txts):
    #    write_pair_txt_room_level_localization(folder_name, scene_names, output_txt)

    #  3rd function: Write pair txt files for building-level, in other words brute force pairs
    write_pair_txt_building_level_localization(split_name1, output_txt_split1)
    write_pair_txt_building_level_localization(split_name2, output_txt_split2)