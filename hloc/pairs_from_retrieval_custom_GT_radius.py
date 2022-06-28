import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
import collections.abc as collections
import sys
from shutil import copy2

from utils.parsers import parse_image_lists, remove_suffix_from_list_of_files
from utils.read_write_model import read_images_binary
from utils.io import list_h5_names, write_dict_to_output_txt

from utils.distance import bin_based_shortlisting

def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
    elif names is not None and isinstance(names, (str, Path)):
        names = parse_image_lists(names)
    elif names is not None and isinstance(names, collections.Iterable):
        names = list(names)
    else:
        raise ValueError('Provide either prefixes of names, a list of '
                         'images, or a path to list file.')
    return names


def main_original(descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    logging.info('Extracting image pairs from a retrieval database.')

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = read_images_binary(db_model / 'images.bin')
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
        if name2idx is None:
            with h5py.File(str(path), 'r') as fd:
                desc = [fd[n][key].__array__() for n in names]
        else:
            desc = []
            for n in names:
                with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                    desc.append(fd[n][key].__array__())
        return torch.from_numpy(np.stack(desc, 0)).to(device).float()

    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum('id,jd->ij', query_desc, db_desc)
    topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    pairs = []
    for query, indices in zip(query_names, topk):
        for i in indices:
            pair = (query, db_names[i])
            pairs.append(pair)

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


def visual_debug_copy(dict_query_ref_full_paths, room_id, scene_type, debug_print, start_f, end_f):
    """
    start_f, end_f = 100, 105 #0, 5
    Input: dict_query_ref_full_paths = {query1:[40 refs], query2: [40 refs],....} (full paths of files to copy)
    """
    # samply_path_to_copy = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/hloc/utils/distance_code_hloc/samply_visualization/"
    samply_path_to_copy = "/home/saishubodh/rrc_projects_2021/graphVPR_project/Hierarchical-Localization/hloc/utils/distance_code_hloc/samply_visualization/" + str(start_f) + "_" + str(end_f) + "/"
    room_scene_folder = samply_path_to_copy +  'scene0'+ room_id + "_" + scene_type + '/'
    query_all     = room_scene_folder + 'query_all'     + '/'
    query1_40refs = room_scene_folder + 'query1_40refs' + '/'
    query2_40refs = room_scene_folder + 'query2_40refs' + '/'
    query3_40refs = room_scene_folder + 'query3_40refs' + '/'
    query4_40refs = room_scene_folder + 'query4_40refs' + '/'
    query5_40refs = room_scene_folder + 'query5_40refs' + '/'

    mkdir_folders = [room_scene_folder, query_all, query1_40refs, query2_40refs, query3_40refs, query4_40refs, query5_40refs]
    refs_folders = [query1_40refs, query2_40refs, query3_40refs, query4_40refs, query5_40refs]
    for each_fol in mkdir_folders:
        Path(each_fol).mkdir(parents=True, exist_ok=True)
    
    query_all_from = sorted(dict_query_ref_full_paths.keys())[:5]
    new_dict_five = {k: dict_query_ref_full_paths[k] for k in query_all_from}
    for i in query_all_from:
        print(Path(i+'.color.jpg'), Path(query_all))
        if not debug_print:
            copy2(Path(i+'.color.jpg'), Path(query_all))
    count = 0
    for one_query, refs in new_dict_five.items():
        query_name_in_ref_folder = "AAquery_" + str(Path(one_query).name)
        print(Path(one_query+'.color.jpg'), Path(refs_folders[count] + query_name_in_ref_folder + '.color.jpg'))
        if not debug_print:
            copy2(Path(one_query+'.color.jpg'), Path(refs_folders[count] + query_name_in_ref_folder + '.color.jpg'))
        for ref in refs:
            print(Path(ref+'.color.jpg'), Path(refs_folders[count]))
            if not debug_print:
                copy2(Path(ref+'.color.jpg'), Path(refs_folders[count]))
        count += 1


def main(room_id, scene_type, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    logging.info('Extracting image pairs from a retrieval database.')

    dataset = Path('/data/InLoc_like_RIO10/sampling10/scene0'+ room_id + "_" + scene_type + '/')
    assert dataset.exists(), dataset 
    
    ref_files_path = dataset /   db_prefix[0]
    query_files_path = dataset / query_prefix[0]
    
    ref_poses_files = sorted(list(ref_files_path.glob('*pose.txt')))
    query_poses_files = sorted(list(query_files_path.glob('*pose.txt')))
    
    ref_files_prefix_str = remove_suffix_from_list_of_files(ref_poses_files)
    query_files_prefix_str = remove_suffix_from_list_of_files(query_poses_files)

    # print(len(ref_files_prefix_str), ref_files_prefix_str[:10])#, '\n', len(query_poses_files),  query_poses_files[:10])
    # print(len(query_files_prefix_str), query_files_prefix_str[:10])#, '\n', len(query_poses_files),  query_poses_files[:10])

    debug_copy = False 
    start_f, end_f = 200, 205 #100, 105 #0, 5 # Do till 200
    if debug_copy:
        query_files, ref_files = query_files_prefix_str[start_f:end_f], np.array(ref_files_prefix_str) #[0:50]
        print(query_files)
        # print("final dict")
        # print(dict_query_ref_full_paths)
    else:
        query_files, ref_files = query_files_prefix_str, np.array(ref_files_prefix_str) #[0:50]

    dict_query_ref_full_paths = {}
    for target_file in query_files:
        iter_no =0
        final_set = set()
        final_list = bin_based_shortlisting(target_file, ref_files, iter_no, final_set)
        dict_query_ref_full_paths[target_file] = final_list

    if debug_copy:
        print('copying files')
        debug_print = False #If you actually want to copy, set to False
        visual_debug_copy(dict_query_ref_full_paths, room_id, scene_type, debug_print, start_f, end_f)
    else:
        write_dict_to_output_txt(dict_query_ref_full_paths, output, query_prefix, db_prefix)


if __name__ == "__main__":
    # print("python3 pairs_from_retrieval_custom_GT_radius.py 
    # --room_id "+room_id+" --scene_type "+scene_type+" 
    # --num_matched "+ num_matched + 
    # " --output ../pairs/graphVPR/rio_metric/scene0"+room_id +"/netvlad"+num_matched+"_scene_GT_radius_"+
    # scene_type +"_sampling10_"+dt + ".txt --query_prefix query/ --db_prefix database/cutouts/")
    parser = argparse.ArgumentParser()
    # parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--room_id', type=str, required=True)
    parser.add_argument('--scene_type', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)

    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
