import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
import collections.abc as collections
import sys
from shutil import copy2

from utils.parsers import parse_image_lists, remove_suffix_from_list_of_files, dict_full_paths_without_rgb_from_partial_paths
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



def visual_debug_copy(testing_type_orgt, dict_query_ref_full_paths, room_id, scene_type, debug_print, start_f=0, end_f=5):
    """
    start_f, end_f = 100, 105 #0, 5
    Input: dict_query_ref_full_paths = {query1:[40 refs], query2: [40 refs],....} (full paths of files to copy)
    """
    #query_all_from = sorted(dict_query_ref_full_paths.keys())[start_f:end_f]
    #new_dict_five = {k: dict_query_ref_full_paths[k] for k in query_all_from}
    #print(new_dict_five)
    #print(start_f, end_f)

    # samply_path_to_copy = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/hloc/utils/distance_code_hloc/samply_visualization/"
    samply_path_to_copy = "/home/saishubodh/rrc_projects_2021/graphVPR_project/Hierarchical-Localization/hloc/utils/distance_code_hloc/samply_visualization/"+testing_type_orgt+ "/" + str(start_f) + "_" + str(end_f) + "/"
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
    
    query_all_from = sorted(dict_query_ref_full_paths.keys())[start_f:end_f]
    #query_all_from = sorted(dict_query_ref_full_paths.keys())[:5]
    new_dict_five = {k: dict_query_ref_full_paths[k] for k in query_all_from}
    for i in query_all_from:
        if not debug_print:
            copy2(Path(i+'.color.jpg'), Path(query_all))
        else:
            print(Path(i+'.color.jpg'), Path(query_all))
    count = 0
    for one_query, refs in new_dict_five.items():
        query_name_in_ref_folder = "AAquery_" + str(Path(one_query).name)
        if not debug_print:
            copy2(Path(one_query+'.color.jpg'), Path(refs_folders[count] + query_name_in_ref_folder +'.color.jpg'))
        else:
            print(Path(one_query+'.color.jpg'), Path(refs_folders[count] + query_name_in_ref_folder +'.color.jpg'))

        for ref in refs:
            if not debug_print:
                copy2(Path(ref+'.color.jpg'), Path(refs_folders[count]))
            else:
                print(Path(ref+'.color.jpg'), Path(refs_folders[count]))
        count += 1

    print(f"creating and copying to folders done, check {samply_path_to_copy}")


def main(testing_type_orgt, room_id, scene_type, descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    """
    Meaning of this file or o20andr20:
    Basically, for (ROI+ARRI-QOI), instead of directly NetVLAD40 which has only 1% rendered,
    do 20+20 (or 40+20) based on ranking of all (say) 1000 ref images, pick top 20 "original" and top 20 "rendered"
    images instead of top40 overall (which is usual thing).
    """
    logging.info('Extracting image pairs from a retrieval database.')

    if testing_type_orgt == "o20andr20":
        ite_till_o, ite_till_r = 20, 20
    elif testing_type_orgt == "o40andr40":
        ite_till_o, ite_till_r = 40, 40
    else:
        raise ValueError("incorrect configuration {testing_type_orgt} passed to this function that this function cannot handle")

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
    #topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    # New code
    num_matched_new = len(db_names) #5000
    #print(len(db_names))
    topk = torch.topk(sim, num_matched_new, dim=1).indices.cpu().numpy()

    #print(f"num_matched_new: {num_matched_new}")
    #print(topk.shape)
    
    #query_names = sorted(query_names) # this is wrong


    dict_query_ref_partial_paths = {}

    pairs = []
    for query, indices in zip(query_names, topk):
        iter_no_end = 0
        for i in indices:
            #print(db_names[i])
            if 'frame' in db_names[i]:
                iter_no_end += 1
                pair = (query, db_names[i])
                pairs.append(pair)
                
                # dict_query_ref_partial_paths[query] = db_names[i]
                dict_query_ref_partial_paths.setdefault(query, []).append(db_names[i])

                if iter_no_end == ite_till_o:
                    break
        
        iter_no_end = 0
        for i in indices:
            #print(db_names[i])
            if 'places' in db_names[i]:
                iter_no_end += 1
                pair = (query, db_names[i])
                pairs.append(pair)
                
                # dict_query_ref_partial_paths[query] = db_names[i]
                dict_query_ref_partial_paths.setdefault(query, []).append(db_names[i])

                if iter_no_end == ite_till_r:
                    break
        
        #print(len(pairs))
    logging.info(f'Found {len(pairs)} pairs.')

    debug_copy = False 
    if debug_copy:
        print('copying files')
        debug_print = False #If you actually want to copy, set to False
        dict_full = dict_full_paths_without_rgb_from_partial_paths(dict_query_ref_partial_paths, room_id, scene_type)
        starts_ends = [(0,5), (100,105), (200,205)]
        for start_f, end_f in starts_ends:
            print(start_f, end_f)
            visual_debug_copy(testing_type_orgt, dict_full, room_id, scene_type, debug_print, start_f, end_f)

    else:
        with open(output, 'w') as f:
            print(f"written pairs to {output}")
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


def main_only_oX0(testing_type_orgt, room_id, scene_type, descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    """
    Meaning of this file or o20andr20:
    Basically, for (ROI+ARRI-QOI), instead of directly NetVLAD40 which has only 1% rendered,
    do 20+20 (or 40+20) based on ranking of all (say) 1000 ref images, pick top 20 "original" and top 20 "rendered"
    images instead of top40 overall (which is usual thing).
    """
    logging.info('Extracting image pairs from a retrieval database.')

    if testing_type_orgt == "o20andgt20":
        #ite_till_o, ite_till_r = 20, 20
        ite_till_o = 20
    elif testing_type_orgt == "o40andgt40":
        #ite_till_o, ite_till_r = 40, 40
        ite_till_o = 40
    else:
        raise ValueError("incorrect configuration {testing_type_orgt} passed to this function that this function cannot handle")

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
    #topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    # New code
    num_matched_new = len(db_names) #5000
    #print(len(db_names))
    topk = torch.topk(sim, num_matched_new, dim=1).indices.cpu().numpy()

    #print(f"num_matched_new: {num_matched_new}")
    #print(topk.shape)
    


    dict_query_ref_partial_paths = {}

    pairs = []
    for query, indices in zip(query_names, topk):
        iter_no_end = 0
        for i in indices:
            #print(db_names[i])
            if 'frame' in db_names[i]:
                iter_no_end += 1
                pair = (query, db_names[i])
                pairs.append(pair)
                
                # dict_query_ref_partial_paths[query] = db_names[i]
                dict_query_ref_partial_paths.setdefault(query, []).append(db_names[i])

                if iter_no_end == ite_till_o:
                    break
        
        #iter_no_end = 0
        #for i in indices:
        #    #print(db_names[i])
        #    if 'places' in db_names[i]:
        #        iter_no_end += 1
        #        pair = (query, db_names[i])
        #        pairs.append(pair)
        #        
        #        # dict_query_ref_partial_paths[query] = db_names[i]
        #        dict_query_ref_partial_paths.setdefault(query, []).append(db_names[i])

        #        if iter_no_end == ite_till_r:
        #            break
        
        print(len(pairs))
    logging.info(f'Found {len(pairs)} pairs using main_only_oX0().')

    debug_copy = False 
    if debug_copy:
        print('copying files')
        debug_print = False #If you actually want to copy, set to False
        dict_full = dict_full_paths_without_rgb_from_partial_paths(dict_query_ref_partial_paths, room_id, scene_type)
        starts_ends = [(0,5), (100,105), (200,205)]
        for start_f, end_f in starts_ends:
            print(start_f, end_f)
            visual_debug_copy(testing_type_orgt, dict_full, room_id, scene_type, debug_print, start_f, end_f)

    # else:
    #     with open(output, 'w') as f:
    #         print(f"written pairs to {output}")
    #         f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

    return dict_query_ref_partial_paths

def main_oX0andgtX0_as_in_gt_radius_given_oX0(dict_oX0, testing_type_orgt, room_id, scene_type, output, num_matched, query_prefix, db_prefix):

    logging.info('Extracting image pairs from a retrieval database.')

    if testing_type_orgt == "o20andgt20":
        #ite_till_o, ite_till_r = 20, 20
        ite_till_o = 20
    elif testing_type_orgt == "o40andgt40":
        #ite_till_o, ite_till_r = 40, 40
        ite_till_o = 40
    else:
        raise ValueError("incorrect configuration {testing_type_orgt} passed to this function that this function cannot handle")

    dataset = Path('/data/InLoc_like_RIO10/sampling10/scene0'+ room_id + "_" + scene_type + '/')
    assert dataset.exists(), dataset 
    
    ref_files_path = dataset /   db_prefix[0]
    query_files_path = dataset / query_prefix[0]
    
    # ref_poses_files = sorted(list(ref_files_path.glob('*pose.txt')))
    ref_poses_files = sorted(list(ref_files_path.glob('*places*pose.txt')))
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

    #dict_query_ref_full_paths = {}
    dict_query_ref_full_paths = dict_full_paths_without_rgb_from_partial_paths(dict_oX0, room_id, scene_type)
    for target_file in query_files:
        iter_no =0
        final_set = set()
        final_list = bin_based_shortlisting(target_file, ref_files, iter_no, final_set)
        # dict_query_ref_full_paths[target_file] = final_list
        for each_final in final_list:
            dict_query_ref_full_paths.setdefault(target_file, []).append(each_final)

    if debug_copy:
        print('copying files')
        debug_print = False #If you actually want to copy, set to False
        visual_debug_copy(dict_query_ref_full_paths, room_id, scene_type, debug_print, start_f, end_f)
    else:
        write_dict_to_output_txt(dict_query_ref_full_paths, output, query_prefix, db_prefix)



if __name__ == "__main__":
    """
    Meaning of this file or o20andr20:
    Basically, for (ROI+ARRI-QOI), instead of directly NetVLAD40 which has only 1% rendered,
    do 20+20 (or 40+20) based on ranking of all (say) 1000 ref images, pick top 20 "original" and top 20 "rendered"
    images instead of top40 overall (which is usual thing).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_type_orgt', type=str, required=True)
    parser.add_argument('--room_id', type=str, required=True)
    parser.add_argument('--scene_type', type=str, required=True)
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)


    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()

    if args.testing_type_orgt == "o20andr20" or args.testing_type_orgt == "o40andr40":
        main(**args.__dict__)
    elif args.testing_type_orgt == "o40andgt40":
        dict_oX0 = main_only_oX0(**args.__dict__)
        main_oX0andgtX0_as_in_gt_radius_given_oX0(dict_oX0, args.testing_type_orgt, args.room_id, args.scene_type, args.output, args.num_matched, args.query_prefix, args.db_prefix)
