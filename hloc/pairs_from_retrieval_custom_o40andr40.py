import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
import collections.abc as collections
import sys

from utils.parsers import parse_image_lists
from utils.read_write_model import read_images_binary
from utils.io import list_h5_names


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


def main(descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    """
    Meaning of this file or o20andr20:
    Basically, for (ROI+ARRI-QOI), instead of directly NetVLAD40 which has only 1% rendered,
    do 20+20 (or 40+20) based on ranking of all (say) 1000 ref images, pick top 20 "original" and top 20 "rendered"
    images instead of top40 overall (which is usual thing).
    """
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
    #topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    # New code
    num_matched_new = 5000
    topk = torch.topk(sim, num_matched_new, dim=1).indices.cpu().numpy()

    #print(f"num_matched_new: {num_matched_new}")
    #print(topk.shape)
    
    #query_names = sorted(query_names)
    pairs = []
    for query, indices in zip(query_names, topk):
        iter_no_end = 0
        for i in indices:
            #print(db_names[i])
            if 'frame' in db_names[i]:
                iter_no_end += 1
                pair = (query, db_names[i])
                pairs.append(pair)
                if iter_no_end == 40:
                    break
        
        iter_no_end = 0
        for i in indices:
            #print(db_names[i])
            if 'places' in db_names[i]:
                iter_no_end += 1
                pair = (query, db_names[i])
                pairs.append(pair)
                if iter_no_end == 40:
                    break
        
        #print(len(pairs))

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        print(f"written pairs to {output}")
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    """
    Meaning of this file or o20andr20:
    Basically, for (ROI+ARRI-QOI), instead of directly NetVLAD40 which has only 1% rendered,
    do 20+20 (or 40+20) based on ranking of all (say) 1000 ref images, pick top 20 "original" and top 20 "rendered"
    images instead of top40 overall (which is usual thing).
    """
    parser = argparse.ArgumentParser()
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
    main(**args.__dict__)
