# This code takes RIO10, and then makes a new dataset suitable for our graphVPR tasks. 
# see full details here: https://www.notion.so/saishubodh/Comprehensive-Room-level-Histogram-All-baselines-4235c83b3afe45ef97b5791d6e3b1f89#c1e4782baba848d8bc867a0573381cf7
# Getting confused with naming of folder hierarchy on disk? Link: https://www.notion.so/saishubodh/Comprehensive-Room-level-Histogram-All-baselines-4235c83b3afe45ef97b5791d6e3b1f89#f705ae1dd88349038f9d28d68fcb12ab
# Quick details: 
# REF: Choose seqAB_01 as ref data (every 30th image). So don't get confused when you look at folder 
# hierarchy on disk, _01 is implicit. You'll only see AB, i.e. RIO_xyz/references/seqAB/frame.jpg where AB -> 1 - 10
# QUERY: and corresponding seqAB_02, seqAB_03 etc (every 300th image) as query data independently 
# (every YY is independently treated as query data).
# similar to REF, here also _02 is implicit. With that being said, the parent folder i.e. _xyz above has
# necessary information. i.e. below 02w01 means seqAB_01 is REF and seqAB_02 is QUERY.
# new dataset path: ../../datasets/graphVPR/RIO10_Rescan02w01 and RIO10_Rescan03w01


from pathlib import Path
import os 
import re
from distutils.dir_util import copy_tree
from shutil import copy2 
import shutil 

def copy_data(paths, split, freqs):
    input_path = paths[0]; out_path = paths[1]
    ref_freq = freqs[0]; query_freq = freqs[1] 

    out_path_ref = out_path + split + "/references/"
    out_path_que = out_path + split + "/query/"
    Path(out_path_ref).mkdir(parents=True, exist_ok=True)
    Path(out_path_que).mkdir(parents=True, exist_ok=True)
    ref_rescan_ID = "01" #always 01
    if split == "RIO10_Rescan02w01":
        query_rescan_ID = "02"
    else:
        query_rescan_ID = "03"

    for scene_path in os.listdir(input_path):
        scene_path_temp = input_path + scene_path + "/"
        scene_id =  re.split('scene' ,scene_path)
        print(f"Scene ID found: {scene_id[1]} at path {scene_path_temp}")
        scene_path_f = scene_path_temp + "seq" + str(scene_id[1]) + "/"

        # Copying ref data
        scene_path_ref = Path(scene_path_f + "seq" + str(scene_id[1]) + "_" + ref_rescan_ID + "/")
        print(f"creating dataset: REF -> seq {scene_id[1]}_{ref_rescan_ID} at {out_path_ref}seq{scene_id[1]}/")
        rgb_files = sorted(list(scene_path_ref.glob('*color.jpg')))
        for rgb_file in rgb_files:
            img_id_full = re.split('frame-|.color.jpg', str(rgb_file))
            img_id = img_id_full[1]
            if int(img_id) % ref_freq == 0:
                Path(out_path_ref + "seq" + str(scene_id[1]) + "/").mkdir(parents=True, exist_ok=True)
                out_path_ref_final = Path(out_path_ref + "seq" + str(scene_id[1]) + "/" + rgb_file.stem + '.jpg') #appending file name to output folder path
                #print(rgb_file, out_path_ref_final)
                copy2(rgb_file,out_path_ref_final)

        # Copying query data
        scene_path_query = Path(scene_path_f + "seq" + str(scene_id[1]) + "_" + query_rescan_ID + "/")
        print(f"creating dataset: QUERY -> seq {scene_id[1]}_{query_rescan_ID} at {out_path_que}seq{scene_id[1]}/")
        rgb_files = sorted(list(scene_path_query.glob('*color.jpg')))
        for rgb_file in rgb_files:
            img_id_full = re.split('frame-|.color.jpg', str(rgb_file))
            img_id = img_id_full[1]
            if int(img_id) % query_freq == 0:
                Path(out_path_que + "seq" + str(scene_id[1]) + "/").mkdir(parents=True, exist_ok=True)
                out_path_que_final = Path(out_path_que + "seq" + str(scene_id[1]) + "/" + rgb_file.stem + '.jpg') #appending file name to output folder path
                #print(rgb_file, out_path_que_final)
                copy2(rgb_file,out_path_que_final)

if __name__ == '__main__':
    ref_freq = 30 #every 30th image
    query_freq = 90 #every 90th image

    input_path = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/"
    out_path = "../../datasets/graphVPR/" # will be creating new splits for DUC1 and DUC2 named DUC1_graphVPRsplit and DUC2_graphVPRsplit here
    # set the above
    split02w01 = "RIO10_Rescan02w01" 
    split03w01 = "RIO10_Rescan03w01" 

    Note = """IMPORTANT NOTE: Running this script won't overwrite the folders as a whole if there is an existing
    folder at output destination. Rather it will append the images to existing folders. This
    is undesirable behaviour. So you MUST delete output folders if they exist already, i.e. manually delete ../../datasets/graphVPR/RIO10_Rescan02w01 etc..
    Enter 'y' after you delete: """

    val = input(Note)
    if val != 'y' :
        raise ValueError(f"Your input must be y, instead got {val}")

    paths = [input_path, out_path]
    freqs = [ref_freq, query_freq]

    copy_data(paths, split02w01, freqs)
    copy_data(paths, split03w01, freqs)