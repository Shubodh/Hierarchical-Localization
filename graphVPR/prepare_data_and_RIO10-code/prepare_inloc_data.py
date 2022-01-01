# This code takes inloc DUC1 and DUC2 reference datasets, and then makes a new dataset where
# DUC1 and DUC2 are two different independent splits
# query image: first image of every subfolder (or room) in say DUC1.
# reference images: rest of images
# new dataset path: ../../datasets/graphVPR/DUC1_graphVPRsplit and DUC2_graphVPRsplit


from pathlib import Path
import os 
from distutils.dir_util import copy_tree
import shutil 

if __name__ == '__main__':
    splits = ["DUC1", "DUC2"]
    split = splits[1]
    input_path = "../../datasets/inloc/cutouts_imageonly/" + split + "/"
    out_path = "../../datasets/graphVPR/" # will be creating new splits for DUC1 and DUC2 named DUC1_graphVPRsplit and DUC2_graphVPRsplit here
    # set the above
    out_path_ref = out_path + split + "_graphVPRsplit/references/"
    out_path_que = out_path + split + "_graphVPRsplit/query/"
    Path(out_path_ref).mkdir(parents=True, exist_ok=True)
    Path(out_path_que).mkdir(parents=True, exist_ok=True)

    #TODO: Uncomment below
    copy_tree(input_path, out_path_ref) # This is 1.

    # 1. Prepare reference folders: Just copy paste from input_path 
    # 2. and query folder: move the first image from every room into query folder

#shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")

    for room_path in os.listdir(out_path_ref):
        room_path_f = out_path_ref + room_path + "/"
        out_path_que_room = out_path_que + room_path + "/"
        Path(out_path_que_room).mkdir(parents=True,exist_ok=True)
        img_name = "DUC_cutout_" + room_path + "_0_0.jpg"
        print("Moving: " + room_path_f + img_name, "TO: " + out_path_que_room + img_name)
        shutil.move(room_path_f + img_name, out_path_que_room + img_name)