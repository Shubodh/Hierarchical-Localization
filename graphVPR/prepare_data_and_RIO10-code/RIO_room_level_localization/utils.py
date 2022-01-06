import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
from pathlib import Path
from PIL import Image
import PyQt5
import random
from scipy.spatial.distance import cdist
import sys
from tqdm import tqdm
import yaml

def camera_intrinsics(camera_params):
        fx, fy, cx, cy  = camera_params['camera_intrinsics']['model']
        width, height = camera_params['camera_intrinsics']['width'], camera_params['camera_intrinsics']['height']
        cam_int = o3d.camera.PinholeCameraIntrinsic(width,height,fx,fy,cx,cy)
        #print(width, height, fx, fy, cx, cy)
        #print("Camera intrinsics matrix:")
        #print(cam_int.intrinsic_matrix)
        return cam_int

def frequency_list_to_dict(my_list):
    '''converts a list to dictionary where each value is frequency of an item.'''
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return freq

def semantics_dict_from_set(set_semantics):
    ''' creating a dict for semantics: unique ID for every semantic category'''
    dict_semantics = {}
    i_sem=0
    for set_s in set_semantics:
        dict_semantics.update({set_s:i_sem})
        i_sem = i_sem + 1
    return dict_semantics

def extract_RIO_instance_file(instance_filename):
    ''' extracting useful things like set_semantics(NOTE: returning it as an ordered list, not set()), dict_instances
    from instance_filename.
    Careful: RIO instance file is slightly different from Mp3d -->
    1. space between 2 word objects instead of _ (mp3d has _)
    2. 1-indexing instead of 0 (mp3d has 0)'''

    #set_semantics = set()  # not using set() as it doesn't preserve order.
    #  Using dict.fromkeys(list(dict_instances.values())) to replicate same behaviour. See below.
    dict_instances = {}

    with open(instance_filename) as fh:
        l = 1
        for line in fh:
            description = list(line.strip().split())

            if len(description)==3:
                desc = description[1] + "_" + description[2]
            elif len(description)==2:
                desc = description[1]
            #set_semantics.add(desc)
            dict_instances.update({description[0]:desc})

            l = l + 1
    

    oset_semantics = list(dict.fromkeys(list(dict_instances.values()))) # ordered set as list
    dict_semantics = semantics_dict_from_set(oset_semantics)

    return dict_instances, oset_semantics, dict_semantics