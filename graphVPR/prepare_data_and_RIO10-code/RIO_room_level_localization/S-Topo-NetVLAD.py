import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from netvlad import NetVLAD
from netvlad import EmbedNet
#from hard_triplet_loss import HardTripletLoss
from torchvision.models import resnet18
from scipy.spatial.distance import cdist

import cv2
import numpy as np
import glob
from pathlib import Path

import sys

from utils import read_image
from matching import getMatchInds, getMatchIndsTfidf, getMatchIndsBinary, getMatchIndsBinaryTfidf
from get_netvlad_pretrained import get_netvlad_pretrained, ENCODER_DIM, NUM_CLUSTERS

def netvlad_model(num_clusters=32):
    # Discard layers at the end of base network
    encoder = resnet18(pretrained=True)
    #print(encoder)
    base_model = nn.Sequential(
        encoder.conv1,
        encoder.bn1,
        encoder.relu,
        encoder.maxpool,
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4,
    )
    dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=num_clusters, dim=dim, alpha=1.0)
    #print(f"descriptor dimension, num_clusters: {dim, num_clusters}")
    #print(f"final vector dimension would be: dim * num_clusters: {dim * num_clusters}")
    model = EmbedNet(base_model, net_vlad).cuda()

    # Define loss
    #criterion = HardTripletLoss(margin=0.1).cuda()
    #output = model(x)
    #triplet_loss = criterion(output, labels)

    #print(f"input, output.shape {x.shape, output}")
    #print(f"labels: {labels}")
    #print(f"triplet_loss {triplet_loss}")
    #rgb_np = np.moveaxis(np.array(rgb), -1, 0)
    #rgb_np = rgb_np[np.newaxis, :]
    #print(rgb_np.shape)

    return model, dim

def accuracy(predictions, gt):
    accVec = np.equal(predictions, gt)
    accu = np.sum(accVec) / accVec.shape[0]  * 100
    print(f"predictions: {predictions}") 
    print(f"Ground truth: {gt}")
    print(f"ACCURACY:  {accu} %. TODO-Later: Check your ground truth, may not be entirely right. i.e. mInds[1] could point to itself too. i.e. for 1, 0 or 1 could be considered as true... not sure, check later")

def norm_topoNetVLAD(featVect, num_clusters,feat_dim):
    # print("DEBUG")
    # print(featVect)
    for i in range(featVect.shape[0]):
        vlad = featVect[i]
        input_shape = vlad.shape
        vlad = vlad.reshape((num_clusters,feat_dim))
        vlad = F.normalize(vlad, p=2, dim=1)  # intra-normalization
        vlad = vlad.view(-1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=0)  # L2 normalize
        featVect[i] = vlad

    # print(featVect)
    # sys.exit()
    return featVect

def average_topoNetVLAD(featVect, num_images_per_room, num_clusters, feat_dim):
    # print("DEBUG")
    # print(featVect)
    for i in range(featVect.shape[0]):
        featVect[i] = featVect[i] / num_images_per_room[i]
    # print(featVect)
    # sys.exit()
    return featVect

@torch.no_grad()
def topoNetVLAD(model, base_path, base_rooms, dim_descriptor_vlad, num_clusters,feat_dim, sample_path, sampling_freq, batch_size, norm_bool):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    featVect_tor = torch.zeros((len(base_rooms), dim_descriptor_vlad)).cuda()
    num_images_per_room = []
    for i_room, base_room in enumerate(base_rooms):
        if base_path == sample_path:
            full_path_str = base_path + base_room
        else:
            full_path_str= base_path+ "scene"+ base_room[:2]+"/seq" +base_room[:2]+"/seq"+ base_room+ "/"
        full_path = Path(full_path_str)
        img_files_all = sorted(list(full_path.glob("*color.jpg")))
        img_files = img_files_all[::sampling_freq] # every 1000th image
        print(f"No. of sampling images in this {base_room} room: {len(img_files)}")
        num_images_per_room.append(len(img_files))
        #print(f" and their names: {img_files}")
        x_all = []
        for i_img, img in enumerate(img_files):
            rgb = read_image(img)
            rgb = rgb.astype(np.float32)
            rgb_np = np.moveaxis(np.array(rgb), -1, 0)
            rgb_np = rgb_np[np.newaxis, :]
            x = torch.from_numpy(rgb_np).float().cuda()
            x_all.append(x)
            if (i_img+1) % batch_size == 0:
                #x_all_batch = torch.cat(x_all[i_img+1 - batch_size:i_img+1], 0) # if x_all=[] is commented
                x_all_batch = torch.cat(x_all, 0).to(device) # if below x_all=[] is uncommented
                output = model.encoder(x_all_batch)
                output = model.pool(output)
                print('OUTPUT SHAPE', output.size(), flush=True)
                featVect_tor[i_room] = featVect_tor[i_room] + torch.sum(output, 0)
                x_all = []
        #print("CURRENTLY HERE. NOw only sampling of images remaining")
    if norm_bool:
        torch.set_printoptions(profile="full")
        #print(f"Before norm: {featVect_tor[0:3,:20]}")
        #featVect_tor = norm_topoNetVLAD(featVect_tor, num_clusters,feat_dim)
        print("doing average topoNetVLAD")
        featVect_tor = average_topoNetVLAD(featVect_tor, num_images_per_room, num_clusters,feat_dim)
        #print(f"After norm: {featVect_tor[0:3,:20]}")
    featVect = featVect_tor.cpu().detach().numpy()
    return featVect

if __name__=='__main__':
    # 1. Given manual info
    checkpoint_path = ''
    model = get_netvlad_pretrained(checkpoint_path)
    dim_descriptor_vlad = (ENCODER_DIM*NUM_CLUSTERS)

    sample_path = "./sample_graphVPR_data/"
    # Note: Use sampling_freq=1 for 100% accuracy on sample_path. Do note that this doesn't seem deterministic
    # so don't always expect 100%.
    base_shublocal_path = "/home/shubodh/Downloads/data-non-onedrive/RIO10_data/"
    base_simserver_path = "/home/shubodh/hdd1/Shubodh/Downloads/data-non-onedrive/RIO10_data/"
    base_adaserver_path = "/data/RIO10_data/"

    sample_rooms = ['01', '02', '03', '04']
    rescan_rooms_ids_small = ['01_01', '01_02', '02_01', '02_02']
    rescan_rooms_ids = ['01_01', '01_02', '02_01', '02_02', '03_01', '03_02', '04_01', '04_02', '05_01', '05_02',
                        '06_01', '06_02', '07_01', '07_02', '08_01', '08_02', '09_01', '09_02', '10_01', '10_02']
    gt = np.array([1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18])
    gt_small = np.array([1,0,3,2])

    # 2. TO SET: Set just the next line
    base_path =base_adaserver_path #sample_path  # base_shublocal_path #base_simserver_path
    sampling_freq = 25# 1 #1
    batch_size = 8 #3 for sample_path, 32 for all else
    norm_bool = True 
    tf_idf = True

    # 3. Code starts
    if base_path == sample_path:
        base_rooms = sample_rooms 
        gt = gt_small
    elif base_path == base_shublocal_path:
        base_rooms=rescan_rooms_ids_small
        gt = gt_small
    else:
        base_rooms = rescan_rooms_ids

    featVect = topoNetVLAD(model, base_path, base_rooms, dim_descriptor_vlad, NUM_CLUSTERS,ENCODER_DIM,
                        sample_path, sampling_freq, batch_size, norm_bool)
    if tf_idf:
        mInds = getMatchIndsTfidf(featVect, featVect, topK=2)
    else:
        mInds = getMatchInds(featVect, featVect, topK=2)

    predictions = mInds[1]
    accuracy(predictions, gt)
    print(f"mInds[0] sanity check {mInds[0]}")
    print(f"Do note that this accuracy is for sampling_freq {sampling_freq}, batch_size {batch_size}, norm {norm_bool}, tf-idf {tf_idf}")
