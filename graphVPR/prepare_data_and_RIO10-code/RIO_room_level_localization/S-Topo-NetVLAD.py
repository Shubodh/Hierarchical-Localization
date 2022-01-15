import torch
import torch.nn as nn
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

def getMatchInds(ft_ref,ft_qry,topK=1,metric='cosine'):
    """
    This function takes two matrics and computes the distance between every vector in the matrices.
    For every query vector a ref. vector having shortest distance is retured
    """
    """
    metric: 'euclidean' or 'cosine' or 'correlation'
    """
    dMat = cdist(ft_ref,ft_qry,metric)
    mInds = np.argsort(dMat,axis=0)[:topK]        # shape: K x ft_qry.shape[0]
    return mInds

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
    print(f"DONE FOR NOW: Getting {accu} % accuracy.")

@torch.no_grad()
def topoNetVLAD(base_path, base_rooms, dim_descriptor_vlad, sample_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    featVect_tor = torch.zeros((len(base_rooms), dim_descriptor_vlad)).cuda()
    for i, base_room in enumerate(base_rooms):
        if base_path == sample_path:
            full_path_str = base_path + base_room
        else:
            full_path_str= base_path+ "scene"+ base_room[:2]+"/seq" +base_room[:2]+"/seq"+ base_room+ "/"
        full_path = Path(full_path_str)
        img_files = sorted(list(full_path.glob("*color.jpg")))
        for img in img_files:
            rgb = read_image(img)
            #rgb = rgb.astype(np.float32)
            #rgb= cv2.imread(str(img), cv2.IMREAD_COLOR)
            rgb_np = np.moveaxis(np.array(rgb), -1, 0)
            rgb_np = rgb_np[np.newaxis, :]
            x = torch.from_numpy(rgb_np).float().cuda()
            output = model(x) #batch_size = 1 currently, TODO: increase batch_size and change following code accordingly.
            featVect_tor[i] = featVect_tor[i] + output
            #print(img)
            #print("CURRENTLY HERE. NOw only sampling of images remaining")
        #sys.exit()
    featVect = featVect_tor.cpu().detach().numpy()
    return featVect

if __name__=='__main__':
    # 1. Given manual info
    num_clusters=32
    model, dim_ind = netvlad_model(num_clusters)
    dim_descriptor_vlad = (dim_ind*num_clusters)

    sample_path = "./sample_graphVPR_data/"
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
    base_path =sample_path #base_shublocal_path # base_shublocal_path #base_simserver_path

    # 3. Code starts
    if base_path == sample_path:
        base_rooms = sample_rooms 
        gt = gt_small
    elif base_path == base_shublocal_path:
        base_rooms=rescan_rooms_ids_small
        gt = gt_small
    elif base_path == base_simserver_path:
        base_rooms = rescan_rooms_ids

    featVect = topoNetVLAD(base_path, base_rooms, dim_descriptor_vlad, sample_path)
    mInds = getMatchInds(featVect, featVect, topK=2)
    predictions = mInds[1]
    accuracy(predictions, gt)
