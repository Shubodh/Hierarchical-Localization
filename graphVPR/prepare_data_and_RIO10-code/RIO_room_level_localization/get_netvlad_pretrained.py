import os

import torchvision.models as models
import torch.nn as nn
import torch

from netvlad_pretrained import NetVLAD


ENCODER_DIM = 512
PRETRAINED = True
NUM_CLUSTERS = 64
VLADV2 = False
NGPU = 1
TRY_CUDA = True


def get_netvlad_pretrained(path):
    encoder = models.vgg16(pretrained=PRETRAINED)
    layers = list(encoder.features.children())[:-2]

    if PRETRAINED:
        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]: 
            for p in l.parameters():
                p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    net_vlad = NetVLAD(num_clusters=NUM_CLUSTERS, dim=ENCODER_DIM, vladv2=VLADV2)
    model.add_module('pool', net_vlad)

    is_cuda = TRY_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    isParallel = False
    if is_cuda and NGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    resume_ckpt = os.path.join(path, 'checkpoints', 'checkpoint.pth.tar')
    if os.path.isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_score']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
    else:
        raise FileNotFoundError('Checkpoint file not found')

    return model
        


                
