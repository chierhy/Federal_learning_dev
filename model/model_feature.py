# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 09:49:00 2021

@author: Xiao Peng
"""

import numpy as np
import torch

net0 = torch.load('trained_model/net_xiao10.pkl')
#net0.eval()


import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


'''方法1，直接可视化'''
import torch
import torchvision
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from matplotlib import pyplot as plt
import math

#print(net0)
conv_layers = []
model_weights = []
model_children = list(net0.children())
counter = 0

for i in range(len(model_children)):
    #print(i)
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            if type(model_children[i][j])==nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i][j].weight)
                conv_layers.append(model_children[i][j])

print(counter)

for i in range(4):
    print(model_weights[i])
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.imshow(model_weights[0][i][0, :, :].detach(), cmap='gray')
plt.show()
