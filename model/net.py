from __future__ import absolute_import, division, print_function
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
from .model.MSHENet import model_construct

warnings.filterwarnings("ignore")


class Net(nn.Module):

    def __init__(self, params):

        super(Net, self).__init__()

        self.crop_size = params.crop_size

        self.model = model_construct(normalize='relu_l2norm',
                                     normalize_features=True,
                                     cyclic_consistency=True)

    def forward(self, source, target, source_256, target_256):
        # parse input

        output_256, output = self.model(source, target, source_256, target_256)

        return output_256, output


def fetch_net(params):
    if params.net_type == "basic":
        net = Net(params)
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return net