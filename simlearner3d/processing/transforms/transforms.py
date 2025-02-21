import math
import re
from typing import Dict, List

import numpy as np
import torch
from simlearner3d.utils.utils import Data
from simlearner3d.utils import utils

import secrets
import random

import torchvision.transforms as transforms

log = utils.get_logger(__name__)

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

__dataset_one_channel_stats = {'mean': 0.485, 'std': 0.229}

class ToTensor:
    """Turn np.arrays into Tensor."""

    def __init__(self, input: np.ndarray):
        self.input = input
    def __call__(self):
        data = torch.from_numpy(self.input)
        if data.dim()==2:
            data=data.unsqueeze(0)
        return data
    
class StandardizeIntensity:
    """Standardize gray scale image values."""

    def __call__(self, data:torch.Tensor):
        data = self.standardize_channel(data)
        return data

    def standardize_channel(self, channel_data: torch.Tensor, clamp_sigma: int = 3):
        """Sample-wise standardization y* = (y-y_mean)/y_std. clamping to ignore large values."""
        mean = channel_data.mean()
        std = channel_data.std() + 10**-6
        if torch.isnan(std):
            std = 1.0
        standard = (channel_data - mean) / std
        clamp = clamp_sigma * std
        clamped = torch.clamp(input=standard, min=-clamp, max=clamp)
        return clamped


class StandardizeIntensityCenterOnZero:
    """Standardize gray scale image values."""

    def __call__(self, data: Data):
        data._left = self.standardize_channel(data._left)
        data._right = self.standardize_channel(data._right)
        return data

    def standardize_channel(self, channel_data: torch.Tensor, min_channel=0.0,max_channel=255.0):
        res_data=channel_data.div(max_channel-min_channel).sub(0.5)
        return res_data
    

def normalize_radiometry(channel_data: torch.Tensor):
    max_value_t=torch.amax(channel_data) # assuming bs,3(channels),H,W
    min_value_t=torch.amin(channel_data)
    res_data=(channel_data.sub(min_value_t)).div(max_value_t-min_value_t+1e-8)
    #print(res_data.max(),res_data.min())
    
    assert(torch.all(res_data<=1) and torch.all(res_data>=0))
    return (res_data - __dataset_one_channel_stats['mean'])/__dataset_one_channel_stats['std']



"""def scale_crop(normalize=__imagenet_one_channel_stats):

    def normalize_radiometry(channel_data: torch.Tensor):
        max_value_t=torch.amax(channel_data) # assuming bs,3(channels),H,W
        min_value_t=torch.amin(channel_data)
        res_data=(channel_data.sub(min_value_t)).div(max_value_t-min_value_t+1e-8)
        print(res_data.max(),res_data.min())
        assert(torch.all(res_data<=1) and torch.all(res_data>=0))
        return res_data
    
    t_list = [
        normalize_radiometry,
        transforms.Normalize(**normalize),
    ]
    return transforms.Compose(t_list)"""


class StandardizeIntensityUsingDatasetStatistics:
    """Standardize gray scale image values."""

    def __call__(self, data: Data):
        data._left = normalize_radiometry(data._left)
        data._right = normalize_radiometry(data._right)
        return data
    
    



    
    