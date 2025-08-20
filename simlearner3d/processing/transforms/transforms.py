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
        res_data=((channel_data.sub(min_channel)).div(max_channel-min_channel)).sub(0.5)
        return res_data
    

def normalize_radiometry(channel_data: torch.Tensor, nb_channel: int =1):
    max_value_t=torch.amax(channel_data) # assuming H,W  --> gray images
    min_value_t=torch.amin(channel_data)
    res_data=(channel_data.sub(min_value_t)).div(max_value_t-min_value_t+1e-8)
    #print(res_data.max(),res_data.min())
    assert(torch.all(res_data<=1) and torch.all(res_data>=0))
    if nb_channel==1:
        return (res_data - __dataset_one_channel_stats['mean'])/__dataset_one_channel_stats['std']
    else:
        res_data=res_data.tile((nb_channel,1,1)) # nb_channel, H, W
        # standardize 
        standardize_fct=transforms.Normalize(**__imagenet_stats)
        return standardize_fct(res_data)
         
        
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
    def __init__(self,nb_channel: int =1):
        self.channels=nb_channel

    def __call__(self, data: Data):
        data._left = normalize_radiometry(data._left,nb_channel=self.channels)
        data._right = normalize_radiometry(data._right,nb_channel=self.channels)
        return data
    
    



    
    