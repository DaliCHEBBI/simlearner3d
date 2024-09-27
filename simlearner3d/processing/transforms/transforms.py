import math
import re
from typing import Dict, List

import numpy as np
import torch
from simlearner3d.utils.utils import Data
from simlearner3d.utils import utils

import secrets
import random


log = utils.get_logger(__name__)



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
    


    
    