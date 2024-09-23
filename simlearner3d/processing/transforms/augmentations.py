import numpy as np
from simlearner3d.utils.utils import Data
from simlearner3d.utils import utils

import secrets
import random


log = utils.get_logger(__name__)



class ClipAndComputeUsingPatchSize:
    def __init__(self,
                 tile_height,
                 patch_size
                 ):
        self.tile_height=tile_height
        self.patch_size=patch_size
    
    def __call__(self, data: Data):
        data = self.clip_sample(data)
        return data
    
    def clip_sample(self, data: Data):
        """ use random clipping taken disparity constraints into account"""
        _notocc=data._disp*data._masq
        _mean_disparity=int(np.mean(_notocc[_notocc!=0.0]))
        x_upl=_mean_disparity+secrets.randbelow(self.tile_height-self.patch_size-_mean_disparity)
        y_upl=secrets.randbelow(self.tile_height-self.patch_size)
        # return a new data object 
        return Data(
            _left=data._left[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size],
            _right=data._right[y_upl:y_upl+self.patch_size,:],
            _disp=data._disp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size],
            _masq=data._masq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size],
            _xupl=x_upl,
        )



class DownScaleImage:
    def __init__(self,scales):
        self.scales=scales
    
    def __call__(self, data: Data):
        scale_factor=random.choice(self.scales)
        data=self.apply_scale(data,scale_factor)
        return data
    
    def apply_scale(self,data: Data,scale_factor):
        import torch.nn.functional as F
        _left_down=F.interpolate(data._left.unsqueeze(0).unsqueeze(0),
                        scale_factor=scale_factor,
                        mode='bilinear')
        _right_down=F.interpolate(data._right.unsqueeze(0).unsqueeze(0),
                        scale_factor=scale_factor,
                        mode='bilinear')
        
        return Data(
            _left=F.interpolate(_left_down, 
                                scale_factor=1/scale_factor,
                                  mode='bilinear').squeeze(),
            _right=F.interpolate(_right_down, 
                                scale_factor=1/scale_factor,
                                  mode='bilinear').squeeze(),
            _disp=data._disp,
            _masq=data._masq,                       
        )



class VerticalFlip:
    def __init__(self,p=0.5):
        self.p=p
    def __call__(self, data: Data):
        data=self.vflip(data)
        return data
    
    def vflip(self, data: Data):
        import torchvision.transforms as T
        if random.random()< self.p:
            return data
        else:
            return Data(
                _left=T.functional.vflip(data._left),
                _right=T.functional.vflip(data._right),
                _disp=T.functional.vflip(data._disp),
                _masq=T.functional.vflip(data._masq),

            )

