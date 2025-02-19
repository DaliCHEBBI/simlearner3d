import os
import os.path as osp
import sys

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import torchvision.transforms as transforms

from pytorch_lightning import (
    LightningModule,
)

import imageio
import tifffile
import torch.nn.functional as F
import numpy as np
import time
import copy

from simlearner3d.models.generic_regression_model import ModelReg

sys.path.append(osp.dirname(osp.dirname(__file__)))

from simlearner3d.utils import utils  # noqa

log = utils.get_logger(__name__)


NEURAL_NET_ARCHITECTURE_CONFIG_GROUP = "neural_net"


@utils.eval_time
def predict(config: DictConfig) -> str:
    """
    Inference pipeline using a pair of tiles of size (1024 x 1024)

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        str: path to ouptut .tif disparity.

    """
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)


    # Those are the 2 needed inputs, in addition to the hydra config.
    assert os.path.exists(config.predict.ckpt_path)
    assert os.path.exists(config.predict.left_image)
    assert os.path.exists(config.predict.right_image)


    # Do not require gradient for faster predictions
    torch.set_grad_enabled(False)
    #model = ModelReg.load_from_checkpoint(config.predict.ckpt_path)
    if model.load_pretrained:
        model.load_trained_assets(config.predict.ckpt_path)
    else:
        kwargs_to_override = copy.deepcopy(model.hparams)
        kwargs_to_override.pop(
            NEURAL_NET_ARCHITECTURE_CONFIG_GROUP, None
        )  # removes that key if it's there
        model = ModelReg.load_from_checkpoint(config.predict.ckpt_path, **kwargs_to_override)

    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    def test(imgL,imgR):
        imgL = imgL.to(device)
        imgR = imgR.to(device)     

        with torch.no_grad():
            disp = model.regressor(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp
    # load images,  check if 3 channels, standardize  

    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])    

    imgL_o=tifffile.imread(config.predict.left_image)
    imgR_o=tifffile.imread(config.predict.right_image)
    if imgL_o.ndim<3:
        imgL_o=np.expand_dims(imgL_o,-1)
        imgL_o=np.tile(imgL_o,(1,1,3)).astype(np.uint8)
    else:
        imgL_o=imgL_o[...,0:3]
        imgL_o=imgL_o.astype(np.uint8)
    if imgR_o.ndim<3:
        imgR_o=np.expand_dims(imgR_o,-1)
        imgR_o=np.tile(imgR_o,(1,1,3)).astype(np.uint8)
    else:
        imgR_o=imgR_o[...,0:3]
        imgR_o=imgR_o.astype(np.uint8)


    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o) 
    print("shape of tensor after transform ",imgL.shape) 

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = test(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))

    if top_pad !=0 and right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    else:       
        img = pred_disp

    os.makedirs(config.predict.output_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(config.predict.output_dir,"disparity_real.tif"), img)

    img = (img*config.predict.disp_scale).astype('uint16')
    
    imageio.imwrite(os.path.join(config.predict.output_dir,"disparity.tif"), img)