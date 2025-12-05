import torch
from pytorch_lightning import LightningModule
from torch import nn


from simlearner3d.models.modules.resnet_fpn import ResNetFPN_8_1, ResNetFPN_16_4, ResNet34Encoder_FPNDecoder, ResNet34Encoder_FPNDecoder_Inference
from simlearner3d.models.modules.msaff import MSNet
from simlearner3d.models.modules.unet import UNet
from simlearner3d.models.modules.unetgatedattention import UNetGatedAttention
from simlearner3d.models.modules.decision_net import DecisionNetwork

import torch.nn.functional as F
from simlearner3d.utils import utils
from simlearner3d.utils.utils import coords_grid

log = utils.get_logger(__name__)


from simlearner3d.models.generic_model_n_uplet import Model
from simlearner3d.models.criterion.masked_triplet_loss import NPairLoss, NMaskedPairLoss

import tifffile as tf
import numpy as np

def read_sample_images ():
    left_images =["/home/MAChebbi/Documents/Rapports/l_30.tif","/home/MAChebbi/Documents/Rapports/l_100.tif"]
    right_images = ["/home/MAChebbi/Documents/Rapports/r_30.tif","/home/MAChebbi/Documents/Rapports/r_100.tif"]
    disparity_image=["/home/MAChebbi/Documents/Rapports/d_30.tif","/home/MAChebbi/Documents/Rapports/d_100.tif"]
    masq_nocc_image = ["/home/MAChebbi/Documents/Rapports/m_30.tif","/home/MAChebbi/Documents/Rapports/m_100.tif"]

    l1 = tf.imread(left_images[0])
    l2 = tf.imread(left_images[1])


    lT1 = torch.from_numpy(l1).unsqueeze(0)
    lT2 = torch.from_numpy(l2).unsqueeze(0)



    left = torch.cat([lT1,lT2], dim=0)



    r1 = tf.imread(right_images[0])
    r2 = tf.imread(right_images[1])


    rT1 = torch.from_numpy(r1).unsqueeze(0)
    rT20 = torch.from_numpy(r2).unsqueeze(0)
    
    if rT1.shape != rT20.shape:
        rT2= torch.zeros_like(rT1)
        rT2[:,:rT20.shape[-2], :rT20.shape[-1]] = rT20
    else:
        rT2=rT20

    right = torch.cat([rT1,rT2], dim=0).unsqueeze(1)


    d1 = tf.imread(disparity_image[0])
    d2 = tf.imread(disparity_image[1])


    dT1 = torch.from_numpy(d1).unsqueeze(0)
    dT2 = torch.from_numpy(d2).unsqueeze(0)


    disparity = torch.cat([dT1,dT2], dim=0)


    m1 = tf.imread(masq_nocc_image[0])
    m2 = tf.imread(masq_nocc_image[1])


    mT1 = torch.from_numpy(m1).unsqueeze(0)
    mT2 = torch.from_numpy(m2).unsqueeze(0)


    masq_nocc = torch.cat([mT1,mT2], dim=0)

    return left, right, disparity, masq_nocc

    


def training_step_n_uplets (feature, offset=0.5, NODATA=-9999):

    left,right, disp, masq_occ = read_sample_images()
    device='cpu'

    masq_defined= disp != NODATA

    print("masq defined ", masq_defined.shape)

    print(left.shape, right.shape, disp.shape, masq_occ.shape)

    images= torch.cat((left,right),dim=0)
    print("images ", images.shape)


    images = (2 * (images / 255.0) - 1.0).contiguous()

    f_maps = feature(images)


    fmap1, fmap2 = f_maps[:left.shape[0]], f_maps[left.shape[0]:]
    #[fmap1, fmap2] = feature([left,right])

    print("feature maps shapes ", fmap1.shape, fmap2.shape )
    pass
    # normalize feature maps 
    #fmap1 = torch.norm(fmap1,p=2.0, dim=1, keepdim=True)
    #fmap2 = torch.norm(fmap2, p=2.0, dim=1, keepdim=True)
    fmap1 = F.normalize(fmap1,p=2.0, dim=1)
    fmap2 = F.normalize(fmap2,p=2.0, dim=1)

    B,D,H1,W1  = fmap1.shape
    _,_,_, W2  = fmap2.shape

    coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 
    #coords = coords.permute(0,2,3,1) # B,H1,W1,2

    print( coords.shape)

    xgrid0, ygrid = coords.split([1,1], dim=1)

    xgrid = xgrid0 + disp
    # noramlize between [-1,1]

    xgrido = 2*xgrid/(W2-1) - 1
    ygrido = 2*ygrid/(H1-1) - 1

    grid = torch.cat([xgrido,ygrido], dim=1).permute(0,2,3,1) # B,H1,W1, 2
    # compute features at coordinates
    f_map2_pos = F.grid_sample(fmap2, grid, align_corners=True)

    f_map2_pos = F.normalize(f_map2_pos,p=2.0, dim=1)

    # compute all pairs cosine similarities 
    # B,H,W1,W2
    all_corr = Model.corr(fmap1,fmap2)

    print( "all corr ", all_corr.shape, torch.min(all_corr), torch.max(all_corr))

    # get all possible negatives 
    # B,1,H,W1
    corr_matching =  torch.sum( fmap1 * f_map2_pos, dim=1) 

    print( "corr matching ", corr_matching.shape, torch.max(corr_matching), torch.min(corr_matching))

    # select non matching cosine similarity values B, H1, W1
    x_lower_bound =  torch.round(xgrid).long().permute(0,2,3,1)

    print( "x lower bound shape ", x_lower_bound.shape) 

    mask_non_matching = torch.ones_like(all_corr).bool()


    masq_defined = torch.logical_and( masq_defined.permute(0,2,3,1), 
                                     x_lower_bound.ge(0) & x_lower_bound.le(W2-1))

    x_lower_bound[torch.logical_not(masq_defined)] = 0

    print("mask lower bound ", torch.min(x_lower_bound))

    mask_non_matching.scatter_(3,x_lower_bound,0)

    print( "****** ", mask_non_matching[0,348,297,288],
          " ",mask_non_matching[0,348,297,289],
          " ",mask_non_matching[0,348,297,290],
          " ", x_lower_bound[0,348,297,0])
    # mask[i,j,k,xlower[i,j,k,l]] = 1
    corr_matching = corr_matching.unsqueeze(-1).repeat(1,1,1,W2)

    print( "****** ", corr_matching[0,348,297,:])
    print( corr_matching.shape)

    training_loss = NMaskedPairLoss()(corr_matching,all_corr,mask_non_matching)
    print("training loss shape ", training_loss.shape)

    training_loss= training_loss.sum().div( 
        torch.logical_and(mask_non_matching, masq_defined).count_nonzero() 
                                           + 1e-12)
    print( training_loss)



def generate_examples_for_test(left_image):
    left=tf.imread(left_image)[0,:,:]
    H,W = left.shape
    # warp image
    leftT= torch.from_numpy(left).view(1,1,H,W).float()



    coords = coords_grid(1,H,W,'cpu') # B,2,H1,W1 *

    xgrid, ygrid = coords.split([1,1], dim=1)
    # generate constant displacement 
    # dispaity = a x 

    D = 0.85  *  xgrid 
    
    # between -1 and 1.0

    D_sec = 0.15 * xgrid /0.85

    # save applied displacement 

    tf.imwrite("applied_disparity_delta.tif", D_sec.detach().squeeze().numpy())

    D= 2*D/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid_p = torch.cat([D,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2


    x_warped = F.grid_sample(leftT, grid_p,)

    tf.imwrite("warped_left.tif", x_warped.detach().squeeze().numpy())





def generate_examples_another( left_image):
    left=tf.imread(left_image)[0,:,:]
    H,W = left.shape


    coords = coords_grid(1,H,W,'cpu') # B,2,H1,W1 *
    xgrid, ygrid = coords.split([1,1], dim=1)
    # generate constant displacement 
    # dispaity = a x 
    D = 0.85  *  xgrid 

    from scipy.interpolate import RectBivariateSpline

    D= D.detach().squeeze().numpy()
    ygrid= ygrid.detach().squeeze().numpy()

    spline = RectBivariateSpline(np.arange(H), np.arange(W), left)

    interpolated = spline(ygrid, D, grid=False)

    tf.imwrite("warped_left_scipy.tif", interpolated)







STEPS= [1.0,0.5,0.25,0.125]
PROBAS=[0.3,0.3,0.3,0.1]
false1=1
false2=4

def test_step_subpixel(feature,NODATA=-9999.0):
    with torch.no_grad():
        feature.eval()
        x0,x1, dispnoc0, Mask0 = read_sample_images()
        x11=x1

        x0 = (2 * (x0 / 255.0) - 1.0).contiguous()
        x1 = (2 * (x1 / 255.0) - 1.0).contiguous()

        device='cpu'     
        MaskDef=(dispnoc0!=NODATA)
        #device='cuda' if x0.is_cuda else 'cpu'
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        FeatsL=feature(x0) 
        FeatsR=feature(x1)

        B,D,H1,W1  = FeatsL.shape
        _,_,_, W2  = FeatsR.shape

        aStep= 0.5 #np.random.choice(STEPS,p=PROBAS)

        Offset_neg=((false1 - false2) * torch.rand(dispnoc0.size(),device=device) + false2) * aStep
        RandSens=torch.rand(dispnoc0.size(),device=device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_neg=Offset_neg*RandSens

        coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 
        xgrid0, ygrid = coords.split([1,1], dim=1)
        xgrid = xgrid0 + dispnoc0
        xgrid_pos = torch.clamp(torch.round(xgrid/aStep)*aStep,0,W2-1)
        xgrid_neg = torch.clamp(torch.round((xgrid-Offset_neg)/aStep)*aStep,0,W2-1)

        # display matching and non matching images 


        xgrid_pos = 2*xgrid_pos/(W2-1) - 1
        xgrid_neg = 2*xgrid_neg/(W2-1) - 1
        ygrid = 2*ygrid/(H1-1) - 1
        grid_p = torch.cat([xgrid_pos,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        grid_n = torch.cat([xgrid_neg,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2


        matching_image = F.grid_sample(x11.float(),grid_p, align_corners=True)

        non_matching_image = F.grid_sample(x11.float(), grid_n, align_corners=True)



        tf.imwrite ("matching_image_{}.tif".format(str(aStep)), 
                    matching_image.squeeze().detach().numpy()
                    )

        tf.imwrite ("non_matching_image_{}.tif".format(str(aStep)), 
                    non_matching_image.squeeze().detach().numpy()
                    )
        


if __name__=="__main__":
    # read model
    model = UNet()
    loss = test_step_subpixel(model)


    #generate_examples_another("/home/MAChebbi/Documents/Rapports/l_100.tif")
    



