from collections import OrderedDict
from random import randint
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from pytorch_lightning import LightningModule
import gc
from models.model import DecisionNetwork5D, DecisionNetwork,DecisionNetworkOnCube,DecisionNetworkOnNCubes
from models.Matcher3D import Matcher3D
import secrets
from models.CosSchedulerWarmUp import CosineAnnealingWarmupRestarts

def balanced_binary_cross_entropy(pred, gt,nogt, pos_w=2.0, neg_w=1.0):
    masked_nogt=nogt.sub(gt)
    # flatten vectors
    pred = pred.view(-1)
    gt = gt.view(-1)
    masked_nogt=masked_nogt.view(-1)
    # select postive/nevative samples
    pos_ind = gt.nonzero().squeeze(-1)
    neg_ind = masked_nogt.nonzero().squeeze(-1)

    # compute weighted loss
    pos_loss = pos_w*F.binary_cross_entropy(pred[pos_ind], gt[pos_ind], reduction='none')
    neg_loss = neg_w*F.binary_cross_entropy(pred[neg_ind], masked_nogt[neg_ind], reduction='none')
    g_loss=pos_loss + neg_loss
    g_loss=g_loss.div(nogt.count_nonzero()+1e-12)
    return g_loss

def mse(coords, coords_gt, prob_gt):

    # flatten vectors
    coords = coords.view(-1, 2)
    coords_gt = coords_gt.view(-1, 2)
    prob_gt = prob_gt.view(-1)

    # select positive samples
    pos_ind = prob_gt.nonzero().squeeze(-1)
    pos_coords = coords[pos_ind, :]
    pos_coords_gt = coords_gt[pos_ind, :]

    return F.mse_loss(pos_coords, pos_coords_gt)


def generate_pointcloud(CUBE, ply_file):
    """
    Generate a colored ply from  the dense cube
    """
    points = []
    for zz in range(CUBE.size()[0]):
        for yy in range(CUBE.size()[1]):
            for xx in range(CUBE.size()[2]):
                val=CUBE[zz,yy,xx]
                points.append("%f %f %f %f %f %f 0\n"%(xx,yy,zz,val,val,val))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property float red
property float green
property float blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

class UNet(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = nn.Sequential(
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features*2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        )
                     )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec_inter = torch.cat((dec1, enc1), dim=1)
        output_feature = self.decoder1(dec_inter)
        return output_feature
    def _block(self, in_channels, features):
        return nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True)
                       )

class UNetInference(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNetInference, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = nn.Sequential(
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features*2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        )
                     )

    def forward(self, x):
        if x.size()[-2] % 16 != 0:
            times = x.size()[-2]//16   
            top_pad = (times+1)*16 - x.size()[-2]
        else:
            top_pad = 0
        if x.size()[-1] % 16 != 0:
            times = x.size()[-1]//16
            right_pad = (times+1)*16-x.size()[-1] 
        else:
            right_pad = 0    

        x = F.pad(x,(0,right_pad, top_pad,0))

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec_inter = torch.cat((dec1, enc1), dim=1)
        output_feature = self.decoder1(dec_inter)

        if top_pad !=0 and right_pad != 0:
            out = output_feature[:,:,top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            out = output_feature[:,:,:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            out = output_feature[:,:,top_pad:,:]
        else:
            out = output_feature
        return out

    def _block(self, in_channels, features):
        return nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True)
                       )

class UNetWithDecisionNetwork(nn.Module):
     def __init__(self,infeats,outfeats):
         super(UNetWithDecisionNetwork,self).__init__()
         self.in_features=infeats
         self.out_features=outfeats
         self.feature=UNetInference(init_features=self.in_features)
         self.decisionNet=DecisionNetwork(self.out_features)
     def forward(self,x):
         f_all=self.feature(x)
         #print(f_all.shape)
         # shape 2,64,w,h
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return torch.sigmoid(out)

class UNETWithDecisionNetwork_LM5D(LightningModule):
    def __init__(self,Inplanes,true1=1,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_LM5D, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.EPSILON=0.08
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.residual_criterion=nn.MSELoss(reduction='none') # add a weighting parameter around 0.1 
        self.inplanes = Inplanes
        self.learning_rate=0.0005
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork5D(2*64)
    def training_step(self,batch, batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0 # shape (N,1,h,w)
        #print( "Mask shape and content ",Mask0.shape)
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0 # shape (N,1,h,w)
        # Forward 
        N=x0.size(0)
        Feats=self.feature(torch.cat((x0,x1),dim=0))
        FeatsL=Feats[0:N]    # shape (N,64,h,w)
        FeatsR=Feats[N:2*N]  # shape (N,64,h,w)
        # Construire les nappes englobantes
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        #print("Index values  ",Index_X.shape)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        # Index_X of shape (N,1,H,W) give x coordinate 
        #print("Index values  ",Index_X.shape)
        # Repete FeatsL and FeatsR 2*false2 times 
        FeatsL=FeatsL.unsqueeze(1).repeat_interleave(2*self.false2,1) # N, 2*false2, 64,h,w
        #print("Features LEFT Repearted ",FeatsL.shape)
        torch._assert(torch.equal(FeatsL[0,0,:,:,:],(FeatsL[0,15,:,:,:])),"issue repeate interleave")
        # Generate positive sample tensor of shape equal to reference anchor tensor
        FeatsQuery=torch.empty(FeatsL.shape).to(self.device)
        MaskQuery=torch.empty((N,2*self.false2,1,dispnoc0.size(-2),dispnoc0.size(-1))).to(self.device)
        for i in np.arange(-self.false2,self.false2):
            Offset=Index_X-dispnoc0+i
            MaskOffNegative=((Offset>=0)*Mask0*(Offset<dispnoc0.size()[-1])).float().to(self.device)
            Offset=(Offset*MaskOffNegative).to(torch.int64)
            Offset=Offset.repeat_interleave(FeatsR.size()[1],1)
            FeatRSample=torch.gather(FeatsR,-1,Offset)
            #print("RIGHT SAMPLE SHAPE ",FeatRSample.shape)
            # Fill 5 Dimensional Tensor
            MaskQuery[:,i+self.false2,:,:,:]=MaskOffNegative
            FeatsQuery[:,i+self.false2,:,:,:]=FeatRSample
        # You have 2 5 Dimensional tensors to pass into the decison DecisionNetwork
        OutSimil=self.decisionNet(torch.cat((FeatsL,FeatsQuery),dim=2)).squeeze()# Dimension (N, 2f, 1 ,H, W) 
        OutSimil=OutSimil*MaskQuery.float().squeeze()
        # MASK_DISPNOC DEFINTION 
        ref_pos=OutSimil[:,self.false2,:,:] # Centerd on Gt disparities
        # search for the most annpying elements in the structure of the cube 
        NORM_DIFF=torch.abs(OutSimil.sigmoid()-ref_pos.sigmoid().unsqueeze(1).repeat_interleave(2*self.false2,1))
        ref_neg=torch.amin(torch.where(NORM_DIFF-self.EPSILON>0,OutSimil,torch.ones(NORM_DIFF.shape,device=self.device)),1)
        ref_neg=ref_neg*Mask0.squeeze()
        # INDICE DES PLUS FAIBLES MAIS SUPERIEURS A EPSILON ==> CES CEUX QUI VONT DEFINIR LESEXEMPLES NEGATIFS 
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones((N,x0.size(-2),x0.size(-1))), torch.zeros((N,x0.size(-2),x0.size(-1)))), dim=0)
        training_loss=self.criterion(sample+1e-20, target.to(device=self.device, dtype=torch.float))*torch.cat((Mask0,Mask0),0).squeeze()
        training_loss=training_loss.sum().div(2*Mask0.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch, batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0 # shape (N,1,h,w)
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0 # shape (N,1,h,w)
        # Forward 
        N=x0.size(0)
        Feats=self.feature(torch.cat((x0,x1),dim=0))
        FeatsL=Feats[0:N]    # shape (N,64,h,w)
        FeatsR=Feats[N:2*N]  # shape (N,64,h,w)
        # Construire les nappes englobantes
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        # Index_X of shape (N,1,H,W) give x coordinate 
        # Repete FeatsL and FeatsR 2*false2 times 
        FeatsL=FeatsL.unsqueeze(1).repeat_interleave(2*self.false2,1) # N, 2*false2, 64,h,w
        # Generate positive sample tensor of shape equal to reference anchor tensor
        FeatsQuery=torch.empty(FeatsL.shape).to(self.device)
        MaskQuery=torch.empty((N,2*self.false2,1,dispnoc0.size(-2),dispnoc0.size(-1))).to(self.device)
        for i in np.arange(-self.false2,self.false2):
            Offset=Index_X-dispnoc0+i
            MaskOffNegative=((Offset>=0)*Mask0*(Offset<dispnoc0.size()[-1])).float().to(self.device)
            Offset=(Offset*MaskOffNegative).to(torch.int64)
            Offset=Offset.repeat_interleave(FeatsR.size()[1],1)
            FeatRSample=torch.gather(FeatsR,-1,Offset)
            # Fill 5 Dimensional Tensor
            MaskQuery[:,i+self.false2,:,:,:]=MaskOffNegative
            FeatsQuery[:,i+self.false2,:,:,:]=FeatRSample
        # You have 2 5 Dimensional tensors to pass into the decison DecisionNetwork
        OutSimil=self.decisionNet(torch.cat((FeatsL,FeatsQuery),dim=2)).squeeze() # Dimension (N, 2f, 1 ,H, W) 
        OutSimil=OutSimil*MaskQuery.float().squeeze()
        # MASK_DISPNOC DEFINTION 
        ref_pos=OutSimil[:,self.false2,:,:] # Centerd on Gt disparities
        # search for the most annpying elements in the structure of the cude 
        NORM_DIFF=torch.abs(OutSimil.sigmoid()-ref_pos.sigmoid().unsqueeze(1).repeat_interleave(2*self.false2,1))
        ref_neg=torch.amin(torch.where(NORM_DIFF-self.EPSILON>0,OutSimil,torch.ones(NORM_DIFF.shape,device=self.device)),1)
        # INDICE DES PLUS FAIBLES MAIS SUPERIEURS A EPSILON ==> CES CEUX QUI VONT DEFINIR LESEXEMPLES NEGATIFS
        ref_neg=ref_neg*Mask0.squeeze()
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones((N,x0.size(-2),x0.size(-1))), torch.zeros((N,x0.size(-2),x0.size(-1)))), dim=0)
        val_loss=self.criterion(sample+1e-20, target.to(device=self.device, dtype=torch.float))*torch.cat((Mask0,Mask0),0).squeeze()
        val_loss=val_loss.sum().div(2*Mask0.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",val_loss, on_epoch=True)
        """if (MaskGlob.count_nonzero().item()):
             Tplus=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_plus).unsqueeze(1),MaskGlob.bool())
             Tmoins=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_minus).unsqueeze(1),MaskGlob.bool())
             if len(torch.nonzero(Tplus.sub(Tplus.mean())))>1:# and torch.not_equal(Tmoins.sub(Tmoins.mean()),0)
                  #print(Tmoins.nelement(), Tplus.nelement())
                  self.logger.experiment.add_histogram('distribution positive',Tplus,global_step=self.nbsteps)
                  self.logger.experiment.add_histogram('distribution negative',Tmoins,global_step=self.nbsteps)
        self.nbsteps=self.nbsteps+1"""
        return val_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)
        # ReduceOnPlateau scheduler 
        """reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        sch_val = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': "val_loss",
            'frequency': 1,
        }"""
        return [optimizer],[scheduler]


class UNETWithDecisionsNetwork_LM_MulScaleBCE(LightningModule):
    def __init__(self,Inplanes,true1=1,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionsNetwork_LM_MulScaleBCE, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.0005
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNetMS(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0
        # Forward
        FeatsL,FeatsL2,FeatsL4,FeatsL8=self.feature(x0) # MulScale Features
        print("Features ==> ",FeatsL.size(),FeatsL2.size(),FeatsL4.size(),FeatsL8.size())
        FeatsR,FeatsR2,FeatsR4,FeatsR8=self.feature(x1) # MulScale Features
        Offset_pos=(-2*self.true1) * torch.rand(dispnoc0.size(),device=self.device) + self.true1 #[-true1,true1]
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=self.device) + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device=self.device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).to(self.device)
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        Offp=Index_X-D_pos.round()
        Offn=Index_X-D_neg.round()
        # Clean Indexes so there is no overhead
        MaskOffPositive=((Offp>=0)*(Offp<dispnoc0.size()[-1])).float().to(self.device)
        MaskOffNegative=((Offn>=0)*(Offn<dispnoc0.size()[-1])).float().to(self.device)
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive)#.to(torch.int64)
        Offn=(Offn*MaskOffNegative)#.to(torch.int64)
        # Rescale Masq Tensors to /2 /4 and /8 
        Offp2=F.interpolate(Offp,FeatsR2.size()[2:],mode='nearest')
        Offp4=F.interpolate(Offp,FeatsR4.size()[2:],mode='nearest')
        Offp8=F.interpolate(Offp,FeatsR8.size()[2:],mode='nearest')

        Offn2=F.interpolate(Offn,FeatsR2.size()[2:],mode='nearest')
        Offn4=F.interpolate(Offn,FeatsR4.size()[2:],mode='nearest')
        Offn8=F.interpolate(Offn,FeatsR8.size()[2:],mode='nearest')
        Offp=Offp.to(torch.int64)
        Offp2=Offp2.to(torch.int64)
        Offp4=Offp4.to(torch.int64)
        Offp8=Offp8.to(torch.int64)

        Offn=Offn.to(torch.int64)
        Offn2=Offn2.to(torch.int64)
        Offn4=Offn4.to(torch.int64)
        Offn8=Offn8.to(torch.int64)
        # Need to repeat interleave training_loss
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        Offp2=Offp2.repeat_interleave(FeatsR2.size()[1],1)
        Offn2=Offn2.repeat_interleave(FeatsR2.size()[1],1)
        Offp4=Offp4.repeat_interleave(FeatsR4.size()[1],1)
        Offn4=Offn4.repeat_interleave(FeatsR4.size()[1],1)
        Offp8=Offp8.repeat_interleave(FeatsR8.size()[1],1)
        Offn8=Offn8.repeat_interleave(FeatsR8.size()[1],1)
        # Get Examples positive and negative
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        FeatsR_plus2=torch.gather(FeatsR2,-1,Offp2)
        FeatsR_plus4=torch.gather(FeatsR4,-1,Offp4)
        FeatsR_plus8=torch.gather(FeatsR8,-1,Offp8)
        # Test gather operator
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        FeatsR_minus2=torch.gather(FeatsR2,-1,Offn2)
        FeatsR_minus4=torch.gather(FeatsR4,-1,Offn4)
        FeatsR_minus8=torch.gather(FeatsR8,-1,Offn8)
        # Mask Global = Mask des batiments + Mask des offsets bien definis
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        MaskGlob2=F.interpolate(MaskGlob,FeatsR2.size()[2:],mode='nearest')
        MaskGlob4=F.interpolate(MaskGlob,FeatsR4.size()[2:],mode='nearest')
        MaskGlob8=F.interpolate(MaskGlob,FeatsR8.size()[2:],mode='nearest')
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        print("decision Net 1  ",FeatsR_plus.size(),FeatsL.size())
        # Scale 2
        ref_pos2=self.decisionNet(torch.cat((FeatsL2,FeatsR_plus2),1))
        print("decision Net 2 ",FeatsR_plus2.size(),FeatsL2.size())
        ref_neg2=self.decisionNet(torch.cat((FeatsL2,FeatsR_minus2),1))
        # Scale 4
        ref_pos4=self.decisionNet(torch.cat((FeatsL4,FeatsR_plus4),1))
        ref_neg4=self.decisionNet(torch.cat((FeatsL4,FeatsR_minus4),1))
        # Scale 8
        ref_pos8=self.decisionNet(torch.cat((FeatsL8,FeatsR_plus8),1))
        ref_neg8=self.decisionNet(torch.cat((FeatsL8,FeatsR_minus8),1))

        sample = torch.cat((ref_pos, ref_neg), dim=0)
        sample2 = torch.cat((ref_pos2, ref_neg2), dim=0)
        sample4 = torch.cat((ref_pos4, ref_neg4), dim=0)
        sample8 = torch.cat((ref_pos8, ref_neg8), dim=0)

        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        target2 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2)), dim=0)
        target4 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4)), dim=0)
        target8 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8)), dim=0)

        training_loss0=self.criterion(sample, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
        training_loss2=self.criterion(sample2, target2.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob2,MaskGlob2),0)
        training_loss4=self.criterion(sample4, target4.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob4,MaskGlob4),0)
        training_loss8=self.criterion(sample8, target8.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob8,MaskGlob8),0)
        """if (torch.any(torch.isnan(training_loss))):
            raise Exception("nan values encountered in training loss ")"""
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss0=training_loss0.sum().div(2*MaskGlob.count_nonzero()+1e-20)
        training_loss2=training_loss2.sum().div(2*MaskGlob2.count_nonzero()+1e-20)
        training_loss4=training_loss4.sum().div(2*MaskGlob4.count_nonzero()+1e-20)
        training_loss8=training_loss8.sum().div(2*MaskGlob8.count_nonzero()+1e-20)
        gc.collect()
        self.log("training_loss",training_loss0+0.5*training_loss2+0.25*training_loss4+0.125*training_loss8, on_epoch=True)
        return training_loss0+0.5*training_loss2+0.25*training_loss4+0.125*training_loss8

    def validation_loss(self,batch,batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0
        # Forward
        FeatsL,FeatsL2,FeatsL4,FeatsL8=self.feature(x0) # MulScale Features
        FeatsR,FeatsR2,FeatsR4,FeatsR8=self.feature(x1) # MulScale Features
        Offset_pos=(-2*self.true1) * torch.rand(dispnoc0.size(),device=self.device) + self.true1 #[-true1,true1]
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=self.device) + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device=self.device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).to(self.device)
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        Offp=Index_X-D_pos.round()
        Offn=Index_X-D_neg.round()
        # Clean Indexes so there is no overhead
        MaskOffPositive=((Offp>=0)*(Offp<dispnoc0.size()[-1])).float().to(self.device)
        MaskOffNegative=((Offn>=0)*(Offn<dispnoc0.size()[-1])).float().to(self.device)
        Offp=(Offp*MaskOffPositive)#.to(torch.int64)
        Offn=(Offn*MaskOffNegative)#.to(torch.int64)
        # Rescale Masq Tensors to /2 /4 and /8 
        Offp2=F.interpolate(Offp,FeatsR2.size()[2:],mode='nearest')
        Offp4=F.interpolate(Offp,FeatsR4.size()[2:],mode='nearest')
        Offp8=F.interpolate(Offp,FeatsR8.size()[2:],mode='nearest')
        Offn2=F.interpolate(Offn,FeatsR2.size()[2:],mode='nearest')
        Offn4=F.interpolate(Offn,FeatsR4.size()[2:],mode='nearest')
        Offn8=F.interpolate(Offn,FeatsR8.size()[2:],mode='nearest')
        Offp=Offp.to(torch.int64)
        Offp2=Offp2.to(torch.int64)
        Offp4=Offp4.to(torch.int64)
        Offp8=Offp8.to(torch.int64)

        Offn=Offn.to(torch.int64)
        Offn2=Offn2.to(torch.int64)
        Offn4=Offn4.to(torch.int64)
        Offn8=Offn8.to(torch.int64)
        # Need to repeat interleave training_loss
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        Offp2=Offp2.repeat_interleave(FeatsR2.size()[1],1)
        Offn2=Offn2.repeat_interleave(FeatsR2.size()[1],1)
        Offp4=Offp4.repeat_interleave(FeatsR4.size()[1],1)
        Offn4=Offn4.repeat_interleave(FeatsR4.size()[1],1)
        Offp8=Offp8.repeat_interleave(FeatsR8.size()[1],1)
        Offn8=Offn8.repeat_interleave(FeatsR8.size()[1],1)
        # Get Examples positive and negative
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        FeatsR_plus2=torch.gather(FeatsR2,-1,Offp2)
        FeatsR_plus4=torch.gather(FeatsR4,-1,Offp4)
        FeatsR_plus8=torch.gather(FeatsR8,-1,Offp8)
        # Test gather operator
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        FeatsR_minus2=torch.gather(FeatsR2,-1,Offn2)
        FeatsR_minus4=torch.gather(FeatsR4,-1,Offn4)
        FeatsR_minus8=torch.gather(FeatsR8,-1,Offn8)
        # Mask Global = Mask des batiments + Mask des offsets bien definis
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        MaskGlob2=F.interpolate(MaskGlob,FeatsR2.size()[2:],mode='nearest')
        MaskGlob4=F.interpolate(MaskGlob,FeatsR4.size()[2:],mode='nearest')
        MaskGlob8=F.interpolate(MaskGlob,FeatsR8.size()[2:],mode='nearest')
        print("Model masks created  ")
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        # Scale 2
        ref_pos2=self.decisionNet(torch.cat((FeatsL2,FeatsR_plus2),1))
        ref_neg2=self.decisionNet(torch.cat((FeatsL2,FeatsR_minus2),1))
        # Scale 4
        ref_pos4=self.decisionNet(torch.cat((FeatsL4,FeatsR_plus4),1))
        ref_neg4=self.decisionNet(torch.cat((FeatsL4,FeatsR_minus4),1))
        # Scale 8
        ref_pos8=self.decisionNet(torch.cat((FeatsL8,FeatsR_plus8),1))
        ref_neg8=self.decisionNet(torch.cat((FeatsL8,FeatsR_minus8),1))

        sample = torch.cat((ref_pos, ref_neg), dim=0)
        sample2 = torch.cat((ref_pos2, ref_neg2), dim=0)
        sample4 = torch.cat((ref_pos4, ref_neg4), dim=0)
        sample8 = torch.cat((ref_pos8, ref_neg8), dim=0)

        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        target2 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2)), dim=0)
        target4 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4)), dim=0)
        target8 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8)), dim=0)

        validation_loss0=self.criterion(sample, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
        validation_loss2=self.criterion(sample2, target2.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob2,MaskGlob2),0)
        validation_loss4=self.criterion(sample4, target4.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob4,MaskGlob4),0)
        validation_loss8=self.criterion(sample8, target8.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob8,MaskGlob8),0)
        validation_loss0=validation_loss0.sum().div(2*MaskGlob.count_nonzero()+1e-20)
        validation_loss2=validation_loss2.sum().div(2*MaskGlob2.count_nonzero()+1e-20)
        validation_loss4=validation_loss4.sum().div(2*MaskGlob4.count_nonzero()+1e-20)
        validation_loss8=validation_loss8.sum().div(2*MaskGlob8.count_nonzero()+1e-20)
        gc.collect()
        self.log("val_loss",validation_loss0+0.5*validation_loss2+0.25*validation_loss4+0.125*validation_loss8, on_epoch=True)
        return validation_loss0+0.5*validation_loss2+0.25*validation_loss4+0.125*validation_loss8

    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)
        return [optimizer],[scheduler]
        """return {
           "optimizer": optimizer,
           "lr_scheduler": {
              "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=10,min_lr=1e-8),
              "interval": "step",
              "monitor": "training_loss",
              "frequency": 1
         },
        }"""

class UNETWithDecisionNetwork_Dense_LM(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_Dense_LM, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        dispnoc0=dispnoc0.unsqueeze(1)
        Mask0=Mask0.unsqueeze(1).mul(dispnoc0!=0.0)
        dispnoc0=dispnoc0*Mask0.float() # Set Nans to 0.0
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        # ADD OFFSET
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        training_loss=self.criterion(sample+1e-12, target.float().cuda())*torch.cat((MaskGlob,MaskGlob),0)
        #if (torch.any(torch.isnan(training_loss))):
            #raise Exception("nan values encountered in training loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(2*MaskGlob.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        Mask0=Mask0.unsqueeze(1).mul(dispnoc0!=0.0)

        #print("initial shapes ", x0.shape,x1.shape,dispnoc0.shape,Mask0.shape)
        dispnoc0=dispnoc0*Mask0.float() # Set Nans to 0.0
        #print("Disparity shaope ",dispnoc0.shape)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        #print("Index SHAPE ", Index_X.shape)
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        #print("offsets shapes ", Offp.shape, Offn.shape)
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        validation_loss=self.criterion(sample+1e-12, target.float().cuda())*torch.cat((MaskGlob,MaskGlob),0)
        #if (torch.any(torch.isnan(training_loss))):
            #raise Exception("nan values encountered in training loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(2*MaskGlob.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.2},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.4},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate*0.6},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]
        optimizer=torch.optim.AdamW(param_grps,lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,60,70,80,90],gamma=0.7)
        return [optimizer],[scheduler]

class UNETWithDecisionNetwork_Dense_LM_N(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_Dense_LM_N, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=dispnoc0!=0.0  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0).float()
        training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=dispnoc0!=0.0  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0).float()
        validation_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        """EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.2},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.4},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate*0.6},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]"""
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,60,70,80,90],gamma=0.7)
        return [optimizer],[scheduler]
    
class UNETWithDecisionNetwork_Dense_LM_N_2(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_Dense_LM_N_2, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        #print("SAMPLE  ==> ",x0.shape,x1.shape,dispnoc0.shape,MaskDef.shape,Mask0.shape,x_offset.shape)
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',torch.max(OCCLUDED))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_pos=(-2*self.true1) * torch.rand(dispnoc0.size()).cuda() + self.true1 #[-true1,true1]
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, prog_bar=True,logger=True, on_step=True, on_epoch=True,sync_dist=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',OCCLUDED.shape)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_pos=(-2*self.true1) * torch.rand(dispnoc0.size()).cuda() + self.true1 #[-true1,true1]
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        validation_loss=self.criterion(sample+1e-12, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,sync_dist=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.01},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.1},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate*0.5},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]
        #optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate,momentum=0.9)
        optimizer=torch.optim.AdamW(param_grps,lr=self.learning_rate)
        """scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1910,
                                          cycle_mult=1.0,
                                          max_lr=0.01,
                                          min_lr=0.0001,
                                          warmup_steps=200,
                                          gamma=0.95)"""
        """scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = 10, # Maximum number of iterations.
                             eta_min = 1e-4)""" # Minimum learning rate.
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 1, T_mult = 1, eta_min = 1e-4)
        """scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            }"""
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,100,150,200],gamma=0.7)
        return [optimizer],[scheduler]
    
    
class UNETWithDecisionNetwork_Dense_LM_N_SubPix(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_Dense_LM_N_SubPix, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        #print("SAMPLE  ==> ",x0.shape,x1.shape,dispnoc0.shape,MaskDef.shape,Mask0.shape,x_offset.shape)
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',torch.max(OCCLUDED))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device='cuda') + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device='cuda')
        RandSens=(RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        Index_X=torch.arange(0,dispnoc0.size()[-1],device='cuda')
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-dispnoc0  
        Offn=Index_X-(dispnoc0+Offset_neg)
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()
        Index_Y=torch.arange(0,dispnoc0.size()[-2],device='cuda').resize(dispnoc0.size()[-2],1)
        Index_Y=Index_Y.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).repeat(x0.size()[0],1,1) # BS,H,W
        
        # Later for 2D add Y_offset and subsequent masks 
        AllGeoP=torch.stack((Offp.squeeze(1).div(FeatsR.size()[-1]//2),Index_Y.div(FeatsR.size()[-2]//2)),-1).sub(1.0)
        AllGeoN=torch.stack((Offn.squeeze(1).div(FeatsR.size()[-1]//2),Index_Y.div(FeatsR.size()[-2]//2)),-1).sub(1.0)
        ###########FeatsR_plus=self.InterpolateSubPix2D(FeatsR,Offp.squeeze(1),Index_Y)
        FeatsR_plus=F.grid_sample(FeatsR,AllGeoP,mode='bilinear', align_corners=False)
        # Test gather operator 
        ######FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        ###########FeatsR_minus=self.InterpolateSubPix2D(FeatsR,Offn.squeeze(1),Index_Y)
        FeatsR_minus=F.grid_sample(FeatsR,AllGeoN,mode='bilinear', align_corners=False)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size(),device='cuda')-OCCLUDED.float(), torch.zeros(x0.size(),device='cuda')), dim=0)
        training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, prog_bar=True,logger=True, on_step=True, on_epoch=True,sync_dist=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',OCCLUDED.shape)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device='cuda') + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device='cuda')
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device='cuda')
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos  
        Offn=Index_X-D_neg 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()
        Index_Y=torch.arange(0,dispnoc0.size()[-2],device='cuda').resize(dispnoc0.size()[-2],1)
        Index_Y=Index_Y.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).repeat(x0.size()[0],1,1) # BS,H,W
        # Later for 2D add Y_offset and subsequent masks 
        AllGeoP=torch.stack((Offp.squeeze(1).div(FeatsR.size()[-1]//2),Index_Y.div(FeatsR.size()[-2]//2)),-1).sub(1.0)
        AllGeoN=torch.stack((Offn.squeeze(1).div(FeatsR.size()[-1]//2),Index_Y.div(FeatsR.size()[-2]//2)),-1).sub(1.0)
        ###########FeatsR_plus=self.InterpolateSubPix2D(FeatsR,Offp.squeeze(1),Index_Y)
        FeatsR_plus=F.grid_sample(FeatsR,AllGeoP,mode='bilinear', align_corners=False)
        # Test gather operator 
        ######FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        ###########FeatsR_minus=self.InterpolateSubPix2D(FeatsR,Offn.squeeze(1),Index_Y)
        FeatsR_minus=F.grid_sample(FeatsR,AllGeoN,mode='bilinear', align_corners=False)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size(),device='cuda')-OCCLUDED.float(), torch.zeros(x0.size(),device='cuda')), dim=0)
        validation_loss=self.criterion(sample+1e-12, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, prog_bar=True,logger=True, on_step=True, on_epoch=True,sync_dist=True)
        return validation_loss
    
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
        
    def configure_optimizers(self):
        EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.04},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.2},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate*0.44},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]
        #optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate,momentum=0.9)
        optimizer=torch.optim.AdamW(param_grps,lr=self.learning_rate)
        """scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1910,
                                          cycle_mult=1.0,
                                          max_lr=0.01,
                                          min_lr=0.0001,
                                          warmup_steps=200,
                                          gamma=0.95)"""
        """scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = 10, # Maximum number of iterations.
                             eta_min = 1e-4)""" # Minimum learning rate.
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 1, T_mult = 1, eta_min = 1e-4)
        """scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            }"""
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200],gamma=0.7)
        return [optimizer],[scheduler]
    
    
import torch.nn as nn
class FastMcCnnInference(nn.Module):
    """
    Define the mc_cnn fast neural network
    """
    def __init__(self):
        super().__init__()
        self.in_channels = 1
        self.num_conv_feature_maps = 64
        self.conv_kernel_size = 3

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.num_conv_feature_maps, kernel_size=self.conv_kernel_size,padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,

            ),
        )

    # pylint: disable=arguments-differ
    # pylint: disable=no-else-return
    def forward(self, sample):
        with torch.no_grad():
            # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 2 dimensions
            features = self.conv_blocks(sample)
            return torch.squeeze(F.normalize(features, p=2.0, dim=1))
        
class MCCNWithDecisionNetwork_Dense_LM_N_2(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(MCCNWithDecisionNetwork_Dense_LM_N_2, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=FastMcCnnInference()
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        #print("SAMPLE  ==> ",x0.shape,x1.shape,dispnoc0.shape,MaskDef.shape,Mask0.shape,x_offset.shape)
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',torch.max(OCCLUDED))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',OCCLUDED.shape)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        validation_loss=self.criterion(sample+1e-12, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        """EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.2},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.4},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate*0.6},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]"""
        #optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate,momentum=0.9)
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        """scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1910,
                                          cycle_mult=1.0,
                                          max_lr=0.01,
                                          min_lr=0.0001,
                                          warmup_steps=200,
                                          gamma=0.95)"""
        """scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = 10, # Maximum number of iterations.
                             eta_min = 1e-4)""" # Minimum learning rate.
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 1, T_mult = 1, eta_min = 1e-4)
        """scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            }"""
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,60,70,80,90],gamma=0.7)
        return [optimizer],[scheduler]
    
class MATCHER(LightningModule):
    def __init__(self):
        super(MATCHER,self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]),reduction="none")
        self.residual_criterion=nn.MSELoss(reduction='none') # add a weighting parameter around 0.1 
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        self.Matcher3D=Matcher3D() 
    def training_step(self,batch,batch_idx):
        cubes,masqs,_=batch
        outcubes=self.Matcher3D(cubes)
        masqSurface=(masqs==2).float()
        masqNotSurface=(masqs>=1).float()
        training_loss=self.criterion(outcubes,masqSurface)*masqNotSurface
        training_loss=training_loss.sum().div(masqNotSurface.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss
    def validation_step(self,batch,batch_idx):
        cubes,masqs,_=batch
        outcubes=self.Matcher3D(cubes)
        masqSurface=(masqs==2).float()
        masqNotSurface=(masqs>=1).float()
        validation_loss=self.criterion(outcubes,masqSurface)*masqNotSurface
        validation_loss=validation_loss.sum().div(masqNotSurface.count_nonzero()+1e-12)
        gc.collect()
        self.log("validation_loss",validation_loss, on_epoch=True)
        return validation_loss
    def forward(self,x):
         return self.Matcher3D(x)

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)
        # ReduceOnPlateau scheduler 
        """reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        sch_val = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': "val_loss",
            'frequency': 1,
        }"""
        return[optimizer],[scheduler]

class UnetMlpMatcher(LightningModule):
    def __init__(self,Inplanes,patch_size,buffer,NANS=-999.0):
        super(UnetMlpMatcher, self).__init__()
        self.training=True
        self.Device=torch.device("cuda:0")
        self.nans=NANS
        self.residual_criterion=nn.MSELoss(reduction='none') # add a weighting parameter around 0.1 
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.PATCH_SIZE=patch_size
        self.BUFF=buffer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.BUFF]),
                            reduction="none")
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetworkOnCube(2*64)  
        self.Matcher3D=Matcher3D() 
    def compute_cube(self,FeatL,FeatR,Disp,Masq, NappeSupLoc, NappeInfLoc,xx,yy,W,x_offset):
        # Initialize Cube
        NappeInf=NappeInfLoc*Masq
        NappeSup=NappeSupLoc*Masq
        lower_lim=torch.min(NappeInf[NappeInf!=0.0]).item()
        upper_lim=torch.max(NappeSup[NappeSup!=0.0]).item()
        HGT= upper_lim-lower_lim
        # IT IS POSSIBLE TO DISCRETIZE THE COST VOLUME TO SUB PIXEL LEVES ==> ???
        """X_field = torch.arange(0,self.PATCH_SIZE,dtype=torch.int64).expand(NappeSupLoc.size()).to(self.Device)
        X_field=X_field.add(xx+x_offset)"""
        #print("xxxxxx    ",X_field)
        #print("XFIELD  ",X_field.shape)
        # Compute the Disparity Field
        #Masq_Field = torch.zeros(CUBE.size(),dtype=torch.int16).to(self.Device)
        DD_=torch.arange(0,HGT).unsqueeze(1).repeat_interleave(self.PATCH_SIZE,1)\
            .unsqueeze(2).repeat_interleave(self.PATCH_SIZE,2).cuda()
        #print("DDDDD ",DD_.shape)
        """for yy in range(self.PATCH_SIZE):
            for xx in range(self.PATCH_SIZE):
                mm_=torch.zeros(HGT).to(self.Device)
                mm_[NappeInfLoc[yy,xx]-lower_lim:NappeSupLoc[yy,xx]-lower_lim]=\
                        torch.ones((NappeSupLoc[yy,xx]-NappeInfLoc[yy,xx]))
                # Add and encoding of 2 
                mm_[round(Disp[yy,xx].item()-lower_lim)]=2
                Masq_Field[:,yy,xx]=mm_"""
        # Selection
        NappeSupLoc=NappeSupLoc.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        NappeInfLoc=NappeInfLoc.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        Masq3D=Masq.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        Disp=Disp.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        #print("disparity shape ",Disp.shape)
        Masq_Field=(DD_<NappeSupLoc-lower_lim) * (DD_>=NappeInfLoc-lower_lim)*Masq3D
        Masq_Field_DD= DD_== torch.round(Disp-lower_lim).mul(Masq).int()
        Masq_Field=Masq_Field.int()+Masq_Field_DD.int()
        #Masq_Field=Masq_Field*Masq.unsqueeze(0).repeat_interleave(HGT,0)
        #print("Field masq shape ",Masq_Field.shape," ", Masq_Field.device)
        """for d in range(HGT):
            FeatsRLoc=FeatR[:,yy:yy+self.PATCH_SIZE,xx+x_offset-(d+lower_lim):xx+x_offset-(d+lower_lim)+self.PATCH_SIZE]
            FeatsLR=torch.cat((,FeatsRLoc),0).unsqueeze(0)
            #print(" SHAPE OF CONCATENATION ",FeatsLR.shape)
            outSim=self.decisionNet(FeatsLR).sigmoid()
            #print("Shape of S",outSim.shape)
            CUBE[d,:,:].copy_(outSim.squeeze())#*Masq_X_D.float()""" #(0, 2, 3, 1)
        feat_ref=FeatL[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]
        cost = Variable(torch.FloatTensor(feat_ref.size()[0]*2, HGT,  feat_ref.size()[1],  feat_ref.size()[2]).zero_()).cuda()
        # COST SHAPE 64*2, DISRANGE,H,W
        for i in range(HGT):
            feat_dest=FeatR[:,yy:yy+self.PATCH_SIZE,xx+x_offset-(i+lower_lim):xx+x_offset-(i+lower_lim)+self.PATCH_SIZE]
            cost[:feat_ref.size()[0], i, :,:]   = feat_ref
            cost[feat_ref.size()[0]:, i, :,:]   = feat_dest
        cost = cost.contiguous()
        CUBE=self.decisionNet(cost).sigmoid().squeeze()
        #print("SHAPES ////////=======>  >>>>>  ", CUBE.shape,"    ", Masq_Field.shape)
        # Generate sample point cloud
        #raise ValueError("Intentionnaly stop ")
        return CUBE,Masq_Field, (lower_lim,upper_lim)

    def pad_cubes_for_batch(self,data):
        # tuple of elements from CubesDataset giving CUBE, Masq, (lower limit , upper limit )
        cubes,masks,_=zip(*data) # lims will not be used for now 
        #print("cubes rquires grad ? ",cubes[0].requires_grad)
        max_size = tuple(max(s) for s in zip(*[cube.shape for cube in cubes]))
        batch_shape = (len(cubes),) + max_size
        #print("batch shape ",batch_shape)
        batched_cubes = cubes[0].new(*batch_shape).zero_()
        batched_masks = masks[0].new(*batch_shape).zero_()
        for cub, pad_cub in zip(cubes, batched_cubes):
            """print("pd ",pad_cub.requires_grad)
            print("cub init ",cub.requires_grad)"""
            pad_cub[: cub.shape[0], : cub.shape[1], : cub.shape[2]].copy_(cub)
            #print("pd after  ",pad_cub.requires_grad)
        for mask, pad_mask in zip(masks, batched_masks):
            pad_mask[: mask.shape[0], : mask.shape[1], : mask.shape[2]].copy_(mask)
        """check lims for loss and supervision""" 
        return batched_cubes.unsqueeze(1),batched_masks.unsqueeze(1)

    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc,masqnoc,x_offset=batch
        # x0 BS, 1, H,W
        # Use the Model to get features and compute similarity
        FeatsL=self.feature(x0)
        FeatsR=self.feature(x1)
        H, W_r=FeatsR.size()[-2],FeatsR.size()[-1] # Because FeatsR takes the whole image 
        W_l=FeatsL.size()[-1]
        xx=secrets.randbelow(W_l-self.PATCH_SIZE)
        yy=secrets.randbelow(H-self.PATCH_SIZE)
        Disp=dispnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE] #BS, PATCH_SIZE, PATCH_SIZE
        Masq=masqnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]
        NappeSupLoc=torch.round(Disp+self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        NappeInfLoc=torch.round(Disp-self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        VarSizeCubes=[]
        for bb_ in range(FeatsL.size()[0]):
            VarSizeCubes.append(self.compute_cube(FeatsL[bb_,:],FeatsR[bb_,:],Disp[bb_,:],Masq[bb_,:],NappeSupLoc[bb_,:],
                                NappeInfLoc[bb_,:],xx,yy,W_r,x_offset[bb_]))
        # Pad Cubes to the same  Size 
        batched_cubes,batched_masqs=self.pad_cubes_for_batch(tuple(VarSizeCubes))
        outCubes=self.Matcher3D(batched_cubes)
        # Forward in Matcher 3D 
        masqSurface=(batched_masqs==2).float()
        masqNotSurface=(batched_masqs>=1).float()
        training_loss=self.criterion(outCubes,masqSurface)*masqNotSurface
        training_loss=training_loss.sum().div(masqNotSurface.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc,masqnoc,x_offset=batch
        # x0 BS, 1, H,W
        # Use the Model to get features and compute similarity
        FeatsL=self.feature(x0)
        FeatsR=self.feature(x1)
        print("Features righht shoae ",FeatsR.shape)
        H, W_r=FeatsR.size()[-2],FeatsR.size()[-1] # Because FeatsR takes the whole image 
        W_l=FeatsL.size()[-1]
        xx=secrets.randbelow(W_l-self.PATCH_SIZE)
        yy=secrets.randbelow(H-self.PATCH_SIZE)
        Disp=dispnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]# BS, PATCH_SIZE, PATCH_SIZE
        Masq=masqnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]
        NappeSupLoc=torch.round(Disp+self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        NappeInfLoc=torch.round(Disp-self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        #print("nappe inf shape ",NappeInfLoc.shape)
        VarSizeCubes=[]
        for bb_ in range(FeatsL.size()[0]):
            VarSizeCubes.append(self.compute_cube(FeatsL[bb_,:],FeatsR[bb_,:],Disp[bb_,:],Masq[bb_,:],NappeSupLoc[bb_,:],
                                NappeInfLoc[bb_,:],xx,yy,W_r,x_offset[bb_]))
        # Pad Cubes to the same  Size 
        batched_cubes,batched_masqs=self.pad_cubes_for_batch(tuple(VarSizeCubes))
        print("SHAPE OF BATCHED ",batched_cubes.shape)
        for bb in range(batched_cubes.size()[0]):
            i=randint(1,200)
            generate_pointcloud(batched_cubes[bb].squeeze().to(torch.device("cpu")), "./CUBE_from_disp_{}.ply".format(str(i)))
            generate_pointcloud(batched_masqs[bb].squeeze().to(torch.device("cpu")), "./MASQ_from_disp_{}.ply".format(str(i)))
        outCubes=self.Matcher3D(batched_cubes)
        # Forward in Matcher 3D 
        masqSurface=(batched_masqs==2).float()
        masqNotSurface=(batched_masqs>=1).float()
        validation_loss=self.criterion(outCubes,masqSurface)*masqNotSurface
        validation_loss=validation_loss.sum().div(masqNotSurface.count_nonzero()+1e-12)
        gc.collect()
        self.log("validation_loss",validation_loss, on_epoch=True)
        return validation_loss

    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.Matcher3D.parameters(),lr=self.learning_rate) # Update only matcher parameters 
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)
        # ReduceOnPlateau scheduler 
        """reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        sch_val = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': "val_loss",
            'frequency': 1,
        }"""
        return [optimizer],[scheduler]



class UnetMlpCubeMatcher(LightningModule):
    def __init__(self,Inplanes,patch_size,buffer,NANS=-999.0,FEATURES_ON=False):
        super(UnetMlpCubeMatcher, self).__init__()
        self.training=True
        self.Device=torch.device("cuda:0")
        self.nans=NANS
        self.residual_criterion=nn.MSELoss(reduction='none') # add a weighting parameter around 0.1 
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.PATCH_SIZE=patch_size
        self.BUFF=buffer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.BUFF]),
                            reduction="none")
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetworkOnNCubes(2*64) 
        self.FEATS_ON=FEATURES_ON
        if self.FEATS_ON: 
            self.Matcher3D=Matcher3D(in_channels=2*64+1)
            print("<============  FEATS_ON activated: Taking feature information into consideration =========>  ")
        else:
            self.Matcher3D=Matcher3D()  
        self.LastOnly=True
    def compute_cube(self,FeatL,FeatR,Disp,Masq, NappeSupLoc, NappeInfLoc,xx,yy,W,x_offset):
        # Initialize Cube
        NappeInf=NappeInfLoc*Masq
        NappeSup=NappeSupLoc*Masq
        NappeInf=NappeInf[NappeInf!=0.0]
        NappeSup=NappeSup[NappeSup!=0.0]
        lower_lim=torch.min(NappeInf).item() if NappeInf.numel()!=0 else -5
        upper_lim=torch.max(NappeSup).item() if NappeSup.numel()!=0 else  5
        HGT= upper_lim-lower_lim+1
        DD_=torch.arange(0,HGT).unsqueeze(1).repeat_interleave(self.PATCH_SIZE,1)\
            .unsqueeze(2).repeat_interleave(self.PATCH_SIZE,2).cuda()
        NappeSupLoc=NappeSupLoc.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        NappeInfLoc=NappeInfLoc.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        Masq3D=Masq.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        Disp=Disp.unsqueeze(0).repeat_interleave(DD_.size()[0],0)
        #print("disparity shape ",Disp.shape)
        Masq_Field=(DD_<=NappeSupLoc-lower_lim) * (DD_>=NappeInfLoc-lower_lim)*Masq3D
        Masq_Field_DD= DD_== torch.round(Disp+1.0).sub_(lower_lim).int()
        Masq_Field_DD=Masq_Field_DD*Masq
        Masq_Field=Masq_Field+Masq_Field_DD
        feat_ref=FeatL[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]
        #print("F===========    EATS REGERENCE SIZE  =========",feat_ref.shape)
        cost = Variable(torch.FloatTensor(feat_ref.size()[0]*2, HGT,  feat_ref.size()[1],  \
            feat_ref.size()[2]).zero_().add(1.0).mul(0.5)).cuda()
        # COST SHAPE 64*2, DISRANGE,H,W
        for i in range(HGT):
            if (xx+x_offset+self.PATCH_SIZE<=(i+lower_lim)) or (xx+x_offset>=W+(i+lower_lim)):
                #print(' FIRST CONDITION   SATISFIED +++++++  ')
                cost[:feat_ref.size()[0], i, :,:]=feat_ref
            elif (xx+x_offset<(i+lower_lim)):
                #print("Positional origin on right image =====>  ",xx+x_offset, "  LOWER LIMIT  ", lower_lim, "IIIIII  ",i)
                cost[:feat_ref.size()[0], i, :,:].copy_(feat_ref)
                cost[feat_ref.size()[0]:, i, :,i+lower_lim-(xx+x_offset):].copy_(FeatR[:,yy:yy+self.PATCH_SIZE,0:xx+x_offset-(i+lower_lim)+self.PATCH_SIZE])
            elif (xx+x_offset>i+lower_lim+W-self.PATCH_SIZE):
                #print("Positional origin on right image =====>  ",xx+x_offset, "  LOWER LIMIT  ", lower_lim, "IIIIII  ",i, "  WWWWW ")
                cost[:feat_ref.size()[0], i, :,:].copy_(feat_ref)
                cost[feat_ref.size()[0]:, i, :,0:W+(i+lower_lim)-xx-x_offset].copy_(FeatR[:,yy:yy+self.PATCH_SIZE,xx+x_offset-(i+lower_lim):])  
            else:
                cost[:feat_ref.size()[0], i, :,:].copy_(feat_ref)
                cost[feat_ref.size()[0]:, i, :,:].copy_(FeatR[:,yy:yy+self.PATCH_SIZE,xx+x_offset-(i+lower_lim):xx+x_offset-(i+lower_lim)+self.PATCH_SIZE])
        cost = cost.contiguous()
        return cost,Masq_Field, (lower_lim,upper_lim)

    def pad_cubes_for_batch(self,data):
        # tuple of elements from CubesDataset giving CUBE, Masq, (lower limit , upper limit )
        cubes,masks,_=zip(*data) # lims will not be used for now 
        #print("cubes rquires grad ? ",cubes[0].requires_grad)
        max_size = tuple(max(s) for s in zip(*[cube.shape for cube in cubes]))
        #print(" PADDING FEATURES CUBES +++++++++++ CUBE MAX SHAPE  ",max_size)
        max_size_masq=tuple(max(s) for s in zip(*[mask.shape for mask in masks]))
        #print(" PADDING FEATURES CUBES +++++++++++ MASQ MAX SHAPE  ",max_size_masq)
        batch_shape = (len(cubes),) + max_size
        #print(" PADDING FEATURES CUBES +++++++++++ CUBES BATCH NEW SHAPE  ",batch_shape)
        batch_shape_masq = (len(masks),) + max_size_masq
        #print(" PADDING FEATURES CUBES +++++++++++ MASQS BATCH NEW SHAPE  ",batch_shape_masq)
        #print("batch shape ",batch_shape)
        batched_cubes = cubes[0].new(*batch_shape).zero_().cuda()
        #print("SHAPE OF FEATURES BASED BATCHED CUBES ",  batched_cubes.shape)
        #batched_cubes.requires_grad_(True)
        """if self.LastOnly:
            batched_cubes.requires_grad_(True)"""
        batched_masks = masks[0].new(*batch_shape_masq).zero_().cuda()
        for cub, dd_ in zip(cubes,tuple([b for b in range(batched_cubes.size()[0])])):
            """print("pd ",pad_cub.requires_grad)
            print("cub init ",cub.requires_grad)"""
            batched_cubes[dd_,:cub.shape[0], : cub.shape[1], : cub.shape[2],: cub.shape[3]].copy_(cub)
            #pad_cub[: cub.shape[0], : cub.shape[1], : cub.shape[2],: cub.shape[3]].copy_(cub)
            #print("pd after  ",pad_cub.requires_grad)
        for mask, dd_ in zip(masks,tuple([b for b in range(batched_masks.size()[0])])):
            batched_masks[dd_,: mask.shape[0], : mask.shape[1], : mask.shape[2]].copy_(mask)
        """check lims for loss and supervision""" 
        return batched_cubes,batched_masks

    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc,masqnoc,x_offset=batch
        # x0 BS, 1, H,W
        # Use the Model to get features and compute similarity
        FeatsL=self.feature(x0)
        FeatsR=self.feature(x1)
        H, W_r=FeatsR.size()[-2],FeatsR.size()[-1] # Because FeatsR takes the whole image 
        W_l=FeatsL.size()[-1]
        xx=secrets.randbelow(W_l-self.PATCH_SIZE)
        yy=secrets.randbelow(H-self.PATCH_SIZE)
        Disp=dispnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE] #BS, PATCH_SIZE, PATCH_SIZE
        Masq=masqnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]
        NappeSupLoc=torch.round(Disp+self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        NappeInfLoc=torch.round(Disp-self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        VarSizeCubes=[]
        for bb_ in range(FeatsL.size()[0]):
            VarSizeCubes.append(self.compute_cube(FeatsL[bb_,:],FeatsR[bb_,:],Disp[bb_,:],Masq[bb_,:],NappeSupLoc[bb_,:],
                                NappeInfLoc[bb_,:],xx,yy,W_r,x_offset[bb_]))
        # Pad Cubes to the same  Size 
        batched_cubes,batched_masqs=self.pad_cubes_for_batch(tuple(VarSizeCubes))
        if self.FEATS_ON:
            batched_cubes=torch.cat((batched_cubes,self.decisionNet(batched_cubes).sigmoid()),1)
        else:
            batched_cubes=self.decisionNet(batched_cubes).sigmoid()
        #batched_cubes.requires_grad_(True)
        outCubes=self.Matcher3D(batched_cubes).squeeze()
        # Forward in Matcher 3D 
        masqSurface=(batched_masqs==2).float()
        #print("CHECK IF MASK SURFACE IS SHOWING VALUIES ======> ",masqSurface.count_nonzero(), "  <===========  ")
        masqNotSurface=(batched_masqs>=1).float()
        training_loss=self.criterion(outCubes,masqSurface)*masqNotSurface
        training_loss=training_loss.sum().div(masqNotSurface.count_nonzero()+1e-12)
        #training_loss=balanced_binary_cross_entropy(outCubes, masqSurface,masqNotSurface, pos_w=self.BUFF, neg_w=1.0)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc,masqnoc,x_offset=batch
        # x0 BS, 1, H,W
        # Use the Model to get features and compute similarity
        FeatsL=self.feature(x0)
        FeatsR=self.feature(x1)
        H, W_r=FeatsR.size()[-2],FeatsR.size()[-1] # Because FeatsR takes the whole image 
        W_l=FeatsL.size()[-1]
        xx=secrets.randbelow(W_l-self.PATCH_SIZE)
        yy=secrets.randbelow(H-self.PATCH_SIZE)
        Disp=dispnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE] #BS, PATCH_SIZE, PATCH_SIZE
        Masq=masqnoc[:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]
        NappeSupLoc=torch.round(Disp+self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        NappeInfLoc=torch.round(Disp-self.BUFF).int() #BS, PATCH_SIZE, PATCH_SIZE
        VarSizeCubes=[]
        #print("batch size for the validatoion STEp ", FeatsL.shape)
        #print("batch size for the validatoion STEp  FEATS RIGHT   ", FeatsR.shape)
        for bb_ in range(FeatsL.size()[0]):
            VarSizeCubes.append(self.compute_cube(FeatsL[bb_,:],FeatsR[bb_,:],Disp[bb_,:],Masq[bb_,:],NappeSupLoc[bb_,:],
                                NappeInfLoc[bb_,:],xx,yy,W_r,x_offset[bb_]))
        """for el in VarSizeCubes:
            i=randint(1,200)
            generate_pointcloud(self.decisionNet(el[0].unsqueeze(0)).sigmoid().squeeze().to(torch.device("cpu")), "./CUBE_from_disp_FeatsOFF_{}.ply".format(str(i)))
            generate_pointcloud(el[1].squeeze().squeeze().to(torch.device("cpu")), "./MASQ_from_disp_FeatsOFF_{}.ply".format(str(i)))"""
        # Pad Cubes to the same  Size 
        batched_cubes,batched_masqs=self.pad_cubes_for_batch(tuple(VarSizeCubes))
        #print("BATCHED FEATURE CUBES CREATED ", batched_cubes.shape)
        #print("BATCHED MASKS CREATED      ",batched_masqs.shape)
        if self.FEATS_ON:
            batched_cubes=torch.cat((batched_cubes,self.decisionNet(batched_cubes).sigmoid()),1)
        else:
            batched_cubes=self.decisionNet(batched_cubes).sigmoid()
            #print("====================   >  SIMILARITY CUBES GENERATED <  ==============     ",sim_cubes.shape)
            """for bb in range(batched_cubes.size()[0]):
                i=randint(1,200)
                print("SIM CUBE SLICE SHAPE ===========    ",batched_cubes[bb].squeeze().shape)
                generate_pointcloud(batched_cubes[bb].squeeze().to(torch.device("cpu")), "./CUBE_from_disp_FeatsON_{}.ply".format(str(i)))
                generate_pointcloud(batched_masqs[bb].squeeze().to(torch.device("cpu")), "./MASQ_from_disp_FeatsON_{}.ply".format(str(i)))"""
        #batched_cubes.requires_grad_(True)
        outCubes=self.Matcher3D(batched_cubes).squeeze()
        # Forward in Matcher 3D 
        masqSurface=(batched_masqs==2).float()
        masqNotSurface=(batched_masqs>=1).float()
        # Display Some Masqs to be sure you get good results 
        """for bb in range(masqNotSurface.size()[0]):
            i=randint(300,500)
            generate_pointcloud(masqNotSurface[bb].squeeze().to(torch.device("cpu")), "./Masq_Bad_{}.ply".format(str(i)))"""
        #print("CHECK IF MASK SURFACE IS SHOWING VALUIES ======> ",masqSurface.count_nonzero(), "  <===========  ")
        validation_loss=self.criterion(outCubes,masqSurface)*masqNotSurface
        validation_loss=validation_loss.sum().div(masqNotSurface.count_nonzero()+1e-12)
        #validation_loss=balanced_binary_cross_entropy(outCubes, masqSurface,masqNotSurface, pos_w=self.BUFF, neg_w=1.0)
        gc.collect()
        self.log("validation_loss",validation_loss, on_epoch=True)
        return validation_loss

    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         print("<======   Forward is accessed during training ======> ")
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out

    def configure_optimizers(self):
        """allparameters=list(self.parameters())
        trainableparams=list(filter(lambda p:p.requires_grad,allparameters))
        optimizer=torch.optim.AdamW(trainableparams,lr=self.learning_rate) # Update only matcher parameters 
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)"""
        # ReduceOnPlateau scheduler 
        param_grps=[
            {'params':self.feature.parameters(),'lr':self.learning_rate**2},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate**2},
            {'params':self.Matcher3D.parameters(), 'lr': self.learning_rate},
        ]
        optimizer=torch.optim.AdamW(param_grps,lr=self.learning_rate) # Update only matcher parameters 
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,60,70,80,90],gamma=0.7)
        """reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=[1e-8,1e-8,1e-6],
            verbose=True
        )
        sch_val = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': "validation_loss",
            'frequency': 1,
        }"""
        return [optimizer],[scheduler]

        
if __name__=="__main__":
     model=UNETWithDecisionNetwork_LM5D(32)
     x0=torch.rand((2,1,64,64))
     x1=torch.rand((2,1,64,64))
     dispnoc0=torch.rand((2,1,64,64))
     print(dispnoc0)
     batch=x0,x1,dispnoc0
     loss=model.training_step(batch,0)
     print(loss)
     
