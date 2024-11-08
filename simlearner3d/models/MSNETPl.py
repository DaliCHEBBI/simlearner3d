from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from pytorch_lightning import LightningModule
from models.model import DecisionNetwork
from networks_other import init_weights
#from datasets.StereoSatDatasetMS import StereoTrSatDataset, StereoValSatDataset 
#import kornia.augmentation as AugEngine
import os 
import gc

def MemStatusPl(loc):
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used before @ "+ loc, round((used_memory/total_memory) * 100, 2))

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes,track_running_stats=True))

def conv1x1(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes,track_running_stats=True))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class MSCAM(nn.Module):
	def __init__(self, in_channels,r):
		super(MSCAM, self).__init__()
		inter_channels=int(float(in_channels/r))
		# LOCAL ATTENTION
		alocal_attention=[]
		aglobal_attention=[]
		alocal_attention.append(conv1x1(in_channels,inter_channels,1,1,0,1))
		alocal_attention.append(nn.ReLU(inplace=True))
		alocal_attention.append(conv1x1(inter_channels,in_channels,1,1,0,1))
		self.local_attention=nn.Sequential(*alocal_attention)

		# GLOBAL ATTENTION Sequential
		#aglobal_attention.append(nn.AdaptiveAvgPool2d(1))
		aglobal_attention.append(conv1x1(in_channels,inter_channels,1,1,0,1))
		aglobal_attention.append(nn.ReLU(inplace=True))
		aglobal_attention.append(conv1x1(inter_channels,in_channels,1,1,0,1))
		self.global_attention=nn.Sequential(*aglobal_attention)
	def forward(self,X,Y):
		X_all=X+Y
		#print("all addition soize  ",X_all.size())
		xl=self.local_attention(X_all)
		xg=self.global_attention(X_all)
		#print("local   and global sizes ",xl.size(), xg.size())
		xlg=xl+xg
		weight=torch.sigmoid(xlg)
		return X.mul(weight)+ Y.mul(weight.mul(-1.0).add(1.0))


class MSNet(torch.nn.Module):
	def __init__(self,Inplanes):
		super(MSNet, self).__init__()
		self.inplanes = Inplanes
		#self.params=param
		self.firstconv = nn.Sequential(convbn(1, 32, 3, 1, 1, 1),
										nn.ReLU(inplace=True),
										convbn(32, 32, 3, 1, 1, 1),
										nn.ReLU(inplace=True),
										convbn(32, 32, 3, 1, 1, 1),
										nn.ReLU(inplace=True))
		
		self.layer1 = self._make_layer(BasicBlock, 64, 3, 1,1,1)
		self.layer2 = self._make_layer(BasicBlock, 64, 3, 1,1,1) 
		self.layer3 = self._make_layer(BasicBlock, 64, 3, 1,1,1)
		self.layer4 = self._make_layer(BasicBlock, 64, 3, 1,1,1)

		# Multi-Scale fusion attention modules 
		self.MultiScaleFeatureFuser3_4=MSCAM(64,4)
		self.MultiScaleFeatureFuser2_3_4=MSCAM(64,4)
		self.MultiScaleFeatureFuser1_2_3_4=MSCAM(64,4)
		
		# common network to compute the last features 
		self.common=nn.Sequential(convbn(64,64,3, 1, 1, 1),
									nn.ReLU(inplace=True),
									convbn(64,64,3, 1, 1, 1),
									nn.ReLU(inplace=True),
									nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation = 1, bias=False))

	def _make_layer(self, block, planes, blocks, stride, pad, dilation):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
					nn.Conv2d(self.inplanes, planes * block.expansion,
					kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(planes * block.expansion))
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
		#self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(planes, planes,1,None,pad,dilation))
		return nn.Sequential(*layers)
	# msnet attention forward 
	def forward(self,x):
		# interpolate input images
		x_by2 = F.interpolate(x, (x.size()[2]//2,x.size()[3]//2),mode='bilinear')
		x_by4 = F.interpolate(x, (x.size()[2]//4,x.size()[3]//4),mode='bilinear')
		x_by8 = F.interpolate(x, (x.size()[2]//8,x.size()[3]//8),mode='bilinear') 
		x1=self.firstconv(x)
		x2=self.firstconv(x_by2)
		x3=self.firstconv(x_by4)
		x4=self.firstconv(x_by8)
		#print(x1.shape,x2.shape,x3.shape,x4.shape)
		x1=self.layer1(x1)
		x2=self.layer2(x2)
		x3=self.layer3(x3)
		x4=self.layer4(x4)
		#print(x1.shape,x2.shape,x3.shape,x4.shape)
		x4=F.interpolate(x4, (x3.size()[2],x3.size()[3]),mode='bilinear')
		x_3_4=self.MultiScaleFeatureFuser3_4(x3,x4)
		x_3_4=F.interpolate(x_3_4, (x2.size()[2],x2.size()[3]),mode='bilinear')
		x_2_3_4=self.MultiScaleFeatureFuser2_3_4(x2,x_3_4)
		x_2_3_4=F.interpolate(x_2_3_4, (x1.size()[2],x1.size()[3]),mode='bilinear')
		x_all=self.MultiScaleFeatureFuser1_2_3_4(x1,x_2_3_4)
		# Pass to last conv block
		x_all=self.common(x_all)
		return x_all


class MSNetPatch(torch.nn.Module):
	def __init__(self,Inplanes):
		super(MSNetPatch, self).__init__()
		self.inplanes = Inplanes
		#self.params=param
		self.firstconv = nn.Sequential(convbn(1, 32, 3, 1, 1, 1),
										nn.ReLU(inplace=True),
										convbn(32, 32, 3, 1, 1, 1),
										nn.ReLU(inplace=True),
										convbn(32, 32, 3, 1, 1, 1),
										nn.ReLU(inplace=True))
		
		self.layer1 = self._make_layer(BasicBlock, 64, 3, 1,1,1)
		self.layer2 = self._make_layer(BasicBlock, 64, 3, 1,1,1) 
		self.layer3 = self._make_layer(BasicBlock, 64, 3, 1,1,1)
		self.layer4 = self._make_layer(BasicBlock, 64, 3, 1,1,1)

		# Multi-Scale fusion attention modules 
		self.MultiScaleFeatureFuser3_4=MSCAM(64,4)
		self.MultiScaleFeatureFuser2_3_4=MSCAM(64,4)
		self.MultiScaleFeatureFuser1_2_3_4=MSCAM(64,4)
		
		# common network to compute the last features 
		self.common=nn.Sequential(convbn(64,64,3, 1, 0, 1),
									nn.ReLU(inplace=True),
									convbn(64,64,3, 1, 0, 1),
									nn.ReLU(inplace=True),
									nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation = 1, bias=False))

	def _make_layer(self, block, planes, blocks, stride, pad, dilation):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
					nn.Conv2d(self.inplanes, planes * block.expansion,
					kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(planes * block.expansion))
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
		#self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(planes, planes,1,None,pad,dilation))
		return nn.Sequential(*layers)
	# msnet attention forward 
	def forward(self,x):
		# interpolate input images
		"""xmid,ymid=28,28
		x_by_Res0=x[:,:,ymid-3:ymid+4,xmid-3:xmid+4]
		x_by_Res2=x[:,:,ymid-7:ymid+7,xmid-7:xmid+7] 
		x_by_Res4=x[:,:,ymid-14:ymid+14,xmid-14:xmid+14]
		#print(x_by_Res0.shape,x_by_Res2.shape,x_by_Res4.shape)
		# x here a multi patch set x is 56 x56
		x_by2 = F.interpolate(x_by_Res2, (7,7),mode='bilinear')
		x_by4 = F.interpolate(x_by_Res4, (7,7),mode='bilinear')  
		x_by8 = F.interpolate(x, (7,7),mode='bilinear')     
		x1=self.firstconv(x_by_Res0)
		x2=self.firstconv(x_by2)
		x3=self.firstconv(x_by4)
		x4=self.firstconv(x_by8)"""
        
		x1=self.firstconv(x[:,0,:,:].unsqueeze(1))
		x2=self.firstconv(x[:,1,:,:].unsqueeze(1))
		x3=self.firstconv(x[:,2,:,:].unsqueeze(1))
		x4=self.firstconv(x[:,3,:,:].unsqueeze(1))
		#print(x1.shape,x2.shape,x3.shape,x4.shape)
		x1=self.layer1(x1)
		x2=self.layer2(x2)
		x3=self.layer3(x3)
		x4=self.layer4(x4)
		#print(x1.shape,x2.shape,x3.shape,x4.shape)
		x_3_4=self.MultiScaleFeatureFuser3_4(x3,x4)
		x_2_3_4=self.MultiScaleFeatureFuser2_3_4(x2,x_3_4)
		x_all=self.MultiScaleFeatureFuser1_2_3_4(x1,x_2_3_4)
		# Pass to last conv block
		x_all=self.common(x_all)
		return x_all
    
class MSNETWithDecisionNetwork_Dense_LM_SubPix(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=1,false2=4,NANS=-999.0):
        super(MSNETWithDecisionNetwork_Dense_LM_SubPix, self).__init__()
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
        self.feature=MSNet(self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
        
    def Gather2DBatch(self,FMap,X,Y):
        OFF=torch.zeros((FMap.size()[0],FMap.size()[1],X.size()[-2],X.size()[-1]),dtype=torch.float32,device='cuda')
        for bb_ in range(FMap.shape[0]):
            OFF[bb_,:,:,:]=FMap[bb_,:,Y[bb_,:,:],X[bb_,:,:]]
        return OFF
    def Gather2DBatchFasterCUDA(self,FMap,X,Y):
        OFF=torch.cat([FMap[bb_,:,Y[bb_,:,:],X[bb_,:,:]].unsqueeze(0) for bb_ in range(FMap.shape[0])],0)
        return OFF

    def InterpolateSubPix2D(self,FeatMap,aGeox,aGeoy):
        '''
        FeatMap: BS,64,H,W
        aGeox  : BS,H,W
        aGeoy  : BS,H,W

        handles subpixellar deformation maps and uses einsum to gather 
        surrounding postions to compute the bilinear interpolation

        '''
        DEF_X=torch.mul((aGeox>=0),(aGeox<FeatMap.shape[-1]))
        DEF_Y=torch.mul((aGeoy>=0),(aGeoy<FeatMap.shape[-2]))
        Y_1=torch.floor(aGeoy.float()).mul(DEF_Y)
        X_1=torch.floor(aGeox.float()).mul(DEF_X)
        Y_2=torch.ceil(aGeoy.float()).mul(DEF_Y)
        X_2=torch.ceil(aGeox.float()).mul(DEF_X)

        aGeoy=aGeoy.mul(DEF_Y)
        aGeox=aGeox.mul(DEF_X)

        Y_2=Y_2.mul(Y_2<FeatMap.shape[-2])
        X_2=X_2.mul(X_2<FeatMap.shape[-1])

        #aMAP_11=FeatMap[:,:,Y_1.to(torch.int64),X_1.to(torch.int64)]
        aMAP_11=self.Gather2DBatchFasterCUDA(FeatMap, X_1.to(torch.int64), Y_1.to(torch.int64))
        #aMAP_21=FeatMap[:,:,Y_2.to(torch.int64),X_1.to(torch.int64)]
        aMAP_21=self.Gather2DBatchFasterCUDA(FeatMap, X_1.to(torch.int64), Y_2.to(torch.int64))
        #aMAP_12=FeatMap[:,:,Y_1.to(torch.int64),X_2.to(torch.int64)]
        aMAP_12=self.Gather2DBatchFasterCUDA(FeatMap, X_2.to(torch.int64), Y_1.to(torch.int64))
        #aMAP_22=FeatMap[:,:,Y_2.to(torch.int64),X_2.to(torch.int64)]
        aMAP_22=self.Gather2DBatchFasterCUDA(FeatMap, X_2.to(torch.int64), Y_2.to(torch.int64))
        #print(aMAP_11.shape)

        # interpolate 
        Y_1GeoY=aGeoy-Y_1
        Y_2GeoY=Y_2-aGeoy
        X_1GeoX=aGeox-X_1
        X_2GeoX=X_2-aGeox
        Y_2GeoY[Y_2GeoY==0]=1
        X_2GeoX[X_2GeoX==0]=1

        TERM_1=torch.einsum("bijk,bjk->bijk",[aMAP_11,Y_2GeoY])+torch.einsum("bijk,bjk->bijk",[aMAP_21,Y_1GeoY])
        TERM_2=torch.einsum("bijk,bjk->bijk",[aMAP_12,Y_2GeoY])+torch.einsum("bijk,bjk->bijk",[aMAP_22,Y_1GeoY])

        anInterpFeatMap=torch.einsum("bijk,bjk->bijk",[TERM_1,X_2GeoX])+torch.einsum("bijk,bjk->bijk",[TERM_2,X_1GeoX])

        return anInterpFeatMap

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
        EarlyFeaturesParams.extend(self.feature.firstconv.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.layer1.parameters())
        MiddleFeauresParams.extend(self.feature.layer2.parameters())
        MiddleFeauresParams.extend(self.feature.layer3.parameters())
        MiddleFeauresParams.extend(self.feature.layer4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser2_3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser1_2_3_4.parameters())

        #BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        LastFeatureParams=[]
        LastFeatureParams.extend(self.feature.common.parameters())
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.04},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.2},
            {'params':LastFeatureParams,'lr':self.learning_rate*0.44},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]
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
    
###############################################################################
class MSNETWithDecisionNetwork_2D_SubPix(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=1,false2=4,NANS=-999.0):
        super(MSNETWithDecisionNetwork_2D_SubPix, self).__init__()
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
        self.feature=MSNet(self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
        # initialize model parameters 
        init_weights(self, init_type='normal')
    
    
    def training_step(self,batch,batch_idx):
        rot1=0
        rot2=2*np.pi
        x0,x1,dispnocX,dispnocY=batch
    
        dispnocX=dispnocX.unsqueeze(1)
        dispnocY=dispnocY.unsqueeze(1)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        
        # negatives pattern 
        # |---------------------|  
        # |---------------------| 
        # |-------      --------|  
        # |------        -------|
        # |------        -------|
        # |-------      --------| 
        # |---------------------|   
        # |---------------------| 
        
        # Random shift in x direction 
        Offset_neg_radius=((self.false1 - self.false2) * torch.rand(dispnocX.size(),device='cuda') + self.false2)
    
        # Random shift in y direction  
        Offset_neg_angle=((rot1 - rot2) * torch.rand(dispnocY.size(),device='cuda') + rot2)
        
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        Index_X=torch.arange(0,dispnocX.size()[-1],device='cuda')
        Index_X=Index_X.expand(dispnocX.size()[-2],dispnocX.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        
        Index_Y=torch.arange(0,dispnocY.size()[-2],device='cuda').resize(dispnocY.size()[-2],1)
        Index_Y=Index_Y.expand(dispnocY.size()[-2],dispnocY.size()[-1]).unsqueeze(0).repeat(x0.size()[0],1,1) # BS,H,W
        
        
        OffpX=Index_X+dispnocX  
        OffpY=Index_Y+dispnocY
        
        OffnX=Index_X+(dispnocX+Offset_neg_radius*torch.cos(Offset_neg_angle))
        OffnY=Index_Y+(dispnocY+Offset_neg_radius*torch.sin(Offset_neg_angle))
    
        AllGeoP=torch.stack((2*OffpX.add(0.5).squeeze(1).div(FeatsR.size()[-1]),
                             2*OffpY.add(0.5).squeeze(1).div(FeatsR.size()[-2])),-1).sub(1.0)
        
        AllGeoN=torch.stack((2*OffnX.add(0.5).squeeze(1).div(FeatsR.size()[-1]),
                                 2*OffnY.add(0.5).squeeze(1).div(FeatsR.size()[-2])),-1).sub(1.0)
        
        FeatsR_plus=F.grid_sample(FeatsR,AllGeoP,mode='bilinear', align_corners=False)
        FeatsR_minus=F.grid_sample(FeatsR,AllGeoN,mode='bilinear', align_corners=False)

        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size(),device='cuda'), torch.zeros(x0.size(),device='cuda')), dim=0)
        
        training_loss=self.criterion(sample+1e-20, target).mean()
        
        gc.collect()
        self.log("training_loss",training_loss, prog_bar=True,logger=True, on_step=True, on_epoch=True,sync_dist=True)
        return training_loss


    def validation_step(self,batch,batch_idx):
        rot1=0
        rot2=2*np.pi
        x0,x1,dispnocX,dispnocY=batch
    
        dispnocX=dispnocX.unsqueeze(1)
        dispnocY=dispnocY.unsqueeze(1)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        
        # negatives pattern 
        # |---------------------|  
        # |---------------------| 
        # |-------      --------|  
        # |------        -------|
        # |------        -------|
        # |-------      --------| 
        # |---------------------|   
        # |---------------------| 
        
        # Random shift in x direction 
        Offset_neg_radius=((self.false1 - self.false2) * torch.rand(dispnocX.size(),device='cuda') + self.false2)
    
        # Random shift in y direction  
        Offset_neg_angle=((rot1 - rot2) * torch.rand(dispnocY.size(),device='cuda') + rot2)
        
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        Index_X=torch.arange(0,dispnocX.size()[-1],device='cuda')
        Index_X=Index_X.expand(dispnocX.size()[-2],dispnocX.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        
        Index_Y=torch.arange(0,dispnocY.size()[-2],device='cuda').resize(dispnocY.size()[-2],1)
        Index_Y=Index_Y.expand(dispnocY.size()[-2],dispnocY.size()[-1]).unsqueeze(0).repeat(x0.size()[0],1,1) # BS,H,W
        
        
        OffpX=Index_X+dispnocX  
        OffpY=Index_Y+dispnocY
        
        OffnX=Index_X+(dispnocX+Offset_neg_radius*torch.cos(Offset_neg_angle))
        OffnY=Index_Y+(dispnocY+Offset_neg_radius*torch.sin(Offset_neg_angle))
    
        AllGeoP=torch.stack((2*OffpX.add(0.5).squeeze(1).div(FeatsR.size()[-1]),
                             2*OffpY.add(0.5).squeeze(1).div(FeatsR.size()[-2])),-1).sub(1.0)
        
        AllGeoN=torch.stack((2*OffnX.add(0.5).squeeze(1).div(FeatsR.size()[-1]),
                                 2*OffnY.add(0.5).squeeze(1).div(FeatsR.size()[-2])),-1).sub(1.0)
        
        FeatsR_plus=F.grid_sample(FeatsR,AllGeoP,mode='bilinear', align_corners=False)
        FeatsR_minus=F.grid_sample(FeatsR,AllGeoN,mode='bilinear', align_corners=False)

        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size(),device='cuda'), torch.zeros(x0.size(),device='cuda')), dim=0)
        
        validation_loss=self.criterion(sample+1e-20, target).mean()
        
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
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200],gamma=0.7)
        return [optimizer],[scheduler]
     
    def config_optimizers_finetuning(self):
        EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.firstconv.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.layer1.parameters())
        MiddleFeauresParams.extend(self.feature.layer2.parameters())
        MiddleFeauresParams.extend(self.feature.layer3.parameters())
        MiddleFeauresParams.extend(self.feature.layer4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser2_3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser1_2_3_4.parameters())

        #BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        LastFeatureParams=[]
        LastFeatureParams.extend(self.feature.common.parameters())
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.04},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.2},
            {'params':LastFeatureParams,'lr':self.learning_rate*0.44},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]
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
###############################################################################    


class MSNETWithDecisionNetwork_Dense_LM_N_2(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=1,false2=4,NANS=-999.0):
        super(MSNETWithDecisionNetwork_Dense_LM_N_2, self).__init__()
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
        self.feature=MSNet(self.inplanes)
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
        self.log("val_loss",validation_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.firstconv.parameters())
        EarlyFeaturesParams.extend(self.feature.layer1.parameters())
        EarlyFeaturesParams.extend(self.feature.layer2.parameters())
        EarlyFeaturesParams.extend(self.feature.layer3.parameters())
        EarlyFeaturesParams.extend(self.feature.layer4.parameters())
        
        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser2_3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser1_2_3_4.parameters())


        #BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        LastFeatureParams=[]
        LastFeatureParams.extend(self.feature.common.parameters())
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate},
            {'params':MiddleFeauresParams,'lr':self.learning_rate},
            {'params':LastFeatureParams,'lr':self.learning_rate},
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
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,200],gamma=0.7)
        return [optimizer],[scheduler]    
 
class MSNETWithDecisionNetwork_Dense_LM_N_3(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=1,false2=4,NANS=-999.0):
        super(MSNETWithDecisionNetwork_Dense_LM_N_3, self).__init__()
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
        self.feature=MSNet(self.inplanes)
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
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()
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
        ref_pos=self.decisionNet(torch.cat((FeatsL+FeatsR_plus,torch.abs(FeatsL-FeatsR_plus)),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL+FeatsR_minus,torch.abs(FeatsL-FeatsR_minus)),1))
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
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()
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
        ref_pos=self.decisionNet(torch.cat((FeatsL+FeatsR_plus,torch.abs(FeatsL-FeatsR_plus)),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL+FeatsR_minus,torch.abs(FeatsL-FeatsR_minus)),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size(),device='cuda')-OCCLUDED.float(), torch.zeros(x0.size(),device='cuda')), dim=0)
        validation_loss=self.criterion(sample+1e-12, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.firstconv.parameters())
        EarlyFeaturesParams.extend(self.feature.layer1.parameters())
        EarlyFeaturesParams.extend(self.feature.layer2.parameters())
        EarlyFeaturesParams.extend(self.feature.layer3.parameters())
        EarlyFeaturesParams.extend(self.feature.layer4.parameters())
        
        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser2_3_4.parameters())
        MiddleFeauresParams.extend(self.feature.MultiScaleFeatureFuser1_2_3_4.parameters())


        #BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        LastFeatureParams=[]
        LastFeatureParams.extend(self.feature.common.parameters())
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.2},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.4},
            {'params':LastFeatureParams,'lr':self.learning_rate*0.6},
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
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,200],gamma=0.7)
        return [optimizer],[scheduler]  

class MSNetPatch_Decision_LM(LightningModule):
    def __init__(self,Inplanes,true1=1,false1=2,false2=8,NANS=0.0):
        super(MSNetPatch_Decision_LM, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y:1.0 - F.cosine_similarity(x, y),reduction='mean')
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.triplet_loss.margin=0.3
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((1,1,1024,1024),dtype=torch.float32)
        self.model=MSNetPatch(self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,xp,xf=batch
        # Forward
        FeatsL=self.model(x0) # x0 BS, 4 ,7,7
        FeatsR_plus=self.model(xp)
        FeatsR_minus=self.model(xf)
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(ref_pos.size()).cuda(), torch.zeros(ref_neg.size()).cuda()), dim=0)
        training_loss=self.criterion(sample+1e-12, target)
        traing_loss_trip=self.triplet_loss(FeatsL,FeatsR_plus,FeatsR_minus)
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss
    def validation_step(self,batch,batch_idx):
        x0,xp,xf=batch
        # Forward
        FeatsL=self.model(x0) 
        FeatsR_plus=self.model(xp)
        FeatsR_minus=self.model(xf)
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(ref_pos.size()).cuda(), torch.zeros(ref_neg.size()).cuda()), dim=0)
        validation_loss=self.criterion(sample+1e-12, target)
        validation_loss_trip=self.triplet_loss(FeatsL,FeatsR_plus,FeatsR_minus)
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        return validation_loss
    def forward(self,x):
        return self.model(x)
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200],gamma=0.7)
        return [optimizer],[scheduler]

if __name__=="__main__":
    ModelMSPATCH_DECISION=MSNetPatch_Decision_LM(32)
    x0=torch.rand((2,1,56,56))
    x1=torch.rand((2,1,56,56))
    x2=torch.rand((2,1,56,56))
    loss=ModelMSPATCH_DECISION.training_step((x0,x1,x2),0)
    print(loss)
    #x0=torch.rand((1,1,768,768))
    #x1=torch.rand((2,4,7,7))
    #print(model(x0).shape)
    #print(model(x1).shape)
    
     