from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import os 

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
	
class MSNETInferenceGatedAttention(torch.nn.Module):
	def __init__(self,Inplanes):
		super(MSNETInferenceGatedAttention, self).__init__()
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
		if x.size()[-2] % 8 != 0:
			times = x.size()[-2]//8   
			top_pad = (times+1)*8 - x.size()[-2]
		else:
			top_pad = 0
		if x.size()[-1] % 8 != 0:
			times = x.size()[-1]//8
			right_pad = (times+1)*8-x.size()[-1] 
		else:
			right_pad = 0    
		x = F.pad(x,(0,right_pad, top_pad,0))
		#print("padded size ",x.shape)
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
		if top_pad !=0 and right_pad != 0:
			out = x_all[:,:,top_pad:,:-right_pad]
		elif top_pad ==0 and right_pad != 0:
			out = x_all[:,:,:,:-right_pad]
		elif top_pad !=0 and right_pad == 0:
			out = x_all[:,:,top_pad:,:]
		else:
			out = x_all
		return out
