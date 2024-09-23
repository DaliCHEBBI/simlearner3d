import torch
import torch.nn as nn
import torchvision
from models.modules.decision_net import DecisionNetwork
import torch.nn.functional as F
resnet = torchvision.models.resnet50(pretrained=True)
mm=nn.Conv2d(1, 64, kernel_size=7, stride=2,padding=3,bias=False)
kern=list(resnet.children())[0].weight[:,0,:,:].unsqueeze(1)

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6
    def __init__(self):
        super().__init__()
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        with torch.no_grad():
            self.input_block[0]=mm
            self.input_block[0].weight.copy_(kern)
        # Reactivate gradient computation
        self.input_block[0].weight.requires_grad=True
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 1, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x, with_output_feature_map=True):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        del pre_pools
        return x

class UNetWithResnet50EncoderInf(nn.Module):
    def __init__(self):
        super().__init__()
        self.DEPTH = 6
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        with torch.no_grad():
            self.input_block[0]=mm
            self.input_block[0].weight.copy_(kern)
        # Reactivate gradient computation
        self.input_block[0].weight.requires_grad=True
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 1, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        #self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        if x.size()[-2] % 32 != 0:
            times = x.size()[-2]//32   
            top_pad = (times+1)*32 - x.size()[-2]
        else:
            top_pad = 0
        if x.size()[-1] % 32 != 0:
            times = x.size()[-1]//32
            right_pad = (times+1)*32-x.size()[-1] 
        else:
            right_pad = 0    

        x = F.pad(x,(0,right_pad, top_pad,0))
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i != (self.DEPTH - 1):
                pre_pools[f"layer_{i}"] = x
            #pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        del pre_pools
        if top_pad !=0 and right_pad != 0:
            out = x[:,:,top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            out = x[:,:,:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            out = x[:,:,top_pad:,:]
        else:
            out = x
        return out

class UNETRESNET50WithDecisionNetwork(nn.Module):
     def __init__(self,outfeats):
         super(UNETRESNET50WithDecisionNetwork,self).__init__()
         self.out_features=outfeats
         self.feature=UNetWithResnet50EncoderInf()
         self.decisionNet=DecisionNetwork(self.out_features)
     def forward(self,x):
         f_all=self.feature(x)
         #print(f_all.shape)
         # shape 2,64,w,h
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return torch.sigmoid(out)