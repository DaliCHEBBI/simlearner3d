import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.modules.decision_net import DecisionNetwork
from models.modules.networks_other import init_weights
from models.modules.grid_attention_layer import GridAttentionBlock2D

class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetDsv, self).__init__()
        self.dsv = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        #return F.interpolate(self.dsv(input), size=outSz, mode='bilinear')
        return self.dsv(input)

class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1), is_batchnorm=True):
        super(UnetGridGatingSignal2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
    
class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)

class UNetGatedAttention(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNetGatedAttention, self).__init__()
        features = init_features
        self.nonlocal_mode='concatenation'
        self.attention_dsample=(2,2)
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)
        self.gating = UnetGridGatingSignal2(features * 16, features * 16, kernel_size=(1, 1), is_batchnorm=True)
        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=features * 2, gate_size=features * 4, inter_size=features * 2,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=features * 4, gate_size=features * 8, inter_size=features * 4,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=features * 8, gate_size=features * 16, inter_size=features * 8,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
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
        # deep supervision
        self.dsv4 = UnetDsv(in_size=features*8, out_size=features//2)
        self.dsv3 = UnetDsv(in_size=features*4, out_size=features//2)
        self.dsv2 = UnetDsv(in_size=features*2, out_size=features//2)
        self.dsv1 = nn.Conv2d(in_channels=features, out_channels=features//2, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2)) #128xh/4
        enc4 = self.encoder4(self.pool3(enc3)) #256xh/8

        bottleneck = self.bottleneck(self.pool4(enc4)) #512xh/16
        
        # Added Gating and Attention heads 
        gating = self.gating(bottleneck)  #512xh/16
        g_enc4, att4 = self.attentionblock4(enc4, gating) # 256 x h/8
        dec4 = self.upconv4(bottleneck) # 256 xh/8
        dec4 = torch.cat((g_enc4, dec4), dim=1) # 512xh/8
        dec4 = self.decoder4(dec4) # 256xh/8
        
        g_enc3, att3 = self.attentionblock3(enc3, dec4) # 128 x h/4
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((g_enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3) # 128xh/4

        g_enc2, att2 = self.attentionblock2(enc2, dec3) # 64 x h/2
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((g_enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)# 64 x h/2
        
        dec1 = self.upconv1(dec2) # 32xh
        # Deep Supervision
        dsv4 = F.interpolate(self.dsv4(dec4), size=x.size()[2:], mode='bilinear')
        dsv3 = F.interpolate(self.dsv3(dec3), size=x.size()[2:], mode='bilinear')
        dsv2 = F.interpolate(self.dsv2(dec2), size=x.size()[2:], mode='bilinear')
        dsv1 = self.dsv1(dec1)
        final = torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1)
        output_feature = self.decoder1(final)  #64xh
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


class UNetInferenceGatedAttention(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNetInferenceGatedAttention, self).__init__()
        features = init_features
        self.nonlocal_mode='concatenation'
        self.attention_dsample=(2,2)
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)
        self.gating = UnetGridGatingSignal2(features * 16, features * 16, kernel_size=(1, 1), is_batchnorm=True)
        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=features * 2, gate_size=features * 4, inter_size=features * 2,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=features * 4, gate_size=features * 8, inter_size=features * 4,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=features * 8, gate_size=features * 16, inter_size=features * 8,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
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
        # deep supervision
        self.dsv4 = UnetDsv(in_size=features*8, out_size=features//2)
        self.dsv3 = UnetDsv(in_size=features*4, out_size=features//2)
        self.dsv2 = UnetDsv(in_size=features*2, out_size=features//2)
        self.dsv1 = nn.Conv2d(in_channels=features, out_channels=features//2, kernel_size=1)

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
        enc3 = self.encoder3(self.pool2(enc2)) #128xh/4
        enc4 = self.encoder4(self.pool3(enc3)) #256xh/8

        bottleneck = self.bottleneck(self.pool4(enc4)) #512xh/16
        
        # Added Gating and Attention heads 
        gating = self.gating(bottleneck)  #512xh/16
        g_enc4, att4 = self.attentionblock4(enc4, gating) # 256 x h/8
        dec4 = self.upconv4(bottleneck) # 256 xh/8
        dec4 = torch.cat((g_enc4, dec4), dim=1) # 512xh/8
        dec4 = self.decoder4(dec4) # 256xh/8
        
        g_enc3, att3 = self.attentionblock3(enc3, dec4) # 128 x h/4
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((g_enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3) # 128xh/4

        g_enc2, att2 = self.attentionblock2(enc2, dec3) # 64 x h/2
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((g_enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)# 64 x h/2
        
        dec1 = self.upconv1(dec2) # 32xh
        # Deep Supervision
        dsv4 = F.interpolate(self.dsv4(dec4), size=x.size()[2:], mode='bilinear')
        dsv3 = F.interpolate(self.dsv3(dec3), size=x.size()[2:], mode='bilinear')
        dsv2 = F.interpolate(self.dsv2(dec2), size=x.size()[2:], mode='bilinear')
        dsv1 = self.dsv1(dec1)
        final = torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1)
        output_feature = self.decoder1(final)  #64xh
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

class UNetGatedAttentionWithDecisionNetwork(nn.Module):
     def __init__(self,infeats,outfeats):
         super(UNetGatedAttentionWithDecisionNetwork,self).__init__()
         self.in_features=infeats
         self.out_features=outfeats
         self.feature=UNetInferenceGatedAttention(init_features=self.in_features)
         self.decisionNet=DecisionNetwork(self.out_features)
     def forward(self,x):
         f_all=self.feature(x)
         #print(f_all.shape)
         # shape 2,64,w,h
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return torch.sigmoid(out)