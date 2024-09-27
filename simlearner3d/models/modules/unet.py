import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from simlearner3d.models.modules.decision_net  import DecisionNetwork

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