###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
###########################################################################
"""(1-6)) """
"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import *
__all__ = ['Fast_NEW3', 'get_scnn']


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Fast_NEW3(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN_NEW3, self).__init__()
        self.aux = aux
        #模块1
        self.learning_to_downsample = LearningToDownsample(16, 32, 64)

        dilation_block_1 = [2]
        self.CFP_Block_1 = nn.Sequential()
        for i in range(0, 1):
            self.CFP_Block_1.add_module("CFP_Module_1_" + str(i), CFPModule(64, d=dilation_block_1[i]))

        self.bn_prelu_1 = BNPReLU(64)
        self.bn_prelu_2 = BNPReLU(128)

        #模块2
        #self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.global_feature_extractor = GlobalFeatureExtractor(65, [32, 64, 128], 64, 6, [2, 2, 2])
        #self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 2, 1])

        dilation_block_2 = [4]
        self.CFP_Block_2 = nn.Sequential()
        for i in range(0, 1):
            self.CFP_Block_2.add_module("CFP_Module_2_" + str(i), CFPModule(128, d=dilation_block_2[i]))

        # self.CFP_Block_2_1 = nn.Sequential()
        # for i in range(0, 1):
        #     self.CFP_Block_2_1.add_module("CFP_Module_2_" + str(i), CFPModule(64, d=dilation_block_2[i],level=1))
        #
        # self.CFP_Block_2_2 = nn.Sequential()
        # for i in range(0, 1):
        #     self.CFP_Block_2_2.add_module("CFP_Module_2_" + str(i), CFPModule(64, d=dilation_block_2[i], level=2))

        #模块3
        self.conv = nn.Sequential(
            nn.Conv2d(65, 129, 1),
            nn.BatchNorm2d(129)
        )
        self.feature_fusion = FeatureFusionModule(64, 65, 65)
        # 模块4
        self.classifier = Classifer(129, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

        self.conv1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool2d(3,1,1)

    def forward(self, x):
        #获取输出图片的尺寸
        size = x.size()[2:]
        #print(x.size())  #[18, 3, 512, 512]
        #模块1 下采样提取特征
        higher_res_features = self.learning_to_downsample(x)
        #print(higher_res_features.size())  #[18, 64, 64, 64]
        h1= self.maxpool(higher_res_features)
        bd1 = self.conv1(higher_res_features) + self.conv1(h1)


        #inception模块
        output1 = self.CFP_Block_1(higher_res_features)
        output1 = self.bn_prelu_1(output1)

        output1 = torch.cat([output1, bd1], 1)

        #x0 = F.interpolate(output1, scale_factor=4, mode='bilinear', align_corners=True)
        #全局特征提取
        x = self.global_feature_extractor(output1)
        #print(x.size())
        x1 = self.maxpool(x)
        bd2 = self.conv2(x) + self.conv2(x1)
        #print(x.size())
        #inception模块
        output2 = self.CFP_Block_2(x)
        #print(output2.size())

        output2 = self.bn_prelu_2(output2)
        output2 = torch.cat([output2, bd2], 1)

        #升维65
        output3 = self.conv(output1)
        #print(output3.shape)

        x = self.feature_fusion(output3, output2)
        #print(x.size()) #[18, 128, 64, 64]

        x = self.classifier(x)
        #print(x.size())  #[18, 2, 64, 64]

        #outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        #print(x.shape)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        #return tuple(outputs)
        return x

class _Downsample_Conv(nn.Module):
    """DownSample_Conv"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(_Downsample_Conv, self).__init__()
        self.conv1x1_1 = _ConvBNReLU(in_channels,out_channels,stride=2,kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels,out_channels,stride=2,kernel_size=1)
        #self.conv1x1_3 = _ConvBNReLU(in_channels, out_channels, stride=2, kernel_size=1)

    def forward(self, x):
        x11 = self.conv1x1_1(x)
        x12 = self.conv1x1_2(x)
        x = x11+x12
        return x

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _DSConvSelf(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, ksize=5, stride=1, padding=2 , **kwargs):
        super(_DSConvSelf, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, kernel_size=ksize , stride=stride, padding=padding, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _RDSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, ksize = 3,stride=1,padding=1, **kwargs):
        super(_RDSConv, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, ksize, stride, padding, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = x + self.dwconv(x)
        x = self.pwconv(x)
        return x

class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1,3,2,1)
        self.patch1 = PatchMerging(dw_channels1, dw_channels2)
        self.conv1 = _RDSConv(dw_channels2, dw_channels2,5,1,2)
        self.patch2 = PatchMerging(dw_channels2, out_channels)
        self.conv2 = _RDSConv(out_channels, out_channels,7,1,3)
        # self.conv3 = _DSConvSelf(out_channels, out_channels, 5, 2 ,2)
        # self.conv4 = _DSConvSelf(out_channels, out_channels, 7, 2, 3)
        # self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        # self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = self.patch1(x)
        x = self.conv1(x)
        #print(x.shape)
        x = self.patch2(x)
        x = self.conv2(x)
        # print(x.shape)
        # x1 = self.conv3(x)d
        # x2 = self.conv4(x)
        #print(x2.size())
        # x = self.dsconv1(x)
        # x = self.dsconv2(x)
        return x


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn + nIn//2)
        self.conv1x1_1 = Conv(nIn, nIn // 2, KSize, 1, padding=1, bn_acti=True)

        #padding = (int(d / 4 + 1), 0), dilation = (int(d / 4 + 1), 1),
        self.dconv3x1_1_1= Conv(nIn // 2, nIn // 8, (dkSize, 1), 1,
                                padding=(1, 0), groups=nIn // 8, bn_acti=True)
        self.dconv1x3_1_1 = Conv(nIn // 8, nIn // 8, (1, 3), 1,
                                 padding=(0, 1), groups=nIn // 8, bn_acti=True)
        self.dconv3x1_1_2 = Conv(nIn // 8, nIn // 4, (5, 1), 1,
                                 padding=(2, 0), groups=nIn // 8, bn_acti=True)
        self.dconv1x3_1_2 = Conv(nIn // 4, nIn // 8, (1, 5), 1,
                                 padding=(0, 2), groups=nIn // 8, bn_acti=True)
        self.dconv3x1_1_3 = Conv(nIn // 8, nIn // 4, (7, 1), 1,
                                 padding=(3, 0), groups=nIn // 8,
                                 bn_acti=True)
        self.dconv1x3_1_3 = Conv(nIn // 4, nIn // 4, (1, 7), 1,
                                 padding=(0, 3), groups=nIn // 4,
                                 bn_acti=True)



        self.dconv3x1_2_1 = Conv(nIn // 2, nIn // 8, (3, 1), 1,
                                 padding=(int(d / 4 + 1), 0), dilation=(int(d / 4 + 1), 1), groups=nIn // 8,
                                 bn_acti=True)
        self.dconv1x3_2_1 = Conv(nIn // 8, nIn // 8, (1, 3), 1,
                                 padding=(0, int(d / 4 + 1)), dilation=(1, int(d / 4 + 1)), groups=nIn // 8,
                                 bn_acti=True)

        self.dconv3x1_2_2 = Conv(nIn // 8, nIn // 4, (5, 1), 1,
                                 padding=(int(d / 2 + 1 + int(d/4)), 0), dilation=(int(d / 4 + 1), 1), groups=nIn // 8,
                                 bn_acti=True)
        self.dconv1x3_2_2 = Conv(nIn // 4, nIn // 8, (1, 5), 1,
                                 padding=(0 , int(d / 2 + 1 + int(d/4))), dilation=(1 , int(d / 4 + 1)), groups=nIn // 8,
                                 bn_acti=True)
        self.dconv3x1_2_3 = Conv(nIn // 8, nIn // 4, (7, 1), 1,
                                 padding=(d + 1 + int(d / 4), 0), dilation=(int(d / 4 + 1), 1), groups=nIn // 8,
                                 bn_acti=True)
        self.dconv1x3_2_3 = Conv(nIn // 4, nIn // 4, (1, 7), 1,
                                 padding=(0, d + 1 + int(d / 4)), dilation=(1, int(d / 4 + 1)), groups=nIn // 4,
                                 bn_acti=True)



        self.dconv3x1_3_1 = Conv(nIn // 2, nIn // 8, (3, 1), 1,
                                 padding=(int(d / 2 + 1), 0), dilation=(int(d / 2 + 1), 1), groups=nIn // 8,
                                 bn_acti=True)
        self.dconv1x3_3_1 = Conv(nIn // 8, nIn // 8, (1, 3), 1,
                                 padding=(0, int(d / 2 + 1)), dilation=(1, int(d / 2 + 1)), groups=nIn // 8,
                                 bn_acti=True)

        self.dconv3x1_3_2 = Conv(nIn // 8, nIn // 4, (5, 1), 1,
                                 padding=(int(d + 2) , 0), dilation=(int(d / 2 + 1) , 1), groups=nIn // 8,
                                 bn_acti=True)
        self.dconv1x3_3_2 = Conv(nIn // 4, nIn // 8, (1, 5), 1,
                                 padding=(0, int(d + 2)), dilation=(1, int(d / 2 + 1)), groups=nIn // 8,
                                 bn_acti=True)

        self.dconv3x1_3_3 = Conv(nIn // 8, nIn // 4, (7, 1), 1,
                                 padding=(d + 4 + int(d/4) , 0), dilation=(int(d / 2 + 1) , 1), groups=nIn // 8, bn_acti=True)
        self.dconv1x3_3_3 = Conv(nIn // 4, nIn // 4, (1, 7), 1,
                                 padding=(0, d + 4 + int(d/4)), dilation=(1, int(d / 2 + 1)), groups=nIn // 4, bn_acti=True)

        # self.dconv3x1_3_4 = Conv(nIn // 8, nIn // 4, (dkSize, 1), 1,
        #                          padding=(1 * d + 1, 0), dilation=(d + 1, 1), groups=nIn // 8,
        #                          bn_acti=True)
        # self.dconv1x3_3_4 = Conv(nIn // 8, nIn // 4, (1, dkSize), 1,
        #                          padding=(0, 1 * d + 1), dilation=(1, d + 1), groups=nIn // 8,
        #                          bn_acti=True)

        self.conv1x1 = Conv(nIn + nIn//2 , nIn, 1, 1, padding=0, bn_acti=False)
        #self.SE = SE(nIn//4)

    def forward(self, input):
        #print(input.size())
        inp = self.bn_relu_1(input)
        inpt = self.conv1x1_1(inp)  #[2, 16, 64, 64]

        o1_1 = self.dconv3x1_1_1(inpt)  # [5, 4, 128, 128]
        o1_1 = self.dconv1x3_1_1(o1_1)  # [2, 2, 64, 64]
        o1_2 = self.dconv3x1_1_2(o1_1)
        o1_2 = self.dconv1x3_1_2(o1_2)  #[2, 4, 64, 64]
        o1_3 = self.dconv3x1_1_3(o1_2)  #[2, 4, 64, 64]
        o1_3 = self.dconv1x3_1_3(o1_3)

        #print(o1_4.size(),o1_1.size())
        o2_1 = self.dconv3x1_2_1(inpt)
        o2_1 = self.dconv1x3_2_1(o2_1)
        o2_2 = self.dconv3x1_2_2(o2_1)
        o2_2 = self.dconv1x3_2_2(o2_2)
        o2_3 = self.dconv3x1_2_3(o2_2)
        o2_3 = self.dconv1x3_2_3(o2_3)

        o3_1 = self.dconv3x1_3_1(inpt)
        o3_1 = self.dconv1x3_3_1(o3_1)
        o3_2 = self.dconv3x1_3_2(o3_1)
        o3_2 = self.dconv1x3_3_2(o3_2)
        o3_3 = self.dconv3x1_3_3(o3_2)
        o3_3 = self.dconv1x3_3_3(o3_3)
        # print(o1_1.size(), o1_2.size(), o1_3.size())
        # print(o2_1.size(), o2_2.size(), o2_3.size())
        # print(o3_1.size(), o3_2.size(), o3_3.size())

        output_1 = torch.cat([o1_1, o1_2, o1_3 ], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3 ], 1)
        output_3 = torch.cat([o3_1, o3_2 ,o3_3 ], 1)
       # print(output_1.size()) #[2, 32, 64, 64]

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        output = torch.cat([ad1 , ad2 , ad3], 1)
        #print(output.size())
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        #print(output.size())

        return output + input


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.SE0 = SE(block_channels[0])
        self.SE1 = SE(block_channels[1])
        self.SE2 = SE(block_channels[2])

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.SE0(x)
        # print(x.shape)
        x = self.bottleneck2(x)
        x = self.SE1(x)
        # print(x.size())
        x = self.bottleneck3(x)
        x = self.SE2(x)
        #print(x.shape)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        #lower_res_feature = self.dwconv(lower_res_feature)
        #lower_res_feature = self.conv_lower_res(lower_res_feature)

        #higher_res_feature = self.conv_higher_res(higher_res_feature)
        #print(higher_res_feature.shape)
        #print(lower_res_feature.shape)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.conv1 = _ConvBNReLU(dw_channels, dw_channels, stride)
        self.conv2 = _ConvBNReLU(dw_channels, dw_channels, stride)
        self.conv3 = nn.Conv2d(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(130, num_classes, 1)
        )

        self.conv3 = nn.Conv2d(129, 1, kernel_size=1, stride=1)

    def forward(self, x):

        bd3 = self.conv3(x)
        #print(bd3.shape)
        x = torch.cat([x, bd3], 1)
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        return x


def get_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    #from data_loader import datasets
    model = FastSCNN_NEW3(2, **kwargs)
    if pretrained:
        if(map_cpu):
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 512, 512)
    model = get_scnn('citys')
    outputs = model(img)
    stat(model, (3, 800, 800))