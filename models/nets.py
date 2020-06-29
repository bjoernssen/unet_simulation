import torch.nn as nn
import torch_sparse
from torchvision import models
from torch.nn import Conv2d
import torch.nn.functional as F
from utils.net_functions import convrelu
import torch
import torch_geometric


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(EncoderBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvolutionBlock(channel1, channel2, kernel_size, padding),
            ConvolutionBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x):
        big_out = self.block(x)
        poolout = self.maxpool(big_out)
        return big_out, poolout


class DecoderBlock(nn.Module):
    def __init__(self, big_channel, channel1, channel2, kernel_size=(3, 3), padding=1):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            ConvolutionBlock(channel1+big_channel, channel2, kernel_size, padding),
            ConvolutionBlock(channel2, channel2, kernel_size, padding),
            ConvolutionBlock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)  #combining channels of  input from encoder and upsampling input
        x = self.block(x)
        return x


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class UnetTumor(nn.Module):
    def __init__(self, input_shape):
        super(UnetTumor, self).__init__()

        channel, height, width = input_shape

        self.down1 = EncoderBlock(channel, 12, kernel_size=(3, 3))
        self.down2 = EncoderBlock(12, 24, kernel_size=(3, 3))
        self.down3 = EncoderBlock(24, 46, kernel_size=(3, 3))
        self.down4 = EncoderBlock(46, 64, kernel_size=(3, 3))
        self.down5 = EncoderBlock(64, 128, kernel_size=(3, 3))

        self.center = ConvolutionBlock(128, 128, kernel_size=(3, 3), padding=1)

        self.up5 = DecoderBlock(128, 128, 64, kernel_size=(3, 3))
        self.up4 = DecoderBlock(64, 64, 46, kernel_size=(3, 3))
        self.up3 = DecoderBlock(46, 46, 24, kernel_size=(3, 3))
        self.up2 = DecoderBlock(24, 24, 12, kernel_size=(3, 3))
        self.up1 = DecoderBlock(12, 12, 12, kernel_size=(3, 3))
        self.conv = Conv2d(12, 1, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        down1, out = self.down1(x)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)

        up5 = self.up5(out, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        out = self.conv(up1)

        return out
