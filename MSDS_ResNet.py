import torch
import torch.nn as nn

from ResNet import build_backbone
from CA import CoordAtt
from ResBlock_nX import build_resblock

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dconv = Conv1x1(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.dconv(x)

class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.outcovn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.outcovn(x)

class Res_UNet(nn.Module):
    def __init__(self, n_channels, pretrained, n_classes):
        super(Res_UNet, self).__init__()
        self.res_hl_ft = build_backbone('resnet34', pretrained=pretrained, in_c=n_channels)
        self.up1 = Up(1088, 512)
        self.up2 = Up(704, 256)
        self.up3 = Up(320, 128)
        self.up4 = Up(192, 64)
        self.drop = nn.Dropout(0.1)
        self.output = outconv(64, n_classes)

        self.res_hl_ft_4x = build_resblock(4, 3)
        self.res_hl_ft_8x = build_resblock(8, 3)

        self.ca_64 = CoordAtt(64, 64)
        self.ca_128 = CoordAtt(128, 128)
        self.ca_256 = CoordAtt(256, 256)
        self.ca_512 = CoordAtt(512, 512)
        self.ca_1024 = CoordAtt(1024, 1024)
        self.ca_2048 = CoordAtt(2048, 2048)

    def forward(self, input_X, input_4X, input_8X):
        x5, x4, x3, x2, x1 = self.res_hl_ft(input_X)
        x4_5, x4_4, x4_3 = self.res_hl_ft_4x(input_4X)
        x8_5, x8_4 = self.res_hl_ft_8x(input_8X)

        x5, x4, x3, x2, x1 = self.ca_512(x5), self.ca_256(x4), self.ca_128(x3), self.ca_64(x2), self.ca_64(x1)
        x4_5, x4_4, x4_3 = self.ca_128(x4_5), self.ca_64(x4_4), self.ca_64(x4_3)
        x8_5, x8_4 = self.ca_64(x8_5), self.ca_64(x8_4)

        x = self.up1(torch.cat([x5, x4_5, x8_5], dim=1), torch.cat([x4, x4_4, x8_4], dim=1))
        x = self.up2(x, torch.cat([x3, x4_3], dim=1))
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.drop(x)
        logits = self.output(x)
        return logits