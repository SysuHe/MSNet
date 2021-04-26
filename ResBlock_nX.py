import torch.nn as nn
from CA import CoordAtt

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResBlock_2X(nn.Module):
    def __init__(self, block, layers, in_c=3):
        super(ResBlock_2X, self).__init__()
        self.inplanes = 64
        self.in_c = in_c

        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resblock1 = self._make_layer(block, 64, layers[0], stride=2)
        self.resblock2 = self._make_layer(block, 128, layers[0], stride=2)
        self.resblock3 = self._make_layer(block, 256, layers[0], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        low_lv_f_2X = x
        x = self.resblock1(x)
        x = self.resblock2(x)
        # x = self.resblock3(x)
        # x = self.resblock4(x)
        return x, low_lv_f_2X

class ResBlock_4X(nn.Module):
    def __init__(self, block, layers, in_c=3):
        super(ResBlock_4X, self).__init__()
        self.inplanes = 64
        self.in_c = in_c

        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1 = self._make_layer(block, 64, layers[0], stride=2)
        self.resblock2 = self._make_layer(block, 128, layers[0], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.resblock1(x1)
        x3 = self.resblock2(x2)
        return x3, x2, x1

class ResBlock_8X(nn.Module):
    def __init__(self, block, layers, in_c=3):
        super(ResBlock_8X, self).__init__()
        self.inplanes = 64
        self.in_c = in_c

        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1 = self._make_layer(block, 64, layers[0], stride=2)
        self.resblock2 = self._make_layer(block, 128, layers[1], stride=2)
        self.resblock3 = self._make_layer(block, 256, layers[2], stride=2)
        self.resblock4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.resblock1(x1)
        return x2, x1

def build_resblock(n_multiple=4, in_c=3):
    if n_multiple == 2:
        return ResBlock_2X(BasicBlock, [3, 4, 6, 3], in_c=in_c)
    elif n_multiple == 4:
        return ResBlock_4X(BasicBlock, [3, 4, 6, 3], in_c=in_c)
    elif n_multiple == 8:
        return ResBlock_8X(BasicBlock, [3, 4, 6, 3], in_c=in_c)
    else:
        raise NotImplementedError