import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from CA import CoordAtt

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def ResNet18(pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 256, 128, 64, 64
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet18'])
    return model

def ResNet34(pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 64
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model

def ResNet50(pretrained=True, in_c=3):
    """
    output, low_level_feat:
    2048, 256
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model

def ResNet101(pretrained=True, in_c=3):
    """
    output, low_level_feat:
    2048, 256
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model

def ResNet152(pretrained=True, in_c=3):
    """
    output, low_level_feat:
    2048, 256
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model

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

class ResNet(nn.Module):
    def __init__(self, block, layers, in_c=3):
        super(ResNet, self).__init__()
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
        x2 = self.resblock1(x1) # 128
        x3 = self.resblock2(x2) # 64
        x4 = self.resblock3(x3) # 32
        x5 = self.resblock4(x4) # 16
        return x5, x4, x3, x2, x1

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def build_backbone(backbone, pretrained, in_c=3):
    if backbone == 'resnet50':
        return ResNet50(pretrained, in_c=in_c)
    elif backbone == 'resnet34':
        return ResNet34(pretrained, in_c=in_c)
    elif backbone == 'resnet18':
        return ResNet18(pretrained, in_c=in_c)
    elif backbone == 'resnet101':
        return ResNet101(pretrained, in_c=in_c)
    elif backbone == 'resnet152':
        return ResNet152(pretrained, in_c=in_c)
    else:
        raise NotImplementedError