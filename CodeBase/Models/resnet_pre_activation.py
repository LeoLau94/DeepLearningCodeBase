import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = [
    'resnet20',
    'resnet32',
    'resnet44',
    'resnet56',
    'resnet164',
    'resnet1001',
    'resnet1202',
    'preactivation_resnet110',
    'preactivation_resnet164',
    'preactivation_resnet1001']


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels,
                     stride=stride, kernel_size=1, bias=True)


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * 4)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu == nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual

        return self.relu(out)


class PreActivationBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, block_channels,
                 out_channels, stride=1, downsample=None):
        super(PreActivationBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, block_channels, stride)
        self.bn2 = nn.BatchNorm2d(block_channels)
        self.conv2 = conv3x3(block_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActivationBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, bottleneck_channels_1,
                 bottleneck_channels_2, out_channels, stride=1,
                 downsample=None):
        super(PreActivationBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv1x1(in_channels, bottleneck_channels_1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels_1)
        self.conv2 = conv3x3(
            bottleneck_channels_1,
            bottleneck_channels_2,
            stride)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels_2)
        self.conv3 = conv1x1(bottleneck_channels_2, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, resolution=32):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.resolution = resolution
        self.conv1 = conv3x3(3, 16)
        self.layer1 = self._make_layer_(block, 16, layers[0])
        self.layer2 = self._make_layer_(block, 32, layers[1], stride=2)
        self.resolution /= 2
        self.layer3 = self._make_layer_(block, 64, layers[2], stirde=2)
        self.resolution /= 2
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(int(self.resolution), stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self._init_params_()

    def _make_layer_(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.in_channels,
                    out_channels *
                    block.expansion,
                    stride),
                nn.BatchNorm2d(
                    out_channels *
                    block.expansion))
        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_params_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreActivation_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10,
                 resolution=32, cfgs=None):
        super(PreActivation_ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.conv_layers = nn.ModuleList()
        self.resolution = resolution
        if cfgs is None:
            self.in_channels = 16
            self.conv_layers.append(self._make_layer_(
                block, 16, 16, 16 * block.expansion, layers[0]))
            self.conv_layers.append(self._make_layer_(
                block, 32, 32, 32 * block.expansion, layers[1], stride=2))
            self.conv_layers.append(self._make_layer_(
                block, 64, 64, 64 * block.expansion, layers[2], stride=2))
            self.bn = nn.BatchNorm2d(64 * block.expansion)
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.conv_layers.append(self._make_layer_cfg(block, cfgs[0]))
            for cfg in cfgs[1:]:
                self.conv_layers.append(
                    self._make_layer_cfg(
                        block, cfg, stride=2))
            self.bn = nn.BatchNorm2d(cfg[-1])
            self.fc = nn.Linear(cfg[-1], num_classes)

        self.avgpool = nn.AvgPool2d(int(self.resolution), stride=1)
        self.relu = nn.ReLU(inplace=True)
        self._init_params_()

    def _make_layer_(self, block, mid_channels_1, mid_channels_2,
                     out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(
                    self.in_channels,
                    out_channels,
                    stride),
                )
        if stride == 2:
            self.resolution /= 2
        layers = []
        layers.append(
            block(
                self.in_channels,
                mid_channels_1,
                mid_channels_2,
                out_channels,
                stride,
                downsample))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    self.in_channels,
                    mid_channels_1,
                    mid_channels_2,
                    out_channels))

        return nn.Sequential(*layers)

    def _make_layer_cfg(self, block, cfg, stride=1):
        downsample = None
        if stride != 1 or cfg[0] != cfg[3]:
            downsample = nn.Sequential(
                conv1x1(
                    cfg[0],
                    cfg[3],
                    stride),
                )
        if stride == 2:
            self.resolution /= 2
        layers = []
        layers.append(
            block(
                cfg[0],
                cfg[1],
                cfg[2],
                cfg[3],
                stride,
                downsample))
        cfg_len = len(cfg)
        for i in range(3, cfg_len - 3, 3):
            layers.append(block(cfg[i], cfg[i + 1], cfg[i + 2], cfg[i + 3]))

        return nn.Sequential(*layers)

    def _init_params_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        for l in self.conv_layers:
            x = l(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20(**kwargs):
    model = ResNet(Block, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    model = ResNet(Block, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    model = ResNet(Block, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    model = ResNet(Block, [9, 9, 9], **kwargs)
    return model


def resnet1202(**kwargs):
    model = ResNet(Block, [200, 200, 200], **kwargs)
    return model


def resnet164(**kwargs):
    model = ResNet(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001(**kwargs):
    model = ResNet(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preactivation_resnet110(**kwargs):
    model = PreActivation_ResNet(PreActivationBlock, [18, 18, 18], **kwargs)
    return model


def preactivation_resnet164(**kwargs):
    model = PreActivation_ResNet(
        PreActivationBottleneck, [
            18, 18, 18], **kwargs)
    return model


def preactivation_resnet1001(**kwargs):
    model = PreActivation_ResNet(
        PreActivationBottleneck, [
            111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = preactivation_resnet164(resolution=64)
    print(net)
    x = Variable(torch.randn(1, 3, 64, 64))
    y = net(x)
    print(y.size())
