import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d


class VaryingBN1d(BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(
            VaryingBN1d,
            self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine)
        self.lr = Parameter(torch.Tensor(1))


class VaryingBN2d(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(
            VaryingBN2d,
            self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine)
        self.lr = Parameter(torch.Tensor(1))


class vgg_varyingBN(nn.Module):
    def __init__(self, num_classes=10, init_weight=True, cfg='D'):
        super(vgg_varyingBN, self).__init__()
        self.cfg_list = {
            'D': [64, 'D1', 64, 'M', 128, 'D', 128, 'M', 256, 'D', 256, 'D', 256, 'M', 512, 'D', 512, 'D', 512, 'M', 512, 'D', 512, 'D', 512, 'M', 512]
            # 'E': []
        }
        if isinstance(cfg, list):
            config = cfg
        elif cfg in self.cfg_list:
            config = self.cfg_list[cfg]
        else:
            raise KeyError
        self.conv_layers = self.make_layers(config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(config[-3], config[-1]),
            VaryingBN1d(config[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config[-1], num_classes)
            )
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg[:-1]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,
                                        stride=2, ceil_mode=True)]
            elif v == 'D':
                layers += [nn.Dropout(0.4)]
            elif v == 'D1':
                layers += [nn.Dropout(0.3)]
            else:
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=1, bias=True)
                layers += [conv2d, VaryingBN2d(v), nn.ReLU(inplace=True), ]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0,math.sqrt(2./n))
                nn.init.kaiming_normal(m.weight.data)
                # m.bias.data.normal_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, VaryingBN1d) or isinstance(m, VaryingBN2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
                m.lr.data.fill_(1)
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0,0.01)
                nn.init.kaiming_normal(m.weight.data)
                # m.bias.data.normal_()
                m.bias.data.zero_()


if __name__ == '__main__':
    net = vgg_varyingBN()
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)
