import torch
import torch.nn as nn

__all__ = ['measure_layer_param', 'measure_layer_flops']


def measure_layer_param(model):
    model.eval()

    total_params = sum([param.nelement() for param in model.parameters()])
    print('Params: {:.2f}M'.format(total_params / 1e6))
    return total_params


def measure_layer_flops(model, input_size):
    model.eval()

    list_conv = []
    list_linear = []
    list_bn = []
    list_pooling = []
    list_relu = []

    multiply_adds = True

    def conv_hook(self, input, output):
        batch_size, in_channels, in_height, in_width = input[0].size()
        out_channels, out_height, out_width = output[0].size()

        conv_ops = self.kernel_size[0] * self.kernel_size[1] * (
            self.in_channels / self.groups) * (2 if multiply_adds else 1)
        add_ops = 1 if self.bias is not None else 0

        total_ops = out_channels * (conv_ops + add_ops)
        flops = batch_size * total_ops * out_height * out_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        mul_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        add_ops = self.bias.nelement()

        flops = batch_size * (mul_ops + add_ops)

        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, in_channels, in_height, in_width = input[0].size()
        out_channels, out_height, out_width = output[0].size()

        conv_ops = self.kernel_size * self.kernel_size

        total_ops = out_channels * conv_ops
        flops = batch_size * total_ops * out_height * out_width

        list_pooling.append(flops)

    def relu_hook(self, input, output):
        total_ops = 1
        for i in input[0].size():
            total_ops *= i
        list_relu.append(total_ops)

    def init_hooks(model):
        children = list(model.children())
        if not children:
            if isinstance(model, nn.Conv2d):
                model.register_forward_hook(conv_hook)
            elif isinstance(model, nn.Linear):
                model.register_forward_hook(linear_hook)
            elif isinstance(model, nn.BatchNorm2d) or isinstance(model, nn.BatchNorm1d):
                model.register_forward_hook(bn_hook)
            elif isinstance(model, nn.AvgPool2d) or isinstance(model, nn.MaxPool2d):
                model.register_forward_hook(pooling_hook)
            elif isinstance(model, nn.ReLU):
                model.register_forward_hook(relu_hook)
            return
        for c in children:
            init_hooks(c)

    init_hooks(model)

    input_data = torch.rand(input_size).unsqueeze(0)
    model(input_data)

    total_flops = sum(
        list_conv) + sum(list_linear) + sum(list_bn) + sum(list_pooling) + sum(list_relu)

    print('FLOPs: {:.2f}B'.format(total_flops / 1e9))

    return total_flops
