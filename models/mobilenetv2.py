"""
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import torch
import torch.nn as nn
import math
from models.help_funcs import Dynamic_conv2d, SplitConcat, DWSKConv, AttSC


__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation, groups=hidden_dim, bias=False, dilation=dilation),
                Dynamic_conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=dilation, groups=hidden_dim,
                               K=8, bias=False, dilation=dilation),
                # SplitConcat(hidden_dim, hidden_dim, M=4, stride=stride),
                # DWSKConv(hidden_dim, hidden_dim, M=4, stride=stride),
                # AttSC(hidden_dim, hidden_dim, M=4, stride=stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation, groups=hidden_dim, bias=False, dilation=dilation),
                Dynamic_conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=dilation, groups=hidden_dim,
                               K=8, bias=False, dilation=dilation),
                # SplitConcat(hidden_dim, hidden_dim, M=4, stride=stride),
                # DWSKConv(hidden_dim, hidden_dim, M=4, stride=stride),
                # AttSC(hidden_dim, hidden_dim, M=4, stride=stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        self.out_channel = []
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.conv_in = conv_3x3_bn(3, input_channel, 1)
        # layers = [conv_3x3_bn(6, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        self.features = []
        self.flags = [True, True, True, False, False]  # whether to output the inter-variable
        for cfg in self.cfgs:
            layers = []
            for t, c, n, s, d in cfg:
                output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
                for i in range(n):
                    layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, d))
                    input_channel = output_channel
            self.out_channel.append(output_channel)
            self.features.append(nn.Sequential(*layers))
        self.features = nn.Sequential(*list([m for m in self.features]))
        # self.out_num = [2, 3]   # output the results of these layers
        # self.features = nn.Sequential(*layers)
        # building last several layers
        # output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        # self.conv = conv_1x1_bn(input_channel, output_channel)
        # del self.out_channel[-1]
        # self.out_channel.append(output_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(output_channel, num_classes)

        # self._initialize_weights()

    def forward(self, x):
        encoder = []
        x = self.conv_in(x)
        for i in range(len(self.features)):
            x = self.features[i](x)
            if self.flags[i] == True:
                encoder.append(x)
        # x = self.conv(x)
        # blocks.append(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x, encoder

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()

def mobilenetv2(pretrained, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    cfgs = [
        # t, c, n, s, dilation
        [[1, 16, 1, 1, 1]],
        [[6, 24, 2, 2, 1]],
        [[6, 32, 3, 2, 1]],
        [[6, 64, 4, 2, 1],
         [6, 96, 3, 1, 1]],
        [[6, 160, 1, 1, 1],
         [6, 160, 2, 1, 2],
         [6, 320, 1, 1, 2]],
    ]
    model = MobileNetV2(cfgs, **kwargs)
    ## 加载部分模型参数
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load('./state_dict_mobilenetv2.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
