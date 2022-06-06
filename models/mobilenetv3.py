import torch
import torch.nn as nn
import torch.nn.functional as F
from models.help_funcs import Dynamic_conv2d, SplitConcat, DWSKConv, AttSC


__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', dilation=1):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel + (kernel - 1) * (dilation - 1) - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        # conv_layer = Dynamic_conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            # conv_layer(exp, exp, kernel, stride=stride, padding=padding, groups=exp, bias=False, dilation=dilation),
            Dynamic_conv2d(exp, exp, kernel, stride=stride, padding=padding, groups=exp, K=8,
                           bias=False, dilation=dilation),
            # SplitConcat(exp, exp, M=4, stride=stride),
            # DWSKConv(exp, exp, M=4, stride=stride),
            # AttSC(exp, exp, M=4, stride=stride),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        assert input_size % 32 == 0
        input_channel = 16
        # last_channel = 1280
        # building first layer
        # last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 1, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s, d
                [[3, 16,  16,  False, 'RE', 1, 1]],
                [[3, 64,  24,  False, 'RE', 2, 1],
                 [3, 72,  24,  False, 'RE', 1, 1]],
                [[5, 72,  40,  True,  'RE', 2, 1],
                 [5, 120, 40,  True,  'RE', 1, 1],
                 [5, 120, 40,  True,  'RE', 1, 1]],
                [[3, 240, 80,  False, 'HS', 2, 1],
                 [3, 200, 80,  False, 'HS', 1, 1],
                 [3, 184, 80,  False, 'HS', 1, 1],
                 [3, 184, 80,  False, 'HS', 1, 1],
                 [3, 480, 112, True,  'HS', 1, 1],
                 [3, 672, 112, True,  'HS', 1, 1]],
                [[5, 672, 160, True,  'HS', 1, 1],
                 [5, 960, 160, True,  'HS', 1, 2],
                 [5, 960, 160, True,  'HS', 1, 2]],
            ]
            self.flags = [False, True, True, True, False, False]  # whether to output the inter-variable
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s, d
                [[3, 16,  16,  True,  'RE', 2, 1]],
                [[3, 72,  24,  False, 'RE', 2, 1],
                 [3, 88,  24,  False, 'RE', 1, 1]],
                [[5, 96,  40,  True,  'HS', 2, 1],
                 [5, 240, 40,  True,  'HS', 1, 1],
                 [5, 240, 40,  True,  'HS', 1, 1],
                 [5, 120, 48,  True,  'HS', 1, 1],
                 [5, 144, 48,  True,  'HS', 1, 1]],
                [[5, 288, 96,  True,  'HS', 1, 1],
                 [5, 576, 96,  True,  'HS', 1, 2],
                 [5, 576, 96,  True,  'HS', 1, 2]],
            ]
            self.flags = [True, True, True, False, False]  # whether to output the inter-variable
        else:
            raise NotImplementedError

        for cfg in mobile_setting:
            layers = []
            for k, exp, c, se, nl, s, d in cfg:
                output_channel = make_divisible(c * width_mult)
                exp_channel = make_divisible(exp * width_mult)
                layers.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl, d))
                input_channel = output_channel
            self.features.append(nn.Sequential(*layers))

        # building last several layers
        # if mode == 'large':
        #     last_conv = make_divisible(960 * width_mult)
        #     self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        #     self.features.append(nn.AdaptiveAvgPool2d(1))
        #     self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        #     self.features.append(Hswish(inplace=True))
        # elif mode == 'small':
        #     last_conv = make_divisible(576 * width_mult)
        #     self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        #     # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        #     self.features.append(nn.AdaptiveAvgPool2d(1))
        #     self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        #     self.features.append(Hswish(inplace=True))
        # else:
        #     raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*list([m for m in self.features]))

        # # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),    # refer to paper section 6
        #     nn.Linear(last_channel, n_class),
        # )

        self._initialize_weights()

    def forward(self, x):
        encoder = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if self.flags[i] == True:
                encoder.append(x)
        # x = x.mean(3).mean(2)
        # x = self.classifier(x)
        return x, encoder

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model


if __name__ == '__main__':
    net = mobilenetv3()
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 3, 224, 224)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile
    flops, params = profile(net, input_size=input_size)
    # print(flops)
    # print(params)
    print('Total params: %.2fM' % (params/1000000.0))
    print('Total flops: %.2fM' % (flops/1000000.0))
    x = torch.randn(input_size)
    out = net(x)



