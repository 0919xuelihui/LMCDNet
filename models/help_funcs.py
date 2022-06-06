import math
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from misc.torchutils import save_DA_vis


class OneLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class OneLayerSepConv2d(nn.Sequential):  # Deep Seperable Convolution
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class TwoLayerSepConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                            padding=0, stride=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, groups=out_channels)
                         )

class Up(nn.Module):
    def __init__(self, factor, in_ch, out_ch=None, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            if out_ch is not None:
                self.up = nn.Sequential(nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True),
                                    OneLayerConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1))
            else:
                self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        else:
            if out_ch is not None:
                self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class FinalClassifier(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1),
                         nn.Conv2d(in_channels // 2, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1),
                         )


##### Dynamic Relu
class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


#####  Modules of STDC Networks:
class DWConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(DWConvX, self).__init__()
        self.dw_conv = nn.Conv2d(in_planes, in_planes, groups=in_planes, kernel_size=kernel,
                                 stride=stride, padding=kernel // 2, bias=False)
        self.con1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_conv(x)
        out = self.relu(self.bn(self.conv1x1(x)))
        return out


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


# Multi-Branch Structure in series
class STDCAdd(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(STDCAdd, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )  # separable convolution
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class STDCConcat(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(STDCConcat, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )  # depth-wise convolution
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class LightRes2Net(nn.Module):
    def __init__(self, inplanes, planes, M=4, stride=1, norm_layer=None):
        super(LightRes2Net, self).__init__()

        self.M = M
        self.width = planes // self.M
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=False)

        if self.M == 1:
            self.nums = 1
        else:
            self.nums = self.M - 1
        if stride != 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):  # depth-wise convolution
            convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=stride, padding=1,
                                   bias=False, groups=self.width))
            bns.append(norm_layer(self.width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):  # stride != 1时不进行迭代卷积操作，每组各自产生结果（不具备多尺度提取能力）
            if i == 0 or self.stride != 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.M != 1 and self.stride == 1:
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.M != 1 and self.stride != 1:
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        return out


class SerialMS(nn.Module):  # Multi-Scale in Series with addition
    def __init__(self, in_planes, out_planes, M, stride=1):
        super(SerialMS, self).__init__()
        # self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)   # 通道缩减
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=out_planes),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False)
            ))

    def forward(self, x):
        # x = self.conv_in(x)
        fea = x
        for i, conv in enumerate(self.convs):
            fea = conv(fea)
            if i == 0:
                feas = torch.unsqueeze(fea, dim=1)
            else:
                feas = torch.cat([feas, torch.unsqueeze(fea, dim=1)], dim=1)
        feas = torch.cat([torch.unsqueeze(x, dim=1), feas], dim=1)
        out = torch.sum(feas, dim=1)
        return out


class SFSerialMS(nn.Module):  # Multi-Scale in Series with addition
    def __init__(self, in_planes, out_planes, M, r=2, stride=1, L=32):
        super(SFSerialMS, self).__init__()
        # self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)   # 通道缩减
        d = max(int(out_planes / r), L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=out_planes),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False)
            ))
        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])
        for i in range(M + 1):
            self.fcs.append(
                nn.Linear(d, out_planes)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.conv_in(x)
        fea = x
        for i, conv in enumerate(self.convs):
            fea = conv(fea)
            if i == 0:
                feas = torch.unsqueeze(fea, dim=1)
            else:
                feas = torch.cat([feas, torch.unsqueeze(fea, dim=1)], dim=1)
        feas = torch.cat([torch.unsqueeze(x, dim=1), feas], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (feas * attention_vectors).sum(dim=1)
        return out


# Multi-Branch Structure in parallel
# class SplitConcat(nn.Module):  # Concatenate output of multi-branch
#     def __init__(self, in_planes, out_planes, M, stride=1):
#         """ Constructor
#         Args:
#             M: the number of branchs.
#             stride: stride, default 1.
#         """
#         super(SplitConcat, self).__init__()
#         mid_planes = out_planes // M
#         self.conv_in = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1)   # 通道缩减
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(mid_planes, out_planes // M, kernel_size=3 + i * 2, stride=stride,
#                           padding=1 + i, groups=out_planes // M),
#                 # nn.Conv2d(mid_planes, out_planes // M, kernel_size=3, stride=stride, dilation=i + 1,
#                 #           padding=i + 1, groups=out_planes // M),  # Atrous Convolution : dilation=i => kernel_size=3+2(i-1)=2i+1
#                 nn.BatchNorm2d(out_planes // M),
#                 nn.ReLU(inplace=False)
#             ))
#
#     def forward(self, x):
#         x = self.conv_in(x)
#         for i, conv in enumerate(self.convs):
#             fea = conv(x)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         out = feas
#         return out


class SplitConcat(nn.Module):  # Add output of multi-branch
    def __init__(self, in_planes, out_planes, M, stride=1):
        """ Constructor
        Args:
            M: the number of branchs.
            stride: stride, default 1.
        """
        super(SplitConcat, self).__init__()
        # self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)   # 通道缩减
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                # nn.Conv2d(out_planes, out_planes, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=out_planes),
                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, dilation=i + 1,
                          padding=i + 1, groups=out_planes),   # Atrous Convolution : dilation=i => kernel_size=3+2(i-1)=2i+1
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False)
            ))

    def forward(self, x):
        # x = self.conv_in(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        # feas = torch.cat([torch.unsqueeze(x, dim=1), feas], dim=1)   # Add the original input
        out = torch.sum(feas, dim=1)
        return out


# class SplitConcat(nn.Module):   # New Version
#     def __init__(self, in_planes, out_planes, M=3, stride=1, dilation=1):
#         """ Constructor
#         Args:
#             M: the number of branchs.
#             stride: stride, default 1.
#         """
#         super(SplitConcat, self).__init__()
#         # mid_planes = out_planes // M
#         # self.conv_in = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1)   # 通道缩减
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes // M, kernel_size=1 + i * 2, stride=stride, padding=i * dilation,
#                           groups=min(in_planes, out_planes // M), dilation=dilation),
#                 nn.BatchNorm2d(out_planes // M),
#                 nn.ReLU(inplace=False)
#             ))
#
#     def forward(self, x):
#         # x = self.conv_in(x)
#         for i, conv in enumerate(self.convs):
#             fea = conv(x)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         out = feas
#         return out


class AttSC(nn.Module):
    def __init__(self, in_planes, out_planes, M=4, dilation=[1, 2, 3, 4, 5], stride=1, r=8):
        super(AttSC, self).__init__()
        self.out_planes = out_planes
        M = len(dilation)
        self.dilation = dilation
        # self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
        # Multi-Scale Part
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                # nn.Conv2d(out_planes, out_planes, kernel_size=3 + i * 2, stride=stride, padding=1 + i,
                #           groups=out_planes),
                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation[i],
                padding=dilation[i], groups=out_planes),  # Atrous Convolution
                # nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation[i],
                # padding=dilation[i]),  # Atrous Convolution
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False)
            ))

        ### ChannelGate
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = OneLayerConv2d(out_planes, out_planes // r, 1)
        self.fc2 = nn.Conv2d(out_planes // r, M * out_planes, 1, 1, 0, bias=False)

        ### SpatialGate
        self.conv_sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # self.fuse = nn.Conv2d(out_planes, out_planes, 1, 1, 0)
        # self.bn_fuse = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        # x = self.conv_in(x)
        brs = [conv(x) for conv in self.convs]
        # brs.append(x)
        gather = sum(brs)  # size:B,C_in,H,W

        ### Channel Attention
        avg_d = self.avgpool(gather)
        avg_d = self.fc2(self.fc1(avg_d))
        max_d = self.maxpool(gather)
        max_d = self.fc2(self.fc1(max_d))
        d = avg_d + max_d
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation), self.out_planes, 1, 1)  # size:B,N,C_in,1,1

        ### Spatial Attention
        avg_out = torch.mean(gather, dim=1, keepdim=True)
        max_out = torch.max(gather, dim=1, keepdim=True)[0]
        s = torch.cat([avg_out, max_out], dim=1)
        s = self.conv_sa(s).unsqueeze(1)  # size:B,1,1,H,W

        ### Fuse two attention tensors
        f = d * s  # size:B,N,C_in,H,W
        f = F.softmax(f, dim=1)
        # save_DA_vis(f)

        return sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation))])


# class DSConv3x3(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, dilation=1):
#         super(DSConv3x3, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
#                       padding=dilation, groups=out_planes),  # Atrous Convolution
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU(inplace=False))
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class AttSC(nn.Module):
#     def __init__(self, in_planes, out_planes, M=4, dilation=[1, 2, 3, 4], stride=1, r=8):
#         super(AttSC, self).__init__()
#         self.out_planes = out_planes
#         M = len(dilation)
#         self.dilation = dilation
#         # self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
#         # Multi-Scale Part
#         self.convs = nn.ModuleList([
#                 DSConv3x3(out_planes, out_planes, stride=stride, dilation=d) for d in dilation
#                 ])
#
#         ### ChannelGate
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = OneLayerConv2d(out_planes, out_planes // r, 1)
#         self.fc2 = nn.Conv2d(out_planes // r, M * out_planes, 1, 1, 0, bias=False)
#
#         ### SpatialGate
#         self.conv_sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#
#         # self.fuse = nn.Conv2d(out_planes, out_planes, 1, 1, 0)
#         # self.bn_fuse = nn.BatchNorm2d(out_planes)
#
#     def forward(self, x):
#         # x = self.conv_in(x)
#         brs = self.convs[0](x)
#         # brs.append(x)
#         # gather = sum(brs)  # size:B,C,H,W
#         #
#         # ### ChannelGate
#         # avg_d = self.avgpool(gather)
#         # avg_d = self.fc2(self.fc1(avg_d))
#         # max_d = self.maxpool(gather)
#         # max_d = self.fc2(self.fc1(max_d))
#         # d = avg_d + max_d
#         # d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation), self.out_planes, 1, 1)  # size:B,N+1,C_in,1,1
#         #
#         # ### SpatialGate
#         # avg_out = torch.mean(gather, dim=1, keepdim=True)
#         # max_out = torch.max(gather, dim=1, keepdim=True)[0]
#         # s = torch.cat([avg_out, max_out], dim=1)
#         # s = self.conv_sa(s).unsqueeze(1)  # size:B,1,1,H,W
#         #
#         # ### Fuse two gates
#         # f = d * s  # size:B,N+1,C_in,H,W
#         # f = F.softmax(f, dim=1)
#
#         return brs


class DWSKConv(nn.Module):  # SplitConcat + SK-Fusion
    def __init__(self, in_planes, out_planes, M, r=2, stride=1, L=32):
        """ Constructor
        Args:
            M: the number of branchs.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(DWSKConv, self).__init__()
        # self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)   # 通道缩减
        d = max(int(out_planes / r), L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                # nn.Conv2d(out_planes, out_planes, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=out_planes),
                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, dilation=i + 1, padding=i + 1,
                          groups=out_planes),  # Atrous Convolution : dilation=i => kernel_size=3+2(i-1)=2i+1
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False)
            ))
        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, out_planes)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.conv_in(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        # feas = torch.cat([torch.unsqueeze(x, dim=1), feas], dim=1)  # Add the original input
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (feas * attention_vectors).sum(dim=1)
        return out


# class DWSKConv(nn.Module):  # New Version
#     def __init__(self, in_planes, out_planes, M, r=2, stride=1, L=32):
#         """ Constructor
#         Args:
#             M: the number of branchs.
#             r: the radio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(DWSKConv, self).__init__()
#         # self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)   # 通道缩减
#         d = max(int(out_planes / r), L)
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=3 + i * 2, stride=stride, padding=1 + i,
#                           groups=min(out_planes, in_planes)),
#                 nn.BatchNorm2d(out_planes),
#                 nn.ReLU(inplace=False)
#             ))
#         self.fc = nn.Linear(out_planes, d)
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(
#                 nn.Linear(d, out_planes)
#             )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # x = self.conv_in(x)
#         for i, conv in enumerate(self.convs):
#             fea = conv(x).unsqueeze_(dim=1)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         fea_U = torch.sum(feas, dim=1)
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector], dim=1)
#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         out = (feas * attention_vectors).sum(dim=1)
#         return out


class DWSKCAM(nn.Module):  # SplitConcat + Channel Attention Tail
    def __init__(self, in_planes, out_planes, M, r=2, stride=1):
        """ Constructor
        Args:
            M: the number of branchs.
            r: the radio for channel dimension reduction
            stride: stride, default 1.
        """
        super(DWSKCAM, self).__init__()
        self.inter_planes = out_planes // M
        self.conv_in = nn.Conv2d(in_planes, self.inter_planes, kernel_size=1, stride=1)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=3 + i * 2, stride=stride,
                          padding=1 + i, groups=self.inter_planes),
                # nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=3, stride=stride, dilation=i+1,
                # padding=1 + i, groups=self.inter_planes),
                nn.BatchNorm2d(self.inter_planes),
                nn.ReLU(inplace=False)
            ))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_planes, out_planes // r, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(out_planes // r, out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_in(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(feas))))
        out = feas * self.sigmoid(avg_out)
        return out


class SCSGE(nn.Module):  # SplitConcat + SGE Module
    def __init__(self, in_planes, out_planes, M=2, stride=1, groups=16):
        """ Constructor
        Args:
            M: the number of branchs.
            stride: stride, default 1.
            groups: groups when calculating attention.
        """
        super(SCSGE, self).__init__()
        self.inter_planes = out_planes // M
        self.conv_in = nn.Conv2d(in_planes, self.inter_planes, kernel_size=1, stride=1)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                # nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=3 + i * 2, stride=stride,
                #           padding=1 + i, groups=self.inter_planes),
                nn.Conv2d(self.inter_planes, self.inter_planes, kernel_size=3, stride=stride, dilation=i+1,
                padding=i+1, groups=self.inter_planes),   # Atrous Convolution : dilation=i => kernel_size=3+2(i-1)=2i+1
                nn.BatchNorm2d(self.inter_planes),
                nn.ReLU(inplace=False)
            ))
        # SGE Module
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_in(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        # SGE Module
        b, c, h, w = feas.size()
        x = feas.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)  # size: b*g, 1, h ,w
        # Normalization
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        out = x * self.sig(t)
        out = out.view(b, c, h, w)
        return out


# class SCSGE(nn.Module):  # New Version
#     def __init__(self, in_planes, out_planes, M=3, stride=1, dilation=1, groups=16):
#         """ Constructor
#         Args:
#             M: the number of branchs.
#             stride: stride, default 1.
#             groups: groups when calculating attention.
#         """
#         super(SCSGE, self).__init__()
#         self.inter_planes = out_planes // M
#         # self.conv_in = nn.Conv2d(in_planes, self.inter_planes, kernel_size=1, stride=1)
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(in_planes, self.inter_planes, kernel_size=1 + i * 2, stride=stride,
#                           padding=i * dilation, groups=self.inter_planes, dilation=dilation),
#                 nn.BatchNorm2d(self.inter_planes),
#                 nn.ReLU(inplace=False)
#             ))
#         # SGE Module
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
#         self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         # x = self.conv_in(x)
#         for i, conv in enumerate(self.convs):
#             fea = conv(x)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         # SGE Module
#         b, c, h, w = feas.size()
#         x = feas.view(b * self.groups, -1, h, w)
#         xn = x * self.avg_pool(x)
#         xn = xn.sum(dim=1, keepdim=True)  # size: b*g, 1, h ,w
#         # Normalization
#         t = xn.view(b * self.groups, -1)
#         t = t - t.mean(dim=1, keepdim=True)
#         std = t.std(dim=1, keepdim=True) + 1e-5
#         t = t / std
#         t = t.view(b, self.groups, h, w)
#         t = t * self.weight + self.bias
#         t = t.view(b * self.groups, 1, h, w)
#         out = x * self.sig(t)
#         out = out.view(b, c, h, w)
#         return out


###### Dynamic Convolution
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, K=4, temperature=31, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:  # 需要进行初始化
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups,
                                                                    self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


####### Blocks for New Lightweight Backbone Based on Deformable Convolution
class PAM(nn.Module):
    """ Lightweight Position attention module"""
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        # out = self.gamma*out + x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.conv1(x))
        out = input * x
        return out


class DConv(nn.Module):
    ''' Deformable Convolution Block
        Type of kernel_size : Tuple '''
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, stride=1,
                 dilation=1, deformable_groups=1):
        super(DConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.init_parameters()

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(input, offset, mask=mask, weight=self.weight, bias=self.bias,
                                             stride=self.stride, padding=self.padding, dilation=self.dilation)


class DConvAtt(nn.Module):
    def __init__(self, in_planes, out_planes, M, r=2, stride=1, L=32):
        super(DConvAtt, self).__init__()
        # Deformable Convolution
        self.dconv = DConv(in_planes, out_planes, kernel_size=[3, 3], stride=stride, padding=[1, 1])
        # Attention
        self.conv_att = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride)
        self.att = SpatialAttention()
        # self.att = PAM(out_planes)
        # Fusion
        d = max(int(out_planes / r), L)
        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(d, out_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        dconv_x = self.dconv(x)
        att_x = self.att(self.conv_att(x))
        dconv_x = dconv_x.unsqueeze_(dim=1)
        att_x = att_x.unsqueeze_(dim=1)
        feas = torch.cat([dconv_x, att_x], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (feas * attention_vectors).sum(dim=1)
        return out


class MultiDConv(nn.Module):
    def __init__(self, in_planes, out_planes, M, r=2, stride=1, L=32):
        super(MultiDConv, self).__init__()
        self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
        d = max(int(out_planes / r), L)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                DConv(out_planes, out_planes, kernel_size=[3 + i * 2, 3 + i * 2],
                      stride=stride, padding=[1 + i, 1 + i]),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False)
            ))
        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, out_planes)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_in(x)
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (feas * attention_vectors).sum(dim=1)
        return out


####  LightBasic & LightBottleneck
class LightBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, conv=STDCConcat, norm_layer=None):
        super(LightBasic, self).__init__()
        self.expansion = 1
        self.stride = stride
        self.downsample = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        blocknum = [4, 4]
        self.conv1 = conv(inplanes, planes, blocknum[0], self.stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes * self.expansion, blocknum[1])
        self.bn2 = norm_layer(planes * self.expansion)
        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, self.stride),
                norm_layer(planes * self.expansion),
            )

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


class LightBasic_SK(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, conv=None, norm_layer=None):
        super(LightBasic_SK, self).__init__()
        self.expansion = 1
        self.stride = stride
        self.downsample = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        branch_num = [2, 2]
        if conv is None:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=self.stride)
        else:
            self.conv1 = conv(inplanes, planes, branch_num[0], stride=self.stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if conv is None:
            self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, padding=1, stride=1)
        else:
            self.conv2 = conv(planes, planes * self.expansion, branch_num[1])
        self.bn2 = norm_layer(planes * self.expansion)
        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, self.stride),
                norm_layer(planes * self.expansion),
            )

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


class LightBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, conv=STDCConcat, norm_layer=None):
        super(LightBottleneck, self).__init__()
        self.expansion = 2
        self.stride = stride
        self.downsample = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0)
        self.bn1 = norm_layer(planes)
        self.blocknum = 4
        self.conv2 = conv(planes, planes, self.blocknum, self.stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, 1, 0)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, self.stride),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class LightBottleneck_SK(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, conv=None, norm_layer=None):
        super(LightBottleneck_SK, self).__init__()
        self.expansion = 2
        self.stride = stride
        self.downsample = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0)
        self.bn1 = norm_layer(planes)
        self.branch_num = 4
        if conv is None:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=self.stride, padding=2, groups=planes)
            # self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=self.stride, padding=2)
        else:
            self.conv2 = conv(planes, planes, self.branch_num, stride=self.stride)
        # self.conv2 = Dynamic_conv2d(planes, planes, kernel_size=3, stride=self.stride, padding=1, groups=planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, 1, 0)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, self.stride),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# New Multi-Scale Block
class MultiScaleBlock(nn.Module):
    '''
    Inspired by CSPBlock
    '''
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, conv=None, norm_layer=None):
        super(MultiScaleBlock, self).__init__()
        self.expansion = 2
        self.branch_num = 2
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)

        # DownSample
        self.conv_down = None
        if self.stride > 1:
            self.conv_down = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=self.stride,
                                       padding=1, groups=inplanes)
            self.bn_down = norm_layer(inplanes)

        # Identity Mapping Branch(Branch 1)
        branch_1 = int(planes)
        self.conv_res = nn.Conv2d(inplanes, branch_1, 1, 1, 0)
        self.bn_res = norm_layer(branch_1)
        # Multi-Scale Branch(Branch 2)
        branch_2 = int(planes)
        self.conv1 = nn.Conv2d(inplanes, branch_2, 1, 1, 0)
        self.bn1 = norm_layer(branch_2)
        self.conv2 = [nn.Conv2d(branch_2, branch_2, kernel_size=3, stride=1, padding=1, groups=branch_2)]
        if conv is None:
            self.conv2.append(nn.Conv2d(branch_2, branch_2, kernel_size=3, stride=1, padding=1, groups=branch_2))
        else:
            self.conv2.append(conv(branch_2, branch_2, self.branch_num, dilation=[1,3,5,7], stride=1))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = norm_layer(branch_2)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, 1, 0)
        self.bn3 = norm_layer(planes * self.expansion)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.relu(self.bn_down(self.conv_down(x)))

        out1 = self.relu(self.bn_res(self.conv_res(x)))
        out2 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out2)))

        out = self.relu(self.bn3(self.conv3(out1 + out2)))

        return out


###### Blocks for Res2Net-based Model
class Bottle2neck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, baseWidth=16, scale=4, norm_layer=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.expansion = 2
        self.stride = stride
        self.downsample = None
        self.scale = scale
        self.width = width
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stride != 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):  # depth-wise convolution
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False, groups=width))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, self.stride),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):  # stride != 1时不进行迭代卷积操作，每组各自产生结果（不具备多尺度提取能力）
            if i == 0 or self.stride != 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stride == 1:
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stride != 1:
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CSBottle2neck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, baseWidth=16, scale=4, norm_layer=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(CSBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.expansion = 2
        self.stride = stride
        self.downsample = None
        self.scale = scale
        self.width = width
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stride != 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False, groups=width))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, self.stride),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x):  # 输入的x为list
        x1 = x[0]
        x2 = x[1]
        residual1 = x1
        residual2 = x2

        out1 = self.conv1(x1)
        out1 = self.relu(self.bn1(out1))
        out2 = self.conv1(x2)
        out2 = self.relu(self.bn1(out2))

        spx1 = torch.split(out1, self.width, 1)
        spx2 = torch.split(out2, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stride != 1:
                sp1 = spx1[i]
                sp2 = spx2[i]
            else:
                sp1 = sp1 + spx2[i]
                sp2 = sp2 + spx1[i]
            sp1 = self.convs[i](sp1)
            sp1 = self.relu(self.bns[i](sp1))
            sp2 = self.convs[i](sp2)
            sp2 = self.relu(self.bns[i](sp2))
            if i == 0:
                out1 = sp1
                out2 = sp2
            else:
                out1 = torch.cat((out1, sp1), 1)
                out2 = torch.cat((out2, sp2), 1)
        if self.scale != 1 and self.stride == 1:
            out1 = torch.cat((out1, spx1[self.nums]), 1)
            out2 = torch.cat((out2, spx2[self.nums]), 1)
        elif self.scale != 1 and self.stride != 1:
            out1 = torch.cat((out1, self.pool(spx1[self.nums])), 1)
            out2 = torch.cat((out2, self.pool(spx2[self.nums])), 1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)
        out2 = self.conv3(out2)
        out2 = self.bn3(out2)

        if self.downsample is not None:
            residual1 = self.downsample(x1)
            residual2 = self.downsample(x2)

        out1 += residual1
        out1 = self.relu(out1)
        out2 += residual2
        out2 = self.relu(out2)

        return [out1, out2]


###### Blocks for Fusion Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAtt_Fusion(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ChannelAtt_Fusion, self).__init__()
        self.co_att = ChannelAttention(in_channels * 2, 8)
        self.conv1x1 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, highfeature, lowfeature):
        x = torch.cat([highfeature, lowfeature], dim=1)
        x = self.co_att(x) * x
        out = self.relu(self.bn(self.conv1x1(x)))
        return out


class ChannelSelect_DecoderLayer(nn.Module):   # inspired by SKUnit
    def __init__(self, in_channels, in_num, r=2, L=32):
        super(ChannelSelect_DecoderLayer, self).__init__()
        d = max(int(in_channels / r), L)
        self.in_channels = in_channels
        self.fc = nn.Linear(in_channels, d)
        self.fcs = nn.ModuleList([])
        for i in range(in_num):
            self.fcs.append(
                nn.Linear(d, in_channels)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3=None):
        fea1 = x1.unsqueeze(dim=1)
        fea2 = x2.unsqueeze(dim=1)
        if x3 is not None:
            fea3 = x3.unsqueeze(dim=1)
            feas = torch.cat([fea1, fea2, fea3], dim=1)
        else:
            feas = torch.cat([fea1, fea2], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)    # most important operation in the module
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (feas * attention_vectors).sum(dim=1)
        return out


####### Modules for Boundary Enhancement
class BoundaryExtraction(nn.Module):
    def __init__(self, pool_size=3):
        super(BoundaryExtraction, self).__init__()
        self.pool_size = pool_size
        self.maxpool = nn.MaxPool2d(self.pool_size, stride=1, padding=self.pool_size // 2)

    def forward(self, inputs):
        return torch.abs(inputs - self.maxpool(inputs))


class SideExtractionLayer(nn.Module):
    def __init__(self, in_channels, factor, kernel_size, padding):
        super(SideExtractionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        # self.up = nn.ConvTranspose2d(1, 1, kernel_size, factor, padding)
        self.up = nn.Upsample(scale_factor=factor)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.up(x)
        return x


####### Modules for Deep Supervision
class DeepSupervision(nn.Module):
    def __init__(self, in_chan, n_classes, up_factor):
        super(DeepSupervision, self).__init__()
        self.up_factor = up_factor
        self.conv = nn.Conv2d(in_chan, n_classes, 1, 1, 0, bias=True)
        if self.up_factor > 1:
            self.upsample = nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.conv(x)
        if self.up_factor > 1:
            out = self.upsample(out)
        return out


class DeepSupervision_Deconv(nn.Module):
    def __init__(self, in_chan, n_classes, factor, kernel_size, padding):
        super(DeepSupervision_Deconv, self).__init__()
        self.factor = factor
        self.conv = nn.Conv2d(in_chan, n_classes, 1, 1, 0, bias=True)
        if self.factor > 1:
            self.upsample = nn.ConvTranspose2d(n_classes, n_classes, kernel_size, factor, padding, groups=n_classes)

    def forward(self, x):
        out = self.conv(x)
        if self.factor > 1:
            out = self.upsample(out)
        return out


####### ASPP Module
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


######  Modules for Self-Attention
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask=None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):  # dim:the dimension of the input sequence
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, out_dim, stride=1):
        super(CrissCrossAttention, self).__init__()
        if in_dim != out_dim or stride > 1:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, groups=min(in_dim, out_dim))
        else:
            self.conv = None
        self.query_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(
                    m_batchsize, width, height, height).permute(0, 2, 1, 3)

        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W)


class SRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # B,heads,N,heads_dim

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # B,C,N -> B,C,H,W
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # B,C,H,W -> B,C,H/sr_ratio,W/sr_ratio -> B,C,H*W/sr_ratio^2 -> B,H*W/sr_ratio^2,C
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2,B,heads,H*W/sr_ratio^2,heads_dim
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2,B,heads,N,heads_dim
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # sr_ratio=1:(B, heads, N, N) | sr_ratio>1:(B,heads,H*W/sr_ratio^2, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B,N,C
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

######  Modules for Global Context Information Extraction
class ConvAtt(nn.Module):  # Based on Spatial Attention and SKNet
    def __init__(self, in_planes, out_planes, r=2, stride=1, L=32):
        super(ConvAtt, self).__init__()
        # Seperable Convolution
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                              groups=min(in_planes, out_planes))
        # Attention
        self.conv_att = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride,
                                  groups=min(in_planes, out_planes))
        self.att = SpatialAttention()
        # Fusion
        d = max(int(out_planes / r), L)
        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(d, out_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv_x = self.conv(x)
        att_x = self.att(self.conv_att(x))
        conv_x = conv_x.unsqueeze_(dim=1)
        att_x = att_x.unsqueeze_(dim=1)
        feas = torch.cat([conv_x, att_x], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (feas * attention_vectors).sum(dim=1)
        return out


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    重叠式PatchEmbed编码
    Args:
         patch_size ：一个patch窗口的宽高为
         stride ：每次窗口滑动的距离
         in_chans: 输入feature map通道数
         embed_dim： 输出token的通道数
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)  # B, embed_dim, H/stride， W/stride
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, Patch_NUM=HW/stride^2, embed_dim
        x = self.norm(x)

        return x, H, W



class SARefine(nn.Module):
    def __init__(self, in_features, out_features, feature_size, stride, num_heads=8, proj_drop=0.0, attn_drop=0.0,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, sr_ratio=1, patch_size=3):
        super(SARefine, self).__init__()

        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(img_size=feature_size, patch_size=patch_size, stride=stride,
                                             in_chans=in_features, embed_dim=out_features)
        # Self Attention
        self.SA = SRAttention(out_features, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        x = self.SA(x, H, W)  # size:B, HW, C
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class CrossAttDec(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0., norm_layer=nn.LayerNorm):
        super(CrossAttDec, self).__init__()
        self.CrossAtt = Cross_Attention(dim, heads=num_heads, dim_head=dim // num_heads, dropout=dropout, softmax=True)
        # self.pos_embedding_decoder = nn.Parameter(torch.randn(1, dim, decoder_pos_size, decoder_pos_size))
        self.norm = norm_layer(dim)

    def forward(self, x, token):  # size for x:b,c,h,w; size for m:b,hw,c
        b, c, h, w = x.shape
        # x = x + self.pos_embedding_decoder
        x = x.flatten(2).transpose(1, 2)  # size: b,hw,c
        x = self.CrossAtt(self.norm(x), self.norm(token))
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x


######  Modules for Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads=heads,
                                                        dim_head=dim_head, dropout=dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x, m, mask=None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
            x = ff(x)
        return x
