import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import tqdm
import thop
from ptflops import get_model_complexity_info

import functools
from einops import rearrange

import models
from models.Siam_concat import *
from models.Siam_diff import *
from models.help_funcs import STDCAdd, STDCConcat, SerialMS, SFSerialMS, LightRes2Net, \
                                DWSKConv, SplitConcat, SCSGE, DWSKCAM, DConvAtt, MultiDConv, \
                                LightBasic, LightBottleneck, LightBasic_SK, LightBottleneck_SK, Bottle2neck, CSBottle2neck, \
                                ChannelAttention, ChannelAtt_Fusion, SpatialAttention, ASPP, Transformer, TransformerDecoder, \
                                TwoLayerConv2d, TwoLayerSepConv2d, OneLayerConv2d, OneLayerSepConv2d, FinalClassifier, Up, \
                                ChannelSelect_DecoderLayer, DeepSupervision, DeepSupervision_Deconv, SideExtractionLayer, BoundaryExtraction, \
                                AttSC, MultiScaleBlock, DyReLUB, SARefine, CrossAttDec, CrissCrossAttention
from misc.torchutils import save_vis_features

###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function, m is the module in the network
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:  # 单GPU或多GPU训练
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if args.net_G == 'basenet_s3':
    #     net = BaseNet(input_nc=3, output_nc=2, output_sigmoid=False, stages_num=3)
    #
    # elif args.net_G == 'basenet_s4':
    #     net = BaseNet(input_nc=3, output_nc=2, output_sigmoid=False, stages_num=4)
    #
    # elif args.net_G == 'Skipbasenet_s4':
    #     net = SkipBaseNet(input_nc=3, output_nc=2, output_sigmoid=False, stages_num=4)
    #
    # elif args.net_G == 'LightBiT_pos_s3':
    #     net = LightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=3, with_pos='learned')
    #
    # elif args.net_G == 'LightBiT_pos_s4':
    #     net = LightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=4, with_pos='learned')
    #
    # elif args.net_G == 'LightBiT_pos_s3_dd8':
    #     net = LightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=3,
    #                    with_pos='learned', enc_depth=1, dec_depth=8)
    #
    # elif args.net_G == 'LightBiT_pos_s4_dd8':
    #     net = LightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=4,
    #                    with_pos='learned', enc_depth=1, dec_depth=8)
    #
    # elif args.net_G == 'LightBiT_pos_s3_dd8_dedim8':
    #     net = LightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=3,
    #                    with_pos='learned', enc_depth=1, dec_depth=8, dim_head=8, decoder_dim_head=8)
    #
    # elif args.net_G == 'LightBiT_pos_s4_dd8_dedim8':
    #     net = LightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=4,
    #                    with_pos='learned', enc_depth=1, dec_depth=8, dim_head=8, decoder_dim_head=8)
    #
    # elif args.net_G == 'SkipLightBiT_pos_s4_dd8':
    #     net = SkipLightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=4,
    #                        with_pos='learned', enc_depth=1, dec_depth=8)
    #
    # elif args.net_G == 'SkipLightBiT_pos_s4_dd8_dedim8':
    #     net = SkipLightBiT(input_nc=3, output_nc=2, token_len=4, stages_num=4,
    #                        with_pos='learned', enc_depth=1, dec_depth=8, dim_head=8, decoder_dim_head=8)
    #
    if args.net_G == 'SkipResNet18_s4':
        net = SkipResNet(input_nc=3, output_nc=2, backbone='resnet18', output_sigmoid=False)

    elif args.net_G == 'SkipResNet18_MSOF_s4':
        net = SkipResNet_MSOF(input_nc=3, output_nc=2, backbone='resnet18', output_sigmoid=False)

    elif args.net_G == 'SkipResNet50_MSOF_s4':
        net = SkipResNet_MSOF(input_nc=3, output_nc=2, backbone='resnet50', output_sigmoid=False)

    elif args.net_G == 'SkipMobileNet_s4':
        net = SkipLightNet(input_nc=3, output_nc=2, backbone='mobilenetv2', output_sigmoid=False)

    elif args.net_G == 'SkipMobileNetv2_MSOF_s4':
        net = SkipLightNet_MSOF(input_nc=3, output_nc=2, backbone='mobilenetv2', output_sigmoid=False)

    elif args.net_G == 'SkipMobileNetv3_MSOF_s4':
        net = SkipLightNet_MSOF(input_nc=3, output_nc=2, backbone='mobilenetv3', output_sigmoid=False)

    elif args.net_G == 'SkipGhostNet_s4':
        net = SkipLightNet(input_nc=3, output_nc=2, backbone='ghostnet', output_sigmoid=False)

    elif args.net_G == 'SkipGhostNet_MSOF_s4':
        net = SkipLightNet_MSOF(input_nc=3, output_nc=2, backbone='ghostnet', output_sigmoid=False)

    elif args.net_G == 'SkipShuffleNetv2_MSOF_s4':
        net = SkipLightNet_MSOF(input_nc=3, output_nc=2, backbone='shufflenetv2', output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_s4':
        net = LightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          block=LightBasic, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes50_s4':
        net = LightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          block=LightBottleneck, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_DConvAttbasic_s4':
        net = LightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          block=LightBasic_SK, output_sigmoid=False)

    # elif args.net_G == 'SkipLightRes18_DConvAttbottle_s4':
    #     net = LightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
    #                       block=LightBottleneck_SK, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_MultiDConvbasic_s4':
    #     znet = LightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
    #                       block=LightBasic_SK, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_MultiDConvbottle_s4':
    #     net = LightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
    #                       block=LightBottleneck_SK, output_sigmoid=False)

    # elif args.net_G == 'SkipLightRes18_SACbasic_s4':
    #     net = LightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       block=LightBasic_SK, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_SACbasic_New_s4':
    #     net = LightResNet18_New(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 2],
    #                       type=[DWSKConv, DWSKConv, DConvAtt], block=LightBasic_SK, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_SACbasic_DiffFPN_s4':
    #     net = LightResNet18_DiffFPN(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       block=LightBasic_SK, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_SACbasic_CSDiffFPN_s4':
    #     net = LightResNet18_CSDiffFPN(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       block=LightBasic_SK, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_SACbasic_deepsup_s4':
    #     net = LightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
    #                       block=LightBasic_SK, output_sigmoid=False, deepsup=True)

    # elif args.net_G == 'SkipLightRes18_SACbasic_attfusion_s4':
    #     net = LightResNet18_attfusion(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
    #                       block=LightBasic_SK, output_sigmoid=False)

    # elif args.net_G == 'SkipLightRes18_SACbasic_s4_bdenhance':
    #     net = LightResNet18_BE(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       block=LightBasic_SK, bdenhance=True)

    elif args.net_G == 'SkipLightRes18_SACbottle_s4':
        net = LightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_SACbottle_s4_SM':
        net = LightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_SACbottle_s4_bdenhance':
        net = LightResNet50_BE(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          block=LightBottleneck_SK, bdenhance=True)

    elif args.net_G == 'SkipLightRes18_SACbottle_simpledec_s4':
        net = LightResNet50_simpledec(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_SACbottle_MSOF_s4':
        net = LightResNet50_MSOF(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_SACbottle_MSOF_s4_1':
        net = LightResNet50_MSOF_1(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_SACbottle_MSOF_s4_2':
        net = LightResNet50_MSOF_2(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_SACbottle_MSOF_s4_new':
        net = LightResNet50_MSOF_new(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                                     block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'SkipLightRes18_SACbottle_MSOF_s4_bd':
        net = LightResNet50_MSOF_BE(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                                    block=LightBottleneck_SK)
    # elif args.net_G == 'SkipLightRes18_SACbottle_MSOF_s4_Recursive':
    #     net = LightResNet50_MSOF_Recursive(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                                        block=LightBottleneck_SK, output_sigmoid=False)

    # elif args.net_G == 'SkipLightRes18_SACbottle_New_s4':
    #     net = LightResNet50_New(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       type=[SCSGE, SCSGE, SCSGE], block=LightBottleneck_SK, output_sigmoid=False)

    # elif args.net_G == 'SkipLightRes18_SACbottle_attfusion_s4':
    #     net = LightResNet50_attfusion(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       block=LightBottleneck_SK, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_attfusion_s4':
    #     net = LightResNet18_attfusion(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
    #                       deepsup=False, block=LightBasic, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_attfusion_222_s4':
    #     net = LightResNet18_attfusion(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
    #                       block=LightBasic, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes50_attfusion_s4':
    #     net = LightResNet50_attfusion(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
    #                       block=LightBottleneck, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_att_deepsup_s4':
    #     net = LightResNet18_attfusion(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
    #                       deepsup=True, block=LightBasic, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes50_att_deepsup_s4':
    #     net = LightResNet50_attfusion(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
    #                       deepsup=True, block=LightBottleneck, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes18_doubleatt_s4':
    #     net = LightResNet18_doubleatt(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
    #                       block=LightBasic, deepsup=False, output_sigmoid=False)
    #
    # elif args.net_G == 'SkipLightRes50_doubleatt_s4':
    #     net = LightResNet50_doubleatt(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
    #                       block=LightBottleneck, deepsup=False, output_sigmoid=False)

    elif args.net_G == 'EFLightRes18_s4':
        net = EFLightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
                          deepsup=False, block=LightBasic, output_sigmoid=False)

    elif args.net_G == 'EFLightRes18_SACbasic_s4':
        net = EFLightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          deepsup=False, block=LightBasic_SK, output_sigmoid=False)

    elif args.net_G == 'EFLightRes18_SACbottle_s4':
        net = EFLightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          deepsup=False, block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'EFLightRes18_SACbottle_simpledec_s4':
        net = EFLightResNet50_simpledec(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'EFLightRes18_SACbottle_MSOF_s4':
        net = EFLightResNet50_MSOF(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'EFLightRes18_SACbottle_MSOF_s4_1':
        net = EFLightResNet50_MSOF_1(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          block=LightBottleneck_SK, output_sigmoid=False)

    elif args.net_G == 'EFLightRes18_SACbottle_MSOF_s4_2':
        net = EFLightResNet50_MSOF_2(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          block=LightBottleneck_SK, output_sigmoid=False)

    # elif args.net_G == 'EFLightRes18_SACbottle_simple':
    #     net = EFLightResNet50_simple(input_nc=3, output_nc=2, stride=[1, 2], blocks=[4, 4],
    #                       deepsup=False, block=LightBottleneck, output_sigmoid=False)

    elif args.net_G == 'LightRes2Net18_SACbottle_s4':
        net = LightRes2Net50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
                          deepsup=False, block=Bottle2neck, output_sigmoid=False)

    elif args.net_G == 'LightRes2Net18_SACbottle_MSOF_s4':
        net = LightRes2Net50_MSOF(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                                  block=Bottle2neck, output_sigmoid=False)

    elif args.net_G == 'CSLightRes2Net18_SACbottle_s4':
        net = CSLightRes2Net50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
                          deepsup=False, block=CSBottle2neck, output_sigmoid=False)

    elif args.net_G == 'Siam_concat':
        net = SiamUnet_conc(3, 2)

    elif args.net_G == 'Siam_diff':
        net = SiamUnet_diff(3, 2)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################

class SkipBaseNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, stages_num=3, block=STDCConcat,
                 norm_layer=None, output_sigmoid=False, if_upsample_2x=True):
        super(SkipBaseNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.stages_num = stages_num
        self.if_upsample_2x = if_upsample_2x

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        blknum = [4, 4, 4]
        stride = [1, 2, 2]
        self.layer1 = self._make_layer(block, 64, 64, 1, blknum[0], stride[0])
        self.layer2 = self._make_layer(block, 64, 128, 1, blknum[1], stride[1])
        if self.stages_num > 3:
            self.layer3 = self._make_layer(block, 128, 256, 1, blknum[2], stride[2])
            # self.aspp = ASPP(512, 256)
        else:
            pass
            # self.aspp = ASPP(256, 128)

        # self.decoder_layer_5 = OneLayerConv2d(in_channels=512, out_channels=256)
        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, inplanes, planes, blocks, blocknum, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(block(inplanes, planes, blocknum, stride))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 256
        y2 = self.decoder_layer_4(y1)  # dim = 128
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = torch.cat([y2, encoder[-1]], dim=1)  # dim = 256
        y4 = self.decoder_layer_3(y3)  # dim = 64
        y4 = self.upsamplex2(y4)

        # decoder layer 2
        y5 = torch.cat([y4, encoder[-2]], dim=1)  # dim = 128
        y6 = self.decoder_layer_2(y5)  # dim = 64
        y6 = self.upsamplex2(y6)

        # decoder layer 1
        y7 = torch.cat([y6, encoder[-3]], dim=1)  # dim = 128
        y8 = self.decoder_layer_1(y7)  # dim =32
        # y8 = self.upsamplex2(y8)

        # final predict
        y_out = self.classifier(y8)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        out = x_5
        if self.stages_num > 3:
            encoder.append(x_5)   # out = 128
            x_6 = self.layer3(x_5)
            out = x_6
        # if self.stages_num == 5:
        #     encoder.append(x_7)
        #     x_8 = self.layer4(x_7)  # in=256, out=512
        #     out = x_8
        # elif self.stages_num > 5:
        #     raise NotImplementedError
        # out = self.conv_pred(out)
        return out, encoder

        ## Decoder Concat
        # encoder1, x1 = self.forward_single(x1)
        # encoder2, x2 = self.forward_single(x2)
        # encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
        #
        # # decoder layer 4
        # y1 = torch.cat([x1, x2], dim=1)  # dim = 512
        # y2 = self.decoder_layer_4(y1)  # dim = 128
        # y2 = self.upsamplex2(y2)
        #
        # # decoder layer 3
        # y3 = torch.cat([y2, en1[-1], en2[-1]], dim=1)  # dim = 384
        # y4 = self.decoder_layer_3(y3)  # dim = 64
        # y4 = self.upsamplex2(y4)
        #
        # # decoder layer 2
        # y5 = torch.cat([y4, en1[-2], en2[-2]], dim=1)  # dim = 192
        # y6 = self.decoder_layer_2(y5)  # dim = 64
        # y6 = self.upsamplex2(y6)
        #
        # # decoder layer 1
        # y7 = torch.cat([y6, en1[-3], en2[-3]], dim=1)  # dim = 192
        # y8 = self.decoder_layer_1(y7)  # dim =32
        # # y8 = self.upsamplex2(y8)
        #
        # # final predict
        # y_out = self.classifier(y8)
        # # sigmoid operation
        # if self.output_sigmoid:
        #     y_out = self.sigmoid(y_out)
        #
        # return y_out


class SkipLightBiT(SkipBaseNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, stages_num=3,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(SkipLightBiT, self).__init__(input_nc, output_nc, stages_num=stages_num,
                                           block=STDCConcat, if_upsample_2x=if_upsample_2x)
        self.conv_squeeze = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.conv_expand = nn.Conv2d(32, 256, kernel_size=3, padding=1)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256 // 8  # the size for the output of the CNN Backbone
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32, decoder_pos_size, decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8, dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth, heads=8,
                                                      dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=decoder_softmax)

        self.dconv4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.dconv3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.dconv2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.dconv1 = OneLayerConv2d(in_channels=128, out_channels=32)

    def _forward_semantic_tokens(self, x):  # process for tokenizer
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)  # size: b,L,h,w
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()  # size:b,L,hw
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  # size:b,c,hw
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):  # size for x:b,c,h,w; size for m:b,l,c(output of TE)
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
        x1 = self.conv_squeeze(x1)
        x2 = self.conv_squeeze(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        x1 = self.conv_expand(x1)
        x2 = self.conv_expand(x2)
        x = torch.abs(x1 - x2)    # out : 256
        x = self.dconv4(x)    # out : 128
        x = self.upsamplex2(x)
        #######  Decoder #######
        x = torch.cat([x, encoder[-1]], dim=1)   # out : 256
        x = self.dconv3(x)  # out : 64
        x = self.upsamplex2(x)
        x = torch.cat([x, encoder[-2]], dim=1)  # out : 128
        x = self.dconv2(x)  # out : 64
        x = self.upsamplex2(x)
        x = torch.cat([x, encoder[-3]], dim=1)  # out : 128
        x = self.dconv1(x)  # out : 32
        # x = self.upsamplex2(x)
        # forward small cnn
        x = self.classifier(x)
        # output to (0,1)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x


class SkipResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, backbone='resnet18', output_sigmoid=False):
        super(SkipResNet, self).__init__()
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=False,
                                          replace_stride_with_dilation=[False, False, True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=False,
                                          replace_stride_with_dilation=[False, False, True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=False,
                                          replace_stride_with_dilation=[False, False, True])
        else:
            raise NotImplementedError
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 256
        y2 = self.decoder_layer_4(y1)  # dim = 128
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = torch.cat([y2, encoder[-1]], dim=1)  # dim = 256
        y4 = self.decoder_layer_3(y3)  # dim = 64
        y4 = self.upsamplex2(y4)

        # decoder layer 2
        y5 = torch.cat([y4, encoder[-2]], dim=1)  # dim = 128
        y6 = self.decoder_layer_2(y5)  # dim = 64
        y6 = self.upsamplex2(y6)

        # decoder layer 1
        y7 = torch.cat([y6, encoder[-3]], dim=1)  # dim = 128
        y8 = self.decoder_layer_1(y7)  # dim =32

        # final predict
        y_out = self.classifier(y8)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.resnet.conv1(x)
        x_2 = self.resnet.bn1(x_1)
        x_2 = self.resnet.relu(x_2)
        encoder.append(x_2)
        x_3 = self.resnet.maxpool(x_2)
        x_4 = self.resnet.layer1(x_3)
        encoder.append(x_4)
        x_5 = self.resnet.layer2(x_4)
        encoder.append(x_5)
        x_6 = self.resnet.layer3(x_5)
        x_7 = self.resnet.layer4(x_6)
        out = x_7
        return out, encoder


class SkipResNet_MSOF(torch.nn.Module):
    def __init__(self, input_nc, output_nc, backbone='resnet18', output_sigmoid=False):
        super(SkipResNet_MSOF, self).__init__()
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=False,
                                          replace_stride_with_dilation=[False, False, True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=False,
                                          replace_stride_with_dilation=[False, False, True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=False,
                                          replace_stride_with_dilation=[False, False, True])
        else:
            raise NotImplementedError

        # MSOF Decoder
        inter_depth = 32
        if backbone == 'resnet18':
            self.head_3 = DeepSupervision(in_chan=512, n_classes=inter_depth, up_factor=8)
            self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
            self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
            self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)
        elif backbone == 'resnet50':
            self.head_3 = DeepSupervision(in_chan=2048, n_classes=inter_depth, up_factor=8)
            self.head_2 = DeepSupervision(in_chan=512, n_classes=inter_depth, up_factor=4)
            self.head_1 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=2)
            self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.resnet.conv1(x)
        x_2 = self.resnet.bn1(x_1)
        x_2 = self.resnet.relu(x_2)
        encoder.append(x_2)
        x_3 = self.resnet.maxpool(x_2)
        x_4 = self.resnet.layer1(x_3)
        encoder.append(x_4)
        x_5 = self.resnet.layer2(x_4)
        encoder.append(x_5)
        x_6 = self.resnet.layer3(x_5)
        x_7 = self.resnet.layer4(x_6)
        out = x_7
        return out, encoder



class SkipLightNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, backbone='mobilenetv2', output_sigmoid=False):
        super(SkipLightNet, self).__init__()
        if backbone == 'mobilenetv2':
            self.backbone = models.mobilenetv2(pretrained=False)
        elif backbone == 'mobilenetv3':
            self.backbone = models.mobilenetv3(pretrained=False, input_size=256, mode='small')
        elif backbone == 'ghostnet':
            self.backbone = models.ghostnet(pretrained=False)
        else:
            raise NotImplementedError
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        # For Mobilenetv2 : 16,24,32,320 ; For Ghostnet : 16,24,40,160
        if backbone == 'mobilenetv2':
            self.decoder_layer_4 = OneLayerConv2d(in_channels=320, out_channels=32)
            self.decoder_layer_3 = OneLayerConv2d(in_channels=64, out_channels=24)
            self.decoder_layer_2 = OneLayerConv2d(in_channels=48, out_channels=16)
            self.decoder_layer_1 = OneLayerConv2d(in_channels=32, out_channels=32)
        elif backbone == 'mobilenetv3':
            self.decoder_layer_4 = OneLayerConv2d(in_channels=96, out_channels=24)
            self.decoder_layer_3 = OneLayerConv2d(in_channels=48, out_channels=16)
            self.decoder_layer_2 = OneLayerConv2d(in_channels=32, out_channels=16)
            self.decoder_layer_1 = OneLayerConv2d(in_channels=32, out_channels=32)
        elif backbone == 'ghostnet':
            self.decoder_layer_4 = OneLayerConv2d(in_channels=160, out_channels=40)
            self.decoder_layer_3 = OneLayerConv2d(in_channels=80, out_channels=24)
            self.decoder_layer_2 = OneLayerConv2d(in_channels=48, out_channels=16)
            self.decoder_layer_1 = OneLayerConv2d(in_channels=32, out_channels=32)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1, encoder1 = self.backbone(x1)
        x2, encoder2 = self.backbone(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)
        y2 = self.decoder_layer_4(y1)
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = torch.cat([y2, encoder[-1]], dim=1)
        y4 = self.decoder_layer_3(y3)
        y4 = self.upsamplex2(y4)

        # decoder layer 2
        y5 = torch.cat([y4, encoder[-2]], dim=1)
        y6 = self.decoder_layer_2(y5)
        y6 = self.upsamplex2(y6)

        # decoder layer 1
        y7 = torch.cat([y6, encoder[-3]], dim=1)
        y8 = self.decoder_layer_1(y7)

        # final predict
        y_out = self.classifier(y8)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out


class SkipLightNet_MSOF(torch.nn.Module):
    def __init__(self, input_nc, output_nc, backbone='mobilenetv2', output_sigmoid=False):
        super(SkipLightNet_MSOF, self).__init__()
        if backbone == 'mobilenetv2':
            self.backbone = models.mobilenetv2(pretrained=False)
        elif backbone == 'mobilenetv3':
            self.backbone = models.mobilenetv3(pretrained=False, input_size=256, mode='large')
        elif backbone == 'ghostnet':
            self.backbone = models.ghostnet(pretrained=False)
        elif backbone == 'shufflenetv2':
            self.backbone = models.ShuffleNetV2()
        elif backbone == 'peleenet':
            self.backbone = models.peleenet()
        else:
            raise NotImplementedError

        # MSOF Decoder
        inter_depth = 32
        if backbone == 'mobilenetv2':
            self.head_3 = DeepSupervision(in_chan=320, n_classes=inter_depth, up_factor=8)
            self.head_2 = DeepSupervision(in_chan=32, n_classes=inter_depth, up_factor=4)
            self.head_1 = DeepSupervision(in_chan=24, n_classes=inter_depth, up_factor=2)
            self.head_0 = DeepSupervision(in_chan=16, n_classes=inter_depth, up_factor=1)
        elif backbone == 'mobilenetv3':  # Large & Small Version
            self.head_3 = DeepSupervision(in_chan=160, n_classes=inter_depth, up_factor=8)
            self.head_2 = DeepSupervision(in_chan=40, n_classes=inter_depth, up_factor=4)
            self.head_1 = DeepSupervision(in_chan=24, n_classes=inter_depth, up_factor=2)
            self.head_0 = DeepSupervision(in_chan=16, n_classes=inter_depth, up_factor=1)
            # self.head_3 = DeepSupervision(in_chan=96, n_classes=inter_depth, up_factor=8)
            # self.head_2 = DeepSupervision(in_chan=24, n_classes=inter_depth, up_factor=4)
            # self.head_1 = DeepSupervision(in_chan=16, n_classes=inter_depth, up_factor=2)
            # self.head_0 = DeepSupervision(in_chan=16, n_classes=inter_depth, up_factor=1)
        elif backbone == 'ghostnet':
            self.head_3 = DeepSupervision(in_chan=160, n_classes=inter_depth, up_factor=8)
            self.head_2 = DeepSupervision(in_chan=40, n_classes=inter_depth, up_factor=4)
            self.head_1 = DeepSupervision(in_chan=24, n_classes=inter_depth, up_factor=2)
            self.head_0 = DeepSupervision(in_chan=16, n_classes=inter_depth, up_factor=1)
        elif backbone == 'shufflenetv2':
            self.head_3 = DeepSupervision(in_chan=464, n_classes=inter_depth, up_factor=8)
            self.head_2 = DeepSupervision(in_chan=232, n_classes=inter_depth, up_factor=4)
            self.head_1 = DeepSupervision(in_chan=116, n_classes=inter_depth, up_factor=2)
            self.head_0 = DeepSupervision(in_chan=24, n_classes=inter_depth, up_factor=1)
        elif backbone == 'peleenet':
            self.head_3 = DeepSupervision(in_chan=704, n_classes=inter_depth, up_factor=8)
            self.head_2 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=4)
            self.head_1 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=2)
            self.head_0 = DeepSupervision(in_chan=32, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1, encoder1 = self.backbone(x1)
        x2, encoder2 = self.backbone(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out


class LightResNet18(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
                 deepsup=False, norm_layer=None, output_sigmoid=False):
        super(LightResNet18, self).__init__()
        self.deepsup = deepsup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, STDCConcat, 64, 64, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, STDCConcat, 64, 128, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, STDCConcat, 128, 256, self.blocks[2], self.stride[2])

        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        # Deep Supervision
        if self.deepsup == True:
            self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
            self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
            self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=output_nc, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x):
        aux_out = []
        x1, encoder1 = self.forward_single(x)
        x2, encoder2 = self.forward_single(x)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 256
        y2 = self.decoder_layer_4(y1)  # dim = 128
        if self.deepsup == True:
            aux1 = self.deepsup_3(y2)
            aux_out.append(aux1)
        y2 = self.upsamplex2(y2)
        # save_vis_features(y1.data, 'y1')
        # save_vis_features(y2.data, 'y2')

        # decoder layer 3
        y3 = self.decoder_layer_3(torch.cat([y2, encoder[-1]], dim=1))  # dim = 64
        # save_vis_features(encoder[-1].data, 'diff[-1]')
        if self.deepsup == True:
            aux2 = self.deepsup_2(y3)
            aux_out.append(aux2)
        y3 = self.upsamplex2(y3)
        # save_vis_features(y3.data, 'y3')

        # decoder layer 2
        y4 = self.decoder_layer_2(torch.cat([y3, encoder[-2]], dim=1))  # dim = 64
        # save_vis_features(encoder[-2].data, 'diff[-2]')
        if self.deepsup == True:
            aux3 = self.deepsup_1(y4)
            aux_out.append(aux3)
        y4 = self.upsamplex2(y4)
        # save_vis_features(y4.data, 'y4')

        # decoder layer 1
        y5 = self.decoder_layer_1(torch.cat([y4, encoder[-3]], dim=1))  # dim =32
        # save_vis_features(encoder[-3].data, 'diff[-3]')
        # save_vis_features(y5.data, 'y5')

        # final predict
        y_out = self.classifier(y5)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        if self.deepsup == True:
            return y_out, aux_out
        else:
            return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet18_New(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, type, block=LightBasic,
                 deepsup=False, norm_layer=None, output_sigmoid=False):
        super(LightResNet18_New, self).__init__()
        self.deepsup = deepsup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, type[0], 64, 64, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, type[1], 64, 128, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, type[2], 128, 256, self.blocks[2], self.stride[2])

        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        # Deep Supervision
        if self.deepsup == True:
            self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
            self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
            self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)

        # self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.classifier = FinalClassifier(in_channels=32, out_channels=output_nc, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        aux_out = []
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 256
        y2 = self.decoder_layer_4(y1)  # dim = 128
        if self.deepsup == True:
            aux1 = self.deepsup_3(y2)
            aux_out.append(aux1)
        y2 = self.upsamplex2(y2)
        # save_vis_features(y1.data, 'y1')
        # save_vis_features(y2.data, 'y2')

        # decoder layer 3
        y3 = self.decoder_layer_3(torch.cat([y2, encoder[-1]], dim=1))  # dim = 64
        # save_vis_features(encoder[-1].data, 'diff[-1]')
        if self.deepsup == True:
            aux2 = self.deepsup_2(y3)
            aux_out.append(aux2)
        y3 = self.upsamplex2(y3)
        # save_vis_features(y3.data, 'y3')

        # decoder layer 2
        y4 = self.decoder_layer_2(torch.cat([y3, encoder[-2]], dim=1))  # dim = 64
        # save_vis_features(encoder[-2].data, 'diff[-2]')
        if self.deepsup == True:
            aux3 = self.deepsup_1(y4)
            aux_out.append(aux3)
        y4 = self.upsamplex2(y4)
        # save_vis_features(y4.data, 'y4')

        # decoder layer 1
        y5 = self.decoder_layer_1(torch.cat([y4, encoder[-3]], dim=1))  # dim =32
        # save_vis_features(encoder[-3].data, 'diff[-3]')
        # save_vis_features(y5.data, 'y5')

        # final predict
        y_out = self.classifier(y5)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        if self.deepsup == True:
            return y_out, aux_out
        else:
            return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet18_BE(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
                 bdenhance=False, norm_layer=None):
        super(LightResNet18_BE, self).__init__()
        self.bdenhance = bdenhance
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, SCSGE, 64, 64, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, SCSGE, 64, 128, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SCSGE, 128, 256, self.blocks[2], self.stride[2])

        # Decoder
        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        # Boundary-Assisted Learning
        if self.bdenhance == True:
            self.se_3 = SideExtractionLayer(in_channels=128, factor=8, kernel_size=16, padding=4)
            self.se_2 = SideExtractionLayer(in_channels=64, factor=4, kernel_size=8, padding=2)
            self.se_1 = SideExtractionLayer(in_channels=64, factor=2, kernel_size=4, padding=1)
            self.SV = BoundaryExtraction(pool_size=5)
            self.bd_generator = nn.Conv2d(4, output_nc, 3, 1)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        side_out = []
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 256
        y2 = self.decoder_layer_4(y1)  # dim = 128
        if self.bdenhance == True:
            aux1 = self.SV(self.se_3(y2))
            side_out.append(aux1)
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = self.decoder_layer_3(torch.cat([y2, encoder[-1]], dim=1))  # dim = 64
        if self.bdenhance == True:
            aux2 = self.SV(self.se_2(y3))
            side_out.append(aux2)
        y3 = self.upsamplex2(y3)

        # decoder layer 2
        y4 = self.decoder_layer_2(torch.cat([y3, encoder[-2]], dim=1))  # dim = 64
        if self.bdenhance == True:
            aux3 = self.SV(self.se_1(y4))
            side_out.append(aux3)
        y4 = self.upsamplex2(y4)

        # decoder layer 1
        y5 = self.decoder_layer_1(torch.cat([y4, encoder[-3]], dim=1))  # dim =32

        # final predict
        y_out = self.classifier(y5)

        # boundary-assisted part
        if self.bdenhance == True:
            y_out = self.softmax(y_out)
            side_out.append(self.SV(torch.max(y_out, dim=1, keepdim=True)[0]))
            boundary_pred = self.bd_generator(torch.cat(side_out, dim=1))
            # boundary_pred = self.sigmoid(boundary_pred)

        if self.bdenhance == True:
            return y_out, boundary_pred
        else:
            return y_out


    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet18_DiffFPN(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
                 norm_layer=None, output_sigmoid=False):
        super(LightResNet18_DiffFPN, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, SCSGE, 64, 64, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, SCSGE, 64, 128, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SCSGE, 128, 256, self.blocks[2], self.stride[2])

        ### FPN Decoder
        filters = [64, 64, 128, 256]
        self.Up1_0 = Up(factor=2, in_ch=filters[1], bilinear=True)
        self.Up2_0 = Up(factor=2, in_ch=filters[2], bilinear=True)
        self.Up3_0 = Up(factor=2, in_ch=filters[3], bilinear=True)

        self.dec0_1 = OneLayerConv2d(filters[0] + filters[1], filters[0])
        self.dec1_1 = OneLayerConv2d(filters[1] + filters[2], filters[1])
        self.Up1_1 = Up(factor=2, in_ch=filters[1], bilinear=True)
        self.dec2_1 = OneLayerConv2d(filters[2] + filters[3], filters[2])
        self.Up2_1 = Up(factor=2, in_ch=filters[2], bilinear=True)

        self.dec0_2 = OneLayerConv2d(filters[0] * 2 + filters[1], filters[0])
        self.dec1_2 = OneLayerConv2d(filters[1] * 2 + filters[2], filters[1])
        self.Up1_2 = Up(factor=2, in_ch=filters[1], bilinear=True)

        self.dec0_3 = OneLayerConv2d(filters[0] * 3 + filters[1], filters[0])

        self.final1 = nn.Conv2d(filters[0], output_nc, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], output_nc, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], output_nc, kernel_size=1)
        self.conv_final = nn.Conv2d(output_nc * 3, output_nc, kernel_size=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # stage 1
        x1_1 = self.conv_in(x1)
        x1_2 = self.bn1(x1_1)
        x1_3 = self.relu(x1_2)

        x2_1 = self.conv_in(x2)
        x2_2 = self.bn1(x2_1)
        x2_3 = self.relu(x2_2)

        # stage 2
        x1_4 = self.maxpool(x1_3)
        x1_5 = self.layer1(x1_4)

        x2_4 = self.maxpool(x2_3)
        x2_5 = self.layer1(x2_4)

        # stage 3
        x1_6 = self.layer2(x1_5)
        x2_6 = self.layer2(x2_5)

        # stage 4
        x1_7 = self.layer3(x1_6)
        x2_7 = self.layer3(x2_6)

        diff = [torch.abs(x2_3 - x1_3), torch.abs(x2_5 - x1_5), torch.abs(x2_6 - x1_6), torch.abs(x2_7 - x1_7)]

        y0_1 = self.dec0_1(torch.cat([diff[0], self.Up1_0(diff[1])], 1))

        y1_1 = self.dec1_1(torch.cat([diff[1], self.Up2_0(diff[2])], 1))
        y0_2 = self.dec0_2(torch.cat([diff[0], y0_1, self.Up1_1(y1_1)], 1))

        y2_1 = self.dec2_1(torch.cat([diff[2], self.Up3_0(diff[3])], 1))
        y1_2 = self.dec1_2(torch.cat([diff[1], y1_1, self.Up2_1(y2_1)], 1))
        y0_3 = self.dec0_3(torch.cat([diff[0], y0_1, y0_2, self.Up1_2(y1_2)], 1))

        output1 = self.final1(y0_1)
        output2 = self.final2(y0_2)
        output3 = self.final3(y0_3)
        y_out = self.conv_final(torch.cat([output1, output2, output3], 1))
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out


class LightResNet18_CSDiffFPN(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
                 norm_layer=None, output_sigmoid=False):
        super(LightResNet18_CSDiffFPN, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, SCSGE, 64, 64, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, SCSGE, 64, 128, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SCSGE, 128, 256, self.blocks[2], self.stride[2])

        ### FPN Decoder
        filters = [64, 64, 128, 256]
        self.up1_0 = Up(factor=2, in_ch=filters[3], out_ch=filters[2], bilinear=True)
        self.up2_0 = Up(factor=2, in_ch=filters[2], out_ch=filters[1], bilinear=True)
        self.up3_0 = Up(factor=2, in_ch=filters[1], out_ch=filters[0], bilinear=True)

        self.cs1_1 = ChannelSelect_DecoderLayer(filters[2], 2)
        self.cs2_1 = ChannelSelect_DecoderLayer(filters[1], 2)
        self.cs3_1 = ChannelSelect_DecoderLayer(filters[0], 2)
        self.up1_1 = Up(factor=2, in_ch=filters[2], out_ch=filters[1], bilinear=True)
        self.up2_1 = Up(factor=2, in_ch=filters[1], out_ch=filters[0], bilinear=True)

        self.cs1_2 = ChannelSelect_DecoderLayer(filters[1], 2)
        self.cs2_2 = ChannelSelect_DecoderLayer(filters[0], 2)

        self.cs1_3 = ChannelSelect_DecoderLayer(filters[0], 2)
        self.up1_2 = Up(factor=2, in_ch=filters[1], out_ch=filters[0], bilinear=True)

        self.predict1 = OneLayerConv2d(in_channels=filters[0], out_channels=32)
        self.predict2 = OneLayerConv2d(in_channels=filters[0], out_channels=32)
        self.predict3 = OneLayerConv2d(in_channels=filters[0], out_channels=32)

        self.final_fusion = ChannelSelect_DecoderLayer(32, 3)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=output_nc, kernel_size=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # stage 1
        x1_1 = self.conv_in(x1)
        x1_2 = self.bn1(x1_1)
        x1_3 = self.relu(x1_2)

        x2_1 = self.conv_in(x2)
        x2_2 = self.bn1(x2_1)
        x2_3 = self.relu(x2_2)

        # stage 2
        x1_4 = self.maxpool(x1_3)
        x1_5 = self.layer1(x1_4)

        x2_4 = self.maxpool(x2_3)
        x2_5 = self.layer1(x2_4)

        # stage 3
        x1_6 = self.layer2(x1_5)
        x2_6 = self.layer2(x2_5)

        # stage 4
        x1_7 = self.layer3(x1_6)
        x2_7 = self.layer3(x2_6)

        diff = [torch.abs(x2_3 - x1_3), torch.abs(x2_5 - x1_5), torch.abs(x2_6 - x1_6), torch.abs(x2_7 - x1_7)]
        y1_0 = diff[3]
        y2_0 = diff[2]
        y3_0 = diff[1]
        y4_0 = diff[0]

        # decoder branch1
        y1_1 = self.cs1_1(self.up1_0(y1_0), y2_0)
        y2_1 = self.cs2_1(self.up2_0(y1_1), y3_0)
        y3_1 = self.cs3_1(self.up3_0(y2_1), y4_0)
        y1 = self.predict1(y3_1)

        y1_2 = self.cs1_2(self.up1_1(y1_1), y2_1)
        y2_2 = self.cs2_2(self.up2_1(y2_1), y3_1)
        y2 = self.predict2(y2_2)

        y1_3 = self.cs1_3(self.up1_2(y1_2), y2_2)
        y3 = self.predict3(y1_3)

        # final predict
        y_out = self.final_fusion(y1, y2, y3)
        y_out = self.classifier(y_out)
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out


class LightResNet50(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck,
                 deepsup=False, norm_layer=None, output_sigmoid=False):
        super(LightResNet50, self).__init__()
        self.deepsup = deepsup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, AttSC, 128, 128, self.blocks[2], self.stride[2])

        self.decoder_layer_4 = OneLayerSepConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerSepConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerSepConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerSepConv2d(in_channels=128, out_channels=32)

        # Deep Supervision
        if self.deepsup == True:
            self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
            self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
            self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        aux_out = []
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 128
        y2 = self.decoder_layer_4(y1)  # dim = 64
        if self.deepsup == True:
            aux1 = self.deepsup_3(y2)
            aux_out.append(aux1)
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = self.decoder_layer_3(torch.cat([y2, encoder[-1]], dim=1))  # dim = 64
        if self.deepsup == True:
            aux2 = self.deepsup_2(y3)
            aux_out.append(aux2)
        y3 = self.upsamplex2(y3)

        # decoder layer 2
        y4 = self.decoder_layer_2(torch.cat([y3, encoder[-2]], dim=1))  # dim = 32
        if self.deepsup == True:
            aux3 = self.deepsup_1(y4)
            aux_out.append(aux3)
        y4 = self.upsamplex2(y4)

        # decoder layer 1
        y5 = self.decoder_layer_1(torch.cat([y4, encoder[-3]], dim=1))  # dim =32

        # final predict
        y_out = self.classifier(y5)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        if self.deepsup == True:
            return y_out, aux_out
        else:
            return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet50_BE(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, bdenhance=False, norm_layer=None):
        super(LightResNet50_BE, self).__init__()
        self.bdenhance = bdenhance
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, SCSGE, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, SCSGE, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SCSGE, 128, 128, self.blocks[2], self.stride[2])

        # Decoder
        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        # Boundary-Assisted Learning
        if self.bdenhance == True:  # SideExtractionLayer : 1x1conv(dimension transform) + Upsample to origin resolution
            self.se_3 = SideExtractionLayer(in_channels=128, factor=8, kernel_size=16, padding=4)
            self.se_2 = SideExtractionLayer(in_channels=64, factor=4, kernel_size=8, padding=2)
            self.se_1 = SideExtractionLayer(in_channels=64, factor=2, kernel_size=4, padding=1)
            self.SV = BoundaryExtraction(pool_size=7)
            self.bd_generator = nn.Conv2d(4, 1, 3, 1)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        side_out = []
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 256
        y2 = self.decoder_layer_4(y1)  # dim = 128
        if self.bdenhance == True:
            aux1 = self.se_3(y2)
            side_out.append(self.SV(aux1))
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = self.decoder_layer_3(torch.cat([y2, encoder[-1]], dim=1))  # dim = 64
        if self.bdenhance == True:
            aux2 = self.se_2(y3)
            side_out.append(self.SV(aux2))
        y3 = self.upsamplex2(y3)

        # decoder layer 2
        y4 = self.decoder_layer_2(torch.cat([y3, encoder[-2]], dim=1))  # dim = 64
        if self.bdenhance == True:
            aux3 = self.se_1(y4)
            side_out.append(self.SV(aux3))
        y4 = self.upsamplex2(y4)

        # decoder layer 1
        y5 = self.decoder_layer_1(torch.cat([y4, encoder[-3]], dim=1))  # dim =32

        # final predict
        y_out = self.classifier(y5)

        # boundary-assisted part
        if self.bdenhance == True:
            y_out = self.softmax(y_out)
            side_out.append(self.SV(torch.unsqueeze(y_out[:, 1, :, :], dim=1)))
            boundary_pred = self.bd_generator(torch.cat(side_out, dim=1))
            # boundary_pred = self.sigmoid(boundary_pred)

        if self.bdenhance == True:
            return y_out, boundary_pred
        else:
            return y_out


    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


# class LightResNet50_New(nn.Module):  # Method: Concat-Skip
#     def __init__(self, input_nc, output_nc, stride, blocks, type, block=LightBottleneck,
#                  deepsup=False, norm_layer=None, output_sigmoid=False):
#         super(LightResNet50_New, self).__init__()
#         self.deepsup = deepsup
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         ### Feature Extractor
#         # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, type[0], 64, 32, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, type[1], 64, 64, self.blocks[1], self.stride[1])
#         self.layer3 = self._make_layer(block, type[2], 128, 128, self.blocks[2], self.stride[2])
#
#         self.decoder_layer_4 = OneLayerConv2d(in_channels=512, out_channels=128)
#         self.decoder_layer_3 = OneLayerConv2d(in_channels=384, out_channels=64)
#         self.decoder_layer_2 = OneLayerConv2d(in_channels=192, out_channels=64)
#         self.decoder_layer_1 = OneLayerConv2d(in_channels=192, out_channels=32)
#
#         # Deep Supervision
#         if self.deepsup == True:
#             self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
#             self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
#             self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)
#
#         self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         aux_out = []
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#
#         # decoder layer 4
#         y1 = torch.cat([x1, x2], dim=1)  # dim = 512
#         y2 = self.decoder_layer_4(y1)  # dim = 128
#         if self.deepsup == True:
#             aux1 = self.deepsup_3(y2)
#             aux_out.append(aux1)
#         y2 = self.upsamplex2(y2)
#
#         # decoder layer 3
#         y3 = self.decoder_layer_3(torch.cat([y2, encoder1[-1], encoder2[-1]], dim=1))  # dim = 64
#         if self.deepsup == True:
#             aux2 = self.deepsup_2(y3)
#             aux_out.append(aux2)
#         y3 = self.upsamplex2(y3)
#
#         # decoder layer 2
#         y4 = self.decoder_layer_2(torch.cat([y3, encoder1[-2], encoder2[-2]], dim=1))  # dim = 64
#         if self.deepsup == True:
#             aux3 = self.deepsup_1(y4)
#             aux_out.append(aux3)
#         y4 = self.upsamplex2(y4)
#
#         # decoder layer 1
#         y5 = self.decoder_layer_1(torch.cat([y4, encoder1[-3], encoder2[-3]], dim=1))  # dim =32
#
#         # final predict
#         y_out = self.classifier(y5)
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         if self.deepsup == True:
#             return y_out, aux_out
#         else:
#             return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # out = 64
#         x_5 = self.layer2(x_4)
#         encoder.append(x_5)   # out = 128
#         x_6 = self.layer3(x_5)
#         out = x_6
#         return out, encoder


class LightResNet50_simpledec(nn.Module):  # 直接上采样插值到原有分辨率
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck,
                 norm_layer=None, output_sigmoid=False):
        super(LightResNet50_simpledec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, AttSC, 128, 128, self.blocks[2], self.stride[2])

        self.decoder_layer = OneLayerConv2d(in_channels=256, out_channels=32)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 256
        y1 = self.decoder_layer(y1)
        y2 = self.upsamplex2(y1)
        y2 = self.upsamplex4(y2)

        # final predict
        y_out = self.classifier(y2)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet50_MSOF(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
        super(LightResNet50_MSOF, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SplitConcat, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet50_MSOF_1(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
        super(LightResNet50_MSOF_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, DWSKConv, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
        # save_vis_features(torch.abs(x1 - x2).data, 'encoder[0]', img_name)
        # save_vis_features(encoder[-1].data, 'encoder[-1]', img_name)
        # save_vis_features(encoder[-2].data, 'encoder[-2]', img_name)
        # save_vis_features(encoder[-3].data, 'encoder[-3]', img_name)

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])
        # save_vis_features(y1.data, 'y1', img_name)
        # save_vis_features(y2.data, 'y2', img_name)
        # save_vis_features(y3.data, 'y3', img_name)
        # save_vis_features(y4.data, 'y4', img_name)

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        # save_vis_features(y_out.data, 'y_out before CAM', img_name)
        # print(self.CAM(y_out).view(-1, 1, 16, 8))
        # save_vis_features(self.CAM(y_out).view(-1, 1, 16, 8).data, 'Channel_Att', img_name)
        y_out = self.CAM(y_out) * y_out
        # save_vis_features(y_out.data, 'y_out after CAM', img_name)
        # save_vis_features(self.squeeze(y_out).data, 'y_out after fusion', img_name)
        # tmp = torch.abs(self.squeeze(y_out) - y1)
        # save_vis_features(tmp.data, 'Difference', img_name)
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet50_MSOF_2(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
        super(LightResNet50_MSOF_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, AttSC, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, AttSC, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, AttSC, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet50_MSOF_new(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
        super(LightResNet50_MSOF_new, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, AttSC, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision_Deconv(in_chan=256, n_classes=inter_depth, factor=8, kernel_size=16, padding=4)
        self.head_2 = DeepSupervision_Deconv(in_chan=128, n_classes=inter_depth, factor=4, kernel_size=8, padding=2)
        self.head_1 = DeepSupervision_Deconv(in_chan=64, n_classes=inter_depth, factor=2, kernel_size=4, padding=1)
        self.head_0 = DeepSupervision_Deconv(in_chan=64, n_classes=inter_depth, factor=1, kernel_size=1, padding=0)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1, encoder1 = self.forward_single(x)
        x2, encoder2 = self.forward_single(x)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightResNet50_MSOF_BE(nn.Module):  # MSOF Decoder with auxiliary module to extract boundary information
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None):
        super(LightResNet50_MSOF_BE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, AttSC, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        # Deep Supervision for Boundary Enhancement
        self.se_3 = SideExtractionLayer(in_channels=256, factor=8, kernel_size=16, padding=4)
        self.se_2 = SideExtractionLayer(in_channels=128, factor=4, kernel_size=8, padding=2)
        self.se_1 = SideExtractionLayer(in_channels=64, factor=2, kernel_size=4, padding=1)
        self.se_0 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.SV = BoundaryExtraction(pool_size=7)
        self.bd_generator = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
        # bd_diff = [self.SV(encoder[i]) for i in range(len(encoder))]

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))

        # boundary enhancement
        # bd1 = self.se_3(torch.abs(x1 - x2))
        bd2 = self.se_2(encoder[-1])
        bd3 = self.se_1(encoder[-2])
        bd4 = self.se_0(encoder[-3])
        bd_out = torch.cat([bd2, bd3, bd4], dim=1)
        bd_out = torch.squeeze(self.bd_generator(bd_out), dim=1)

        return y_out, bd_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)  # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)  # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)  # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


# class LightResNet50_MSOF_Recursive(nn.Module):
#     '''
#         Use Same Block to Reduce the Parameter
#     '''
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
#         super(LightResNet50_MSOF_Recursive, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         ### Feature Extractor
#         self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#         #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, SplitConcat, 64, 32, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, SplitConcat, 64, 64, self.blocks[1], self.stride[1])
#         self.layer3 = self._make_layer(block, SplitConcat, 128, 128, self.blocks[2], self.stride[2])
#
#         # MSOF Decoder
#         inter_depth = 32
#         self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
#         self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
#         self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
#         self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)
#
#         self.CAM = ChannelAttention(in_channels=inter_depth * 4)
#         self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
#         self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         unit = block(inplanes, planes, stride=1, conv=type)  # 需要重复的模块
#         layers.extend([unit for i in range(blocks - 1)])
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#         encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
#
#         # decoder
#         y1 = self.head_3(torch.abs(x1 - x2))
#         y2 = self.head_2(encoder[-1])
#         y3 = self.head_1(encoder[-2])
#         y4 = self.head_0(encoder[-3])
#
#         # final predict
#         y_out = torch.cat([y1, y2, y3, y4], dim=1)
#         y_out = self.CAM(y_out) * y_out
#         y_out = self.classifier(self.squeeze(y_out))
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # out = 64
#         x_5 = self.layer2(x_4)
#         encoder.append(x_5)   # out = 128
#         x_6 = self.layer3(x_5)
#         out = x_6
#         return out, encoder


# class LightResNet50_MSOF(nn.Module):
'''
    Add Self-Attention Block to extract the global context information
'''
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
#         super(LightResNet50_MSOF, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         ### Feature Extractor
#         self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#         #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.num_heads = [2, 4, 8]
#         self.layer1 = self._make_layer(block, None, 64, 16, self.blocks[0], self.stride[0])
#         self.SARefine1 = SARefine(64, 32, 128, stride=1, num_heads=self.num_heads[0], proj_drop=0.1, qkv_bias=False, qk_scale=None,
#                                   norm_layer=nn.LayerNorm, sr_ratio=8)
#         # self.CCAtt1 = nn.Sequential(CrissCrossAttention(64, 32, stride=1), CrissCrossAttention(32, 32, stride=1))
#         # self.CrossAtt1 = CrossAttDec(64, num_heads=self.num_heads[0])
#         self.layer2 = self._make_layer(block, None, 64, 32, self.blocks[1], self.stride[1])
#         self.SARefine2 = SARefine(64, 64, 128, stride=2, num_heads=self.num_heads[1], proj_drop=0.1, qkv_bias=False, qk_scale=None,
#                                   norm_layer=nn.LayerNorm, sr_ratio=4)
#         # self.CCAtt2 = nn.Sequential(CrissCrossAttention(64, 64, stride=2), CrissCrossAttention(64, 64, stride=1))
#         # self.CrossAtt2 = CrossAttDec(128, num_heads=self.num_heads[1])
#         self.layer3 = self._make_layer(block, None, 128, 64, self.blocks[2], self.stride[2])
#         self.SARefine3 = SARefine(128, 128, 64, stride=2, num_heads=self.num_heads[2], proj_drop=0.1, qkv_bias=False, qk_scale=None,
#                                   norm_layer=nn.LayerNorm, sr_ratio=2)
#         # self.CCAtt3 = nn.Sequential(CrissCrossAttention(128, 128, stride=2), CrissCrossAttention(128, 128, stride=1))
#         # self.CrossAtt3 = CrossAttDec(256, num_heads=self.num_heads[2])
#
#         # MSOF Decoder
#         inter_depth = 32
#         self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
#         self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
#         self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
#         self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)
#
#         self.CAM = ChannelAttention(in_channels=inter_depth * 4)
#         self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
#         self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#         encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
#
#         # decoder
#         y1 = self.head_3(torch.abs(x1 - x2))
#         y2 = self.head_2(encoder[-1])
#         y3 = self.head_1(encoder[-2])
#         y4 = self.head_0(encoder[-3])
#
#         # final predict
#         y_out = torch.cat([y1, y2, y3, y4], dim=1)
#         y_out = self.CAM(y_out) * y_out
#         y_out = self.classifier(self.squeeze(y_out))
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         token_1 = self.SARefine1(x_3)
#         # token_1 = self.CCAtt1(x_3)
#         x_4 = self.layer1(x_3)
#         # x_4 = self.CrossAtt1(x_4, token_1)
#         x_4 = torch.cat([x_4, token_1], dim=1)
#         encoder.append(x_4)   # out = 64
#         token_2 = self.SARefine2(x_4)
#         # token_2 = self.CCAtt2(x_4)
#         x_5 = self.layer2(x_4)
#         # x_5 = self.CrossAtt2(x_5, token_2)
#         x_5 = torch.cat([x_5, token_2], dim=1)
#         encoder.append(x_5)   # out = 128
#         token_3 = self.SARefine3(x_5)
#         # token_3 = self.CCAtt3(x_5)
#         x_6 = self.layer3(x_5)
#         # x_6 = self.CrossAtt3(x_6, token_3)
#         x_6 = torch.cat([x_6, token_3], dim=1)
#         out = x_6
#         return out, encoder


# class LightResNet50_MSOF_new(nn.Module):  # Another Structure for MSOF
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
#         super(LightResNet50_MSOF_new, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         ### Feature Extractor
#         self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#         #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
#         self.layer3 = self._make_layer(block, None, 128, 128, self.blocks[2], self.stride[2])
#
#         # MSOF Decoder
#         inter_depth = 32
#         self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
#         self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
#         self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
#         self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)
#
#         self.CAM = ChannelAttention(in_channels=inter_depth * 4)
#         self.classifier = TwoLayerSepConv2d(in_channels=inter_depth * 4, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#         encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
#
#         # decoder
#         y1 = self.head_3(torch.abs(x1 - x2))
#         y2 = self.head_2(encoder[-1])
#         y3 = self.head_1(encoder[-2])
#         y4 = self.head_0(encoder[-3])
#
#         # final predict
#         y_out = torch.cat([y1, y2, y3, y4], dim=1)
#         y_out = self.CAM(y_out) * y_out
#         y_out = self.classifier(y_out)
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # out = 64
#         x_5 = self.layer2(x_4)
#         encoder.append(x_5)   # out = 128
#         x_6 = self.layer3(x_5)
#         out = x_6
#         return out, encoder


######## Channel Attention for Fusion
# class LightResNet18_attfusion(nn.Module):
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
#                  deepsup=False, norm_layer=None, output_sigmoid=False):
#         super(LightResNet18_attfusion, self).__init__()
#         self.deepsup = deepsup
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         # Feature Extractor
#         # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, SCSGE, 64, 64, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, SCSGE, 64, 128, self.blocks[1], self.stride[1])
#         self.layer3 = self._make_layer(block, SCSGE, 128, 256, self.blocks[2], self.stride[2])
#
#         # Decoder
#         self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
#         # self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128, kernel_size=1)
#         self.decoder_layer_3 = ChannelAtt_Fusion(in_channels=128, out_channels=64)
#         self.decoder_layer_2 = ChannelAtt_Fusion(in_channels=64, out_channels=64)
#         self.decoder_layer_1 = ChannelAtt_Fusion(in_channels=64, out_channels=32)
#
#         # Deep Supervision
#         if self.deepsup == True:
#             self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
#             self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
#             self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)
#
#         self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         aux_out = []
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#         encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
#
#         # decoder layer 4
#         y1 = torch.abs(x1 - x2)  # dim = 256
#         y2 = self.decoder_layer_4(y1)  # dim = 128
#         if self.deepsup == True:
#             aux1 = self.deepsup_3(y2)
#             aux_out.append(aux1)
#         y2 = self.upsamplex2(y2)
#
#         # decoder layer 3
#         y3 = self.decoder_layer_3(y2, encoder[-1])  # dim = 64
#         if self.deepsup == True:
#             aux2 = self.deepsup_2(y3)
#             aux_out.append(aux2)
#         y3 = self.upsamplex2(y3)
#
#         # decoder layer 2
#         y4 = self.decoder_layer_2(y3, encoder[-2])  # dim = 64
#         if self.deepsup == True:
#             aux3 = self.deepsup_1(y4)
#             aux_out.append(aux3)
#         y4 = self.upsamplex2(y4)
#
#         # decoder layer 1
#         y5 = self.decoder_layer_1(y4, encoder[-3])  # dim =32
#
#         # final predict
#         y_out = self.classifier(y5)
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         if self.deepsup == True:
#             return y_out, aux_out
#         else:
#             return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # out = 64
#         x_5 = self.layer2(x_4)
#         encoder.append(x_5)   # out = 128
#         x_6 = self.layer3(x_5)
#         out = x_6
#         return out, encoder
#
#
# class LightResNet50_attfusion(nn.Module):
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck,
#                  deepsup=False, norm_layer=None, output_sigmoid=False):
#         super(LightResNet50_attfusion, self).__init__()
#         self.deepsup = deepsup
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         # Feature Extractor
#         # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, SCSGE, 64, 32, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, SCSGE, 64, 64, self.blocks[1], self.stride[1])
#         self.layer3 = self._make_layer(block, SCSGE, 128, 128, self.blocks[2], self.stride[2])
#
#         # Decoder
#         self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
#         # self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128, kernel_size=1)
#         self.decoder_layer_3 = ChannelAtt_Fusion(in_channels=128, out_channels=64)
#         self.decoder_layer_2 = ChannelAtt_Fusion(in_channels=64, out_channels=64)
#         self.decoder_layer_1 = ChannelAtt_Fusion(in_channels=64, out_channels=32)
#
#         # Deep Supervision
#         if self.deepsup == True:
#             self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
#             self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
#             self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)
#
#         self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         aux_out = []
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#         encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
#
#         # decoder layer 4
#         y1 = torch.abs(x1 - x2)  # dim = 256
#         y2 = self.decoder_layer_4(y1)  # dim = 128
#         if self.deepsup == True:
#             aux1 = self.deepsup_3(y2)
#             aux_out.append(aux1)
#         y2 = self.upsamplex2(y2)
#
#         # decoder layer 3
#         y3 = self.decoder_layer_3(y2, encoder[-1])  # dim = 64
#         if self.deepsup == True:
#             aux2 = self.deepsup_2(y3)
#             aux_out.append(aux2)
#         y3 = self.upsamplex2(y3)
#
#         # decoder layer 2
#         y4 = self.decoder_layer_2(y3, encoder[-2])  # dim = 64
#         if self.deepsup == True:
#             aux3 = self.deepsup_1(y4)
#             aux_out.append(aux3)
#         y4 = self.upsamplex2(y4)
#
#         # decoder layer 1
#         y5 = self.decoder_layer_1(y4, encoder[-3])  # dim =32
#
#         # final predict
#         y_out = self.classifier(y5)
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         if self.deepsup == True:
#             return y_out, aux_out
#         else:
#             return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # out = 64
#         x_5 = self.layer2(x_4)
#         encoder.append(x_5)   # out = 128
#         x_6 = self.layer3(x_5)
#         out = x_6
#         return out, encoder


######  With Spatial and Channel Attention
# class LightResNet18_doubleatt(nn.Module):
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
#                  deepsup=False, norm_layer=None, output_sigmoid=False):
#         super(LightResNet18_doubleatt, self).__init__()
#         self.deepsup = deepsup
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         # Feature Extractor
#         # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, SplitConcat, 64, 64, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, SplitConcat, 64, 128, self.blocks[1], self.stride[1])
#         self.layer3 = self._make_layer(block, SplitConcat, 128, 256, self.blocks[2], self.stride[2])
#
#         # Decoder
#         self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
#         self.decoder_layer_3 = ChannelAtt_Fusion(in_channels=128, out_channels=64)
#         self.decoder_layer_2 = ChannelAtt_Fusion(in_channels=64, out_channels=64)
#         self.decoder_layer_1 = ChannelAtt_Fusion(in_channels=64, out_channels=32)
#
#         # Spatial Attention
#         self.spatial_att = SpatialAttention()
#
#         # Deep Supervision
#         if self.deepsup == True:
#             self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
#             self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
#             self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)
#
#         self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         aux_out = []
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#         encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
#
#         # decoder layer 4
#         y1 = torch.abs(x1 - x2)  # dim = 256
#         y2 = self.decoder_layer_4(y1)  # dim = 128
#         y2 = self.spatial_att(y2)
#         if self.deepsup == True:
#             aux1 = self.deepsup_3(y2)
#             aux_out.append(aux1)
#         y2 = self.upsamplex2(y2)
#
#         # decoder layer 3
#         y3 = self.decoder_layer_3(y2, encoder[-1])  # dim = 64
#         y3 = self.spatial_att(y3)
#         if self.deepsup == True:
#             aux2 = self.deepsup_2(y3)
#             aux_out.append(aux2)
#         y3 = self.upsamplex2(y3)
#
#         # decoder layer 2
#         y4 = self.decoder_layer_2(y3, encoder[-2])  # dim = 64
#         y4 = self.spatial_att(y4)
#         if self.deepsup == True:
#             aux3 = self.deepsup_1(y4)
#             aux_out.append(aux3)
#         y4 = self.upsamplex2(y4)
#
#         # decoder layer 1
#         y5 = self.decoder_layer_1(y4, encoder[-3])  # dim =32
#         y5 = self.spatial_att(y5)
#
#         # final predict
#         y_out = self.classifier(y5)
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         if self.deepsup == True:
#             return y_out, aux_out
#         else:
#             return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # out = 64
#         x_5 = self.layer2(x_4)
#         encoder.append(x_5)   # out = 128
#         x_6 = self.layer3(x_5)
#         out = x_6
#         return out, encoder
#
#
# class LightResNet50_doubleatt(nn.Module):
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck,
#                  deepsup=False, norm_layer=None, output_sigmoid=False):
#         super(LightResNet50_doubleatt, self).__init__()
#         self.deepsup = deepsup
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         # Feature Extractor
#         # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, SplitConcat, 64, 32, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, SplitConcat, 64, 64, self.blocks[1], self.stride[1])
#         self.layer3 = self._make_layer(block, SplitConcat, 128, 128, self.blocks[2], self.stride[2])
#
#         # Decoder
#         self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
#         self.decoder_layer_3 = ChannelAtt_Fusion(in_channels=128, out_channels=64)
#         self.decoder_layer_2 = ChannelAtt_Fusion(in_channels=64, out_channels=64)
#         self.decoder_layer_1 = ChannelAtt_Fusion(in_channels=64, out_channels=32)
#
#         # Spatial Attention
#         self.spatial_att = SpatialAttention()
#
#         # Deep Supervision
#         if self.deepsup == True:
#             self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
#             self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
#             self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)
#
#         self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         aux_out = []
#         x1, encoder1 = self.forward_single(x1)
#         x2, encoder2 = self.forward_single(x2)
#         encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]
#
#         # decoder layer 4
#         y1 = torch.abs(x1 - x2)  # dim = 256
#         y2 = self.decoder_layer_4(y1)  # dim = 128
#         y2 = self.spatial_att(y2)
#         if self.deepsup == True:
#             aux1 = self.deepsup_3(y2)
#             aux_out.append(aux1)
#         y2 = self.upsamplex2(y2)
#
#         # decoder layer 3
#         y3 = self.decoder_layer_3(y2, encoder[-1])  # dim = 64
#         y3 = self.spatial_att(y3)
#         if self.deepsup == True:
#             aux2 = self.deepsup_2(y3)
#             aux_out.append(aux2)
#         y3 = self.upsamplex2(y3)
#
#         # decoder layer 2
#         y4 = self.decoder_layer_2(y3, encoder[-2])  # dim = 64
#         y4 = self.spatial_att(y4)
#         if self.deepsup == True:
#             aux3 = self.deepsup_1(y4)
#             aux_out.append(aux3)
#         y4 = self.upsamplex2(y4)
#
#         # decoder layer 1
#         y5 = self.decoder_layer_1(y4, encoder[-3])  # dim =32
#         y5 = self.spatial_att(y5)
#
#         # final predict
#         y_out = self.classifier(y5)
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         if self.deepsup == True:
#             return y_out, aux_out
#         else:
#             return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # out = 64
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # out = 64
#         x_5 = self.layer2(x_4)
#         encoder.append(x_5)   # out = 128
#         x_6 = self.layer3(x_5)
#         out = x_6
#         return out, encoder


######  Early Fusion Model
class EFLightResNet18(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
                 deepsup=False, norm_layer=None, output_sigmoid=False):
        super(EFLightResNet18, self).__init__()
        self.deepsup = deepsup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, SplitConcat, 64, 64, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, SplitConcat, 64, 128, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SplitConcat, 128, 256, self.blocks[2], self.stride[2])

        # Decoder
        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        # Deep Supervision
        if self.deepsup == True:
            self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
            self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
            self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        aux_out = []
        x, encoder = self.forward_single(torch.cat([x1, x2], dim=1))

        # decoder layer 4
        y1 = self.decoder_layer_4(x)  # dim = 128
        if self.deepsup == True:
            aux1 = self.deepsup_3(y1)
            aux_out.append(aux1)
        y1 = self.upsamplex2(y1)

        # decoder layer 3
        y2 = torch.cat([y1, encoder[-1]], dim=1)  # dim = 256
        y3 = self.decoder_layer_3(y2)  # dim = 64
        if self.deepsup == True:
            aux2 = self.deepsup_2(y3)
            aux_out.append(aux2)
        y3 = self.upsamplex2(y3)

        # decoder layer 2
        y4 = torch.cat([y3, encoder[-2]], dim=1)  # dim = 128
        y5 = self.decoder_layer_2(y4)  # dim = 64
        if self.deepsup == True:
            aux3 = self.deepsup_2(y5)
            aux_out.append(aux3)
        y5 = self.upsamplex2(y5)

        # decoder layer 1
        y6 = torch.cat([y5, encoder[-3]], dim=1)  # dim = 128
        y7 = self.decoder_layer_1(y6)  # dim =32

        # final predict
        y_out = self.classifier(y7)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        if self.deepsup == True:
            return y_out, aux_out
        else:
            return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class EFLightResNet50(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBasic,
                 deepsup=False, norm_layer=None, output_sigmoid=False):
        super(EFLightResNet50, self).__init__()
        self.deepsup = deepsup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc * 2, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, SplitConcat, 32, 16, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, SplitConcat, 32, 32, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SplitConcat, 64, 64, self.blocks[2], self.stride[2])

        self.decoder_layer_4 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=128, out_channels=32)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=64, out_channels=32)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=64, out_channels=32)

        # Deep Supervision
        if self.deepsup == True:
            self.deepsup_3 = DeepSupervision(in_chan=64, n_classes=2, up_factor=8)
            self.deepsup_2 = DeepSupervision(in_chan=32, n_classes=2, up_factor=4)
            self.deepsup_1 = DeepSupervision(in_chan=32, n_classes=2, up_factor=2)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        aux_out = []
        x, encoder = self.forward_single(torch.cat([x1, x2], dim=1))

        # decoder layer 4
        y1 = self.decoder_layer_4(x)  # dim = 128
        if self.deepsup == True:
            aux1 = self.deepsup_3(y1)
            aux_out.append(aux1)
        y1 = self.upsamplex2(y1)

        # decoder layer 3
        y2 = torch.cat([y1, encoder[-1]], dim=1)  # dim = 256
        y3 = self.decoder_layer_3(y2)  # dim = 64
        if self.deepsup == True:
            aux2 = self.deepsup_2(y3)
            aux_out.append(aux2)
        y3 = self.upsamplex2(y3)

        # decoder layer 2
        y4 = torch.cat([y3, encoder[-2]], dim=1)  # dim = 128
        y5 = self.decoder_layer_2(y4)  # dim = 64
        if self.deepsup == True:
            aux3 = self.deepsup_2(y5)
            aux_out.append(aux3)
        y5 = self.upsamplex2(y5)

        # decoder layer 1
        y6 = torch.cat([y5, encoder[-3]], dim=1)  # dim = 128
        y7 = self.decoder_layer_1(y6)  # dim =32

        # final predict
        y_out = self.classifier(y7)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        if self.deepsup == True:
            return y_out, aux_out
        else:
            return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 32
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 64
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class EFLightResNet50_simpledec(nn.Module):  # 直接上采样插值到原有分辨率
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck,
                 norm_layer=None, output_sigmoid=False):
        super(EFLightResNet50_simpledec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, SplitConcat, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, SplitConcat, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, SplitConcat, 128, 128, self.blocks[2], self.stride[2])

        self.decoder_layer = OneLayerConv2d(in_channels=256, out_channels=32)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=type))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x, encoder = self.forward_single(torch.cat([x1, x2], dim=1))

        # decoder layer 4
        y1 = self.decoder_layer(x)
        y2 = self.upsamplex2(y1)
        y2 = self.upsamplex4(y2)

        # final predict
        y_out = self.classifier(y2)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class EFLightResNet50_MSOF(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
        super(EFLightResNet50_MSOF, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, None, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, None, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, None, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x, encoder = self.forward_single(torch.cat([x1, x2], dim=1))

        # decoder
        y1 = self.head_3(x)
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 32
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 64
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class EFLightResNet50_MSOF_1(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
        super(EFLightResNet50_MSOF_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, DWSKConv, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, DWSKConv, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, DWSKConv, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x, encoder = self.forward_single(torch.cat([x1, x2], dim=1))

        # decoder
        y1 = self.head_3(x)
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 32
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 64
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class EFLightResNet50_MSOF_2(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck, norm_layer=None, output_sigmoid=False):
        super(EFLightResNet50_MSOF_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc * 2, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, AttSC, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, AttSC, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, AttSC, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, conv=None))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, conv=type))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x, encoder = self.forward_single(torch.cat([x1, x2], dim=1))

        # decoder
        y1 = self.head_3(x)
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 32
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 64
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


# class EFLightResNet50_simple(nn.Module):  # Inspired by the MAFF-Net
#     def __init__(self, input_nc, output_nc, stride, blocks, block=LightBottleneck,
#                  deepsup=False, norm_layer=None, output_sigmoid=False):
#         super(EFLightResNet50_simple, self).__init__()
#         self.deepsup = deepsup
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.relu = nn.ReLU()
#         self.upsamplex2 = nn.Upsample(scale_factor=2)
#
#         ### Feature Extractor
#         self.conv_in = nn.Sequential(nn.Conv2d(2 * input_nc, 64, kernel_size=3, stride=2, padding=1, bias=False),
#                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
#         self.bn1 = norm_layer(128)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.stride = stride
#         self.blocks = blocks
#         self.layer1 = self._make_layer(block, SplitConcat, 128, 64, self.blocks[0], self.stride[0])
#         self.layer2 = self._make_layer(block, SplitConcat, 256, 128, self.blocks[1], self.stride[1])
#
#         self.decoder_layer_3 = OneLayerConv2d(in_channels=768, out_channels=256, kernel_size=1)
#         self.decoder_layer_2 = nn.Sequential(OneLayerConv2d(in_channels=256, out_channels=128),
#                                              nn.Dropout(0.5),
#                                              OneLayerConv2d(in_channels=128, out_channels=64),
#                                              nn.Dropout(0.1))
#         self.decoder_layer_1 = OneLayerConv2d(in_channels=64, out_channels=32, kernel_size=1)
#
#         # Deep Supervision
#         if self.deepsup == True:
#             self.deepsup_2 = DeepSupervision(in_chan=256, n_classes=2, up_factor=4)
#             self.deepsup_1 = DeepSupervision(in_chan=32, n_classes=2, up_factor=2)
#
#         self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
#         # self.classifier = FinalClassifier(in_channels=32, out_channels=output_nc, kernel_size=1)
#         self.output_sigmoid = output_sigmoid
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_layer(self, block, type, inplanes, planes, blocks, stride=1):
#         layers = []
#         layers.append(block(inplanes, planes, stride, conv=type))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes, stride=1, conv=type))
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2):
#         aux_out = []
#         x, encoder = self.forward_single(torch.cat([x1, x2], dim=1))
#
#         # decoder layer 4
#         x = self.upsamplex2(x)
#         y1 = torch.cat([x, encoder[-1]], dim=1)  # dim=512+256=768
#         y1 = self.decoder_layer_3(y1)  # dim = 256
#         if self.deepsup == True:
#             aux1 = self.deepsup_2(y1)
#             aux_out.append(aux1)
#         y1 = self.upsamplex2(y1)
#
#         # decoder layer 3
#         y2 = self.decoder_layer_2(y1)  # dim = 64
#         y3 = self.decoder_layer_1(y2)  # dim = 32
#         if self.deepsup == True:
#             aux2 = self.deepsup_1(y3)
#             aux_out.append(aux2)
#         y3 = self.upsamplex2(y3)
#
#         # final predict
#         y_out = self.classifier(y3)
#         # sigmoid operation
#         if self.output_sigmoid:
#             y_out = self.sigmoid(y_out)
#
#         if self.deepsup == True:
#             return y_out, aux_out
#         else:
#             return y_out
#
#     def forward_single(self, x):
#         encoder = []
#         x_1 = self.conv_in(x)
#         x_2 = self.bn1(x_1)
#         x_2 = self.relu(x_2)
#         encoder.append(x_2)   # 1/2, out = 128
#         x_3 = self.maxpool(x_2)
#         x_4 = self.layer1(x_3)
#         encoder.append(x_4)   # 1/4, out = 256
#         x_5 = self.layer2(x_4)
#         out = x_5  # 1/8, out = 512
#         return out, encoder


###### Model Based on Res2Net
class LightRes2Net50(torch.nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=Bottle2neck,
                 deepsup=False, norm_layer=None, output_sigmoid=False):
        super(LightRes2Net50, self).__init__()
        self.deepsup = deepsup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, 128, 128, self.blocks[2], self.stride[2])

        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        # Deep Supervision
        if self.deepsup == True:
            self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
            self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
            self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        aux_out = []
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(x1 - x2)  # dim = 512
        y2 = self.decoder_layer_4(y1)  # dim = 256
        if self.deepsup == True:
            aux1 = self.deepsup_3(y2)
            aux_out.append(aux1)
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = self.decoder_layer_3(torch.cat([y2, encoder[-1]], dim=1))  # dim = 128
        if self.deepsup == True:
            aux2 = self.deepsup_2(y3)
            aux_out.append(aux2)
        y3 = self.upsamplex2(y3)

        # decoder layer 2
        y4 = self.decoder_layer_2(torch.cat([y3, encoder[-2]], dim=1))  # dim = 64
        if self.deepsup == True:
            aux3 = self.deepsup_1(y4)
            aux_out.append(aux3)
        y4 = self.upsamplex2(y4)

        # decoder layer 1
        y5 = self.decoder_layer_1(torch.cat([y4, encoder[-3]], dim=1))  # dim =32

        # final predict
        y_out = self.classifier(y5)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        if self.deepsup == True:
            return y_out, aux_out
        else:
            return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)   # out = 64
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)   # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)   # out = 128
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class LightRes2Net50_MSOF(torch.nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=Bottle2neck, norm_layer=None, output_sigmoid=False):
        super(LightRes2Net50_MSOF, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ### Feature Extractor
        self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, 128, 128, self.blocks[2], self.stride[2])

        # MSOF Decoder
        inter_depth = 32
        self.head_3 = DeepSupervision(in_chan=256, n_classes=inter_depth, up_factor=8)
        self.head_2 = DeepSupervision(in_chan=128, n_classes=inter_depth, up_factor=4)
        self.head_1 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=2)
        self.head_0 = DeepSupervision(in_chan=64, n_classes=inter_depth, up_factor=1)

        self.CAM = ChannelAttention(in_channels=inter_depth * 4)
        self.squeeze = nn.Conv2d(inter_depth * 4, inter_depth, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=inter_depth, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1, encoder1 = self.forward_single(x1)
        x2, encoder2 = self.forward_single(x2)
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder
        y1 = self.head_3(torch.abs(x1 - x2))
        y2 = self.head_2(encoder[-1])
        y3 = self.head_1(encoder[-2])
        y4 = self.head_0(encoder[-3])

        # final predict
        y_out = torch.cat([y1, y2, y3, y4], dim=1)
        y_out = self.CAM(y_out) * y_out
        y_out = self.classifier(self.squeeze(y_out))
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        return y_out

    def forward_single(self, x):
        encoder = []
        x_1 = self.conv_in(x)
        x_2 = self.bn1(x_1)
        x_2 = self.relu(x_2)
        encoder.append(x_2)  # out = 32
        x_3 = self.maxpool(x_2)
        x_4 = self.layer1(x_3)
        encoder.append(x_4)  # out = 64
        x_5 = self.layer2(x_4)
        encoder.append(x_5)  # out = 64
        x_6 = self.layer3(x_5)
        out = x_6
        return out, encoder


class CSLightRes2Net50(nn.Module):
    def __init__(self, input_nc, output_nc, stride, blocks, block=CSBottle2neck,
                 deepsup=False, norm_layer=None, output_sigmoid=False):
        super(CSLightRes2Net50, self).__init__()
        self.deepsup = deepsup
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        ### Feature Extractor
        # self.conv_in = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_in = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stride = stride
        self.blocks = blocks
        self.layer1 = self._make_layer(block, 64, 32, self.blocks[0], self.stride[0])
        self.layer2 = self._make_layer(block, 64, 64, self.blocks[1], self.stride[1])
        self.layer3 = self._make_layer(block, 128, 128, self.blocks[2], self.stride[2])

        self.decoder_layer_4 = OneLayerConv2d(in_channels=256, out_channels=128)
        self.decoder_layer_3 = OneLayerConv2d(in_channels=256, out_channels=64)
        self.decoder_layer_2 = OneLayerConv2d(in_channels=128, out_channels=64)
        self.decoder_layer_1 = OneLayerConv2d(in_channels=128, out_channels=32)

        # Deep Supervision
        if self.deepsup == True:
            self.deepsup_3 = DeepSupervision(in_chan=128, n_classes=2, up_factor=8)
            self.deepsup_2 = DeepSupervision(in_chan=64, n_classes=2, up_factor=4)
            self.deepsup_1 = DeepSupervision(in_chan=64, n_classes=2, up_factor=2)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        # self.classifier = FinalClassifier(in_channels=32, out_channels=2, kernel_size=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        ## Encoder
        encoder1 = []
        encoder2 = []
        # Stage1
        x_11 = self.conv_in(x1)
        x_12 = self.bn1(x_11)
        x_12 = self.relu(x_12)
        encoder1.append(x_12)  # out = 64
        x_21 = self.conv_in(x2)
        x_22 = self.bn1(x_21)
        x_22 = self.relu(x_22)
        encoder2.append(x_22)  # out = 64
        # Stage2
        x_13 = self.maxpool(x_12)
        x_23 = self.maxpool(x_22)
        x_14, x_24 = self.layer1([x_13, x_23])
        encoder1.append(x_14)  # out = 128
        encoder2.append(x_24)  # out = 128
        # Stage3
        x_15, x_25 = self.layer2([x_14, x_24])
        encoder1.append(x_15)  # out = 256
        encoder2.append(x_25)  # out = 256
        # Stage4
        x_16, x_26 = self.layer3([x_15, x_25])
        out1 = x_16
        out2 = x_26

        ## Decoder
        aux_out = []
        encoder = [torch.abs(encoder1[i] - encoder2[i]) for i in range(len(encoder1))]

        # decoder layer 4
        y1 = torch.abs(out1 - out2)  # dim = 512
        y2 = self.decoder_layer_4(y1)  # dim = 256
        if self.deepsup == True:
            aux1 = self.deepsup_3(y2)
            aux_out.append(aux1)
        y2 = self.upsamplex2(y2)

        # decoder layer 3
        y3 = self.decoder_layer_3(torch.cat([y2, encoder[-1]], dim=1))  # dim = 128
        if self.deepsup == True:
            aux2 = self.deepsup_2(y3)
            aux_out.append(aux2)
        y3 = self.upsamplex2(y3)

        # decoder layer 2
        y4 = self.decoder_layer_2(torch.cat([y3, encoder[-2]], dim=1))  # dim = 64
        if self.deepsup == True:
            aux3 = self.deepsup_1(y4)
            aux_out.append(aux3)
        y4 = self.upsamplex2(y4)

        # decoder layer 1
        y5 = self.decoder_layer_1(torch.cat([y4, encoder[-3]], dim=1))  # dim =32

        # final predict
        y_out = self.classifier(y5)
        # sigmoid operation
        if self.output_sigmoid:
            y_out = self.sigmoid(y_out)

        if self.deepsup == True:
            return y_out, aux_out
        else:
            return y_out


if __name__ == "__main__":
    # torch.cuda.set_device(0)
    ###### Define the network
    # SOTA Method
    net = SiamUnet_conc(3, 2)
    # net = SiamUnet_diff(3, 2)
    # SkipLightResNet
    # net = SkipResNet(input_nc=3, output_nc=2, backbone='resnet18', output_sigmoid=False)
    # net = SkipLightNet(input_nc=3, output_nc=2, backbone='ghostnet', output_sigmoid=False)
    # net = SkipResNet_MSOF(input_nc=3, output_nc=2, backbone='resnet50', output_sigmoid=False)
    # net = SkipLightNet_MSOF(input_nc=3, output_nc=2, backbone='shufflenetv2', output_sigmoid=False)

    # net = LightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[2, 2, 2],
    #                   block=LightBasic, output_sigmoid=False)
    # net = LightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                     block=LightBottleneck_SK, output_sigmoid=False)
    # net = LightResNet50_simpledec(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                               block=LightBottleneck_SK, output_sigmoid=False)
    # net = LightResNet50_MSOF_new(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                          block=LightBottleneck_SK, output_sigmoid=False)
    # net = LightResNet50_MSOF(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                          block=LightBottleneck_SK, output_sigmoid=False)
    # net = LightResNet50_MSOF_Recursive(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                                    block=LightBottleneck_SK, output_sigmoid=False)
    # net = LightResNet50_New(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                         type=[DWSKCAM, DWSKCAM, DWSKCAM], block=LightBottleneck_SK, output_sigmoid=False)
    # net = LightResNet18_DiffFPN(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                             block=LightBasic_SK, output_sigmoid=False)

    # net = LightResNet18_CSDiffFPN(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                               block=LightBasic_SK, output_sigmoid=False)

    # EFLightRes_SK
    # net = EFLightResNet18(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[1, 1, 1],
    #                       deepsup=False, block=LightBasic_SK, output_sigmoid=False)

    # net = EFLightResNet50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       deepsup=False, block=LightBottleneck_SK, output_sigmoid=False)
    # net = EFLightResNet50_simpledec(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                                 block=LightBottleneck_SK, output_sigmoid=False)
    # net = EFLightResNet50_MSOF_2(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                            block=LightBottleneck_SK, output_sigmoid=False)

    # Res2Net-based
    # net = LightRes2Net50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       deepsup=False, block=Bottle2neck, output_sigmoid=False)

    # net = CSLightRes2Net50(input_nc=3, output_nc=2, stride=[1, 2, 2], blocks=[4, 4, 4],
    #                       deepsup=False, block=CSBottle2neck, output_sigmoid=False)

    net = init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[1])
    ###### information of the defined network
    # summary(net, input_size=(3, 256, 256), batch_size=8)
    # stat(net,(3,256,256))
    ###### Parameters
    # print('# parameters:', sum(param.numel() for param in net.parameters()))
    ###### FLOPs
    # input1 = torch.rand(1, 3, 256, 256).cuda()
    # input2 = torch.rand(1, 3, 256, 256).cuda()
    # flops, params = thop.profile(net, inputs=(input1, input2), verbose=True)
    # print('FLOPs = ' + str(flops / 1E9) + 'G')
    # print('Params = ' + str(params / 1E6) + 'M')
    # flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    ###### calculate the inference time
    repetitions = 300
    input1 = torch.rand(8, 3, 256, 256).to(1)
    input2 = torch.rand(8, 3, 256, 256).to(1)

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = net(input1, input2)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这里使用PyTorch 官方推荐的接口
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = net(input1, input2)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))

# CPU Inference Time
#     repetitions = 300
#     input1 = torch.rand(1, 3, 256, 256)
#     input2 = torch.rand(1, 3, 256, 256)
#
#     # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
#     print('warm up ...\n')
#     with torch.no_grad():
#         for _ in range(100):
#             _ = net(input1, input2)
#
#     # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
#     torch.cuda.synchronize()
#
#     # 设置用于测量时间的 cuda Event, 这里使用PyTorch 官方推荐的接口
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#     # 初始化一个时间容器
#     timings = np.zeros((repetitions, 1))
#
#     print('testing ...\n')
#     with torch.no_grad():
#         for rep in tqdm.tqdm(range(repetitions)):
#             starter.record()
#             _ = net(input1, input2)
#             ender.record()
#             torch.cuda.synchronize()  # 等待GPU任务完成
#             curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
#             timings[rep] = curr_time
#
#     avg = timings.sum() / repetitions
#     print('\navg={}\n'.format(avg))
