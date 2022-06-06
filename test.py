from argparse import ArgumentParser
import torch
import os

import utils
from models.evaluator import *

print(torch.cuda.is_available())


"""
test the CD model and output the results image
"""


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='LEVIR-256-sgd-0.001warmstep3-noflip-0.6w-AttSCins4_SkipLightRes18_SACbottle_MSOF_444_M4s4_DW5_ce0.5', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoints/LightBiT/LightRes_SK/Ablation/', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test_path', default='./data/LEVIR-CD-256/vis', help='path to test images')
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='SkipLightRes18_SACbottle_MSOF_s4_2', type=str, help=
                        # 'basenet_s3(4) | LightBiT_pos_s3(4) | LightBiT_pos_s3(4)_dd8 | LightBiT_pos_s3(4)_dd8_dedim8|'
                        # 'Skipbasenet_s4 | SkipLightBiT_pos_s4_dd8 | SkipLightBiT_pos_s4_dd8_dedim8'
                        'SkipResNet18_s4'
                        'SkipLightRes18_s4 | SkipLightRes50_s4 | SkipLightRes18_222_s4'
                        'SkipLightRes18_attfusion_s4 | SkipLightRes18_attfusion_222_s4 | SkipLightRes50_attfusion_s4'
                        'SkipLightRes18_att_deepsup_s4 | SkipLightRes50_att_deepsup_s4 '
                        'SkipLightRes18_doubleatt_s4 | SkipLightRes50_doubleatt_s4'
                        'EFLightRes18_s4 | EFLightRes18_SACbasic_s4 | EFLightRes18_SACbottle_s4 | EFLightRes18_SACbottle_simple'
                        'SkipLightRes18_SACbasic_s4 | SkipLightRes18_SACbottle_s4 | SkipLightRes18_SACbasic_New_s4 | SkipLightRes18_SACbottle_New_s4'
                        'SkipLightRes18_SACbasic_deepsup_s4 | SkipLightRes18_SACbasic_s4_bdenhance | SkipLightRes18_SACbottle_s4_bdenhance'
                        'SkipLightRes18_NewDec_SACbasic_s4 | SkipLightRes18_NewDec_SACbottle_s4'
                        'SkipLightRes18_SACbasic_attfusion_s4 | SkipLightRes18_SACbottle_attfusion_s4'
                        'SkipLightRes18_DConvAttbasic_s4 | SkipLightRes18_DConvAttbottle_s4'
                        'SkipLightRes18_MultiDConvbasic_s4 | SkipLightRes18_MultiDConvbottle_s4'
                        'LightRes2Net18_SACbottle_s4 | CSLightRes2Net18_SACbottle_s4')
    parser.add_argument('--loss', default='ce', type=str, help='ce | wce | mix | focal | bd_ce')
    parser.add_argument('--bd_enhance', default=False, type=bool, help='True | False')
    parser.add_argument('--CELoss_weight', type=float, default=0.6, help='The CrossEntropyLoss weight')
    parser.add_argument('--deep_sup', default=False, type=bool, help='True | False')


    args = parser.parse_args()
    utils.get_device(args)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  dir to store the test results
    args.test_dir = os.path.join(args.checkpoint_root, args.project_name, 'test')
    os.makedirs(args.test_dir, exist_ok=True)

    dataloader = utils.test_loader(args)
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()


if __name__ == '__main__':
    main()
