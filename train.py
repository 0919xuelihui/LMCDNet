'''
  SkipLightResNet for RS-CD
'''
from argparse import ArgumentParser
import torch
import os
from models.trainer import CDTrainer
import utils

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

def train(args):
    dataloaders = utils.train_valid_loader(args)  # training datasets & valid datasets
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.test_loader(args)
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='LEVIR-256-sgd-0.001warmstep3-noflip-0.6w-AttSCins4_SkipLightRes18_SACbottle_MSOF_bd_444_M4s4_DW5_coeff2k7_new1', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoints/LightBiT/LightRes_SK/Ablation/', type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--train_path', default='./data/LEVIR-CD-256/train', help='path to training images')
    parser.add_argument('--val_path', default='./data/LEVIR-CD-256/val', help='path to validation images')
    parser.add_argument('--test_path', default='./data/LEVIR-CD-256/test', help='path to test images')
    parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')

    parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
    parser.add_argument('--val_batch_size', default=8, type=int, help='validation batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='test batch size')

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='SkipLightRes18_SACbottle_MSOF_s4_bd', type=str, help=
                        # 'basenet_s3(4) | LightBiT_pos_s3(4) | LightBiT_pos_s3(4)_dd8 | LightBiT_pos_s3(4)_dd8_dedim8|'
                        # 'Skipbasenet_s4 | SkipLightBiT_pos_s4_dd8 | SkipLightBiT_pos_s4_dd8_dedim8'
                        'SkipResNet18_s4 | SkipMobileNet_s4 | SkipGhostNet_s4'
                        'SkipResNet18_MSOF_s4 | SkipResNet50_MSOF_s4 | SkipMobileNetv2_MSOF_s4 | SkipMobileNetv3_MSOF_s4 | SkipGhostNet_MSOF_s4'
                        'SkipLightRes18_s4 | SkipLightRes50_s4 '
                        'SkipLightRes18_NewDec_SACbasic_s4 | SkipLightRes18_NewDec_SACbottle_s4'
                        'SkipLightRes18_attfusion_s4 | SkipSkipLightRes18_MultiDConvbasic_s4 | SkipLightRes50_attfusion_s4'
                        'SkipLightRes18_att_deepsup_s4 | SkipLightRes50_att_deepsup_s4 '
                        'SkipLightRes18_doubleatt_s4 | SkipLightRes50_doubleatt_s4'
                        'EFLightRes18_s4 | EFLightRes18_SACbasic_s4 | EFLightRes18_SACbottle_s4 | EFLightRes18_SACbottle_simple'
                        'EFLightRes18_SACbottle_simpledec_s4'
                        'SkipLightRes18_SACbasic_s4 | SkipLightRes18_SACbottle_s4 | SkipLightRes18_SACbasic_New_s4 | SkipLightRes18_SACbottle_New_s4'
                        'SkipLightRes18_SACbottle_simpledec_s4 | SkipLightRes18_SACbottle_MSOF_s4'
                        'SkipLightRes18_SACbasic_DiffFPN_s4 | SkipLightRes18_SACbasic_CSDiffFPN_s4'
                        'SkipLightRes18_SACbasic_deepsup_s4 | SkipLightRes18_SACbasic_s4_bdenhance | SkipLightRes18_SACbottle_s4_bdenhance'
                        'SkipLightRes18_NewDec_SACbasic_s4 | SkipLightRes18_NewDec_SACbottle_s4'
                        'SkipLightRes18_SACbasic_attfusion_s4 | SkipLightRes18_SACbottle_attfusion_s4'
                        'SkipLightRes18_DConvAttbasic_s4 | SkipLightRes18_DConvAttbottle_s4'
                        'SkipLightRes18_MultiDConvbasic_s4 | SkipLightRes18_MultiDConvbottle_s4'
                        'LightRes2Net18_SACbottle_s4 | LightRes2Net18_SACbottle_MSOF_s4 | CSLightRes2Net18_SACbottle_s4')
    parser.add_argument('--loss', default='ce', type=str, help='ce | wce | mix | focal | bd_ce')
    parser.add_argument('--BD_weight', type=float, default=2, help='The BoundaryLoss weight')
    parser.add_argument('--CELoss_weight', type=float, default=0.6, help='The CrossEntropyLoss weight')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='The DiceLoss weight in MixLoss')

    # training setting
    parser.add_argument('--deep_sup', default=False, type=bool, help='True | False')
    parser.add_argument('--bd_enhance', default=True, type=bool, help='True | False')
    parser.add_argument('--trick', default=False, type=bool, help='True | False')
    parser.add_argument('--optimizer', default='sgd', type=str, help='sgd,adam,adamW')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--max_epochs', default=125, type=int)
    parser.add_argument('--is_warmup', default=True, type=bool, help='True | False')
    parser.add_argument('--lr_policy', default='step', type=str, help='linear | step | multistep | cosine')
    parser.add_argument('--warmup_lr_policy', default='step', type=str, help='poly | exp | step')
    # parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.checkpoint_root, args.project_name, 'vis')
    os.makedirs(args.vis_dir, exist_ok=True)
    #  test dir
    args.test_dir = os.path.join(args.checkpoint_root, args.project_name, 'test')
    os.makedirs(args.test_dir, exist_ok=True)

    train(args)
    test(args)
