import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import data_config
from datasets.CD_dataset import CDDataset
from datasets.dataset import train_valid_dataset_6channels_single_label, train_valid_dataset_6channels_ST


# def get_loader(data_name, img_size=256, batch_size=8, split='test',
#                is_train=False, dataset='CDDataset'):
#     dataConfig = data_config.DataConfig().get_data_config(data_name)  # 根据数据集名称获取数据集地址
#     root_dir = dataConfig.root_dir
#     label_transform = dataConfig.label_transform
#
#     if dataset == 'CDDataset':
#         data_set = CDDataset(root_dir=root_dir, split=split,
#                                  img_size=img_size, is_train=is_train,
#                                  label_transform=label_transform)
#     else:
#         raise NotImplementedError(
#             'Wrong dataset name %s (choose one from [CDDataset])'
#             % dataset)
#
#     shuffle = is_train
#     dataloader = DataLoader(data_set, batch_size=batch_size,
#                                  shuffle=shuffle, num_workers=4)
#
#     return dataloader
#
#
# def get_loaders(args):
#
#     data_name = args.data_name
#     dataConfig = data_config.DataConfig().get_data_config(data_name)
#     root_dir = dataConfig.root_dir
#     label_transform = dataConfig.label_transform
#     split = args.split
#     split_val = 'val'
#     if hasattr(args, 'split_val'):
#         split_val = args.split_val
#     if args.dataset == 'CDDataset':
#         training_set = CDDataset(root_dir=root_dir, split=split,
#                                  img_size=args.img_size,is_train=True,
#                                  label_transform=label_transform)
#         val_set = CDDataset(root_dir=root_dir, split=split_val,
#                                  img_size=args.img_size,is_train=False,
#                                  label_transform=label_transform)
#     else:
#         raise NotImplementedError(
#             'Wrong dataset name %s (choose one from [CDDataset,])'
#             % args.dataset)
#
#     datasets = {'train': training_set, 'val': val_set}
#     dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
#                                  shuffle=True, num_workers=args.num_workers)
#                    for x in ['train', 'val']}
#
#     return dataloaders


def train_valid_loader(args):
    # train_dataset = train_valid_dataset_6channels_single_label(args.train_path, args.img_size, args.img_size, args.flip)
    # valid_dataset = train_valid_dataset_6channels_single_label(args.val_path, args.img_size, args.img_size)
    train_dataset = train_valid_dataset_6channels_ST(args.train_path, args.img_size, args.img_size, args.flip)
    train_dataset_noflip = train_valid_dataset_6channels_ST(args.train_path, args.img_size, args.img_size, 0)
    train_dataset_flip = train_valid_dataset_6channels_ST(args.train_path, args.img_size, args.img_size, 1)
    valid_dataset = train_valid_dataset_6channels_ST(args.val_path, args.img_size, args.img_size)
    # dataset = train_valid_dataset_6channels_ST(args.data_path, args.img_size, args.img_size, args.flip)
    # num_train = len(dataset)
    # # indices = list(range(num_train))
    # batch_num = np.floor(num_train / args.batch_size)
    # batch_num_train = np.floor(args.train_portion * batch_num)
    # split = int(batch_num_train * args.batch_size)
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [split, num_train - split])  # seperate datasets respectively

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=4, drop_last=True)
    train_loader_noflip = torch.utils.data.DataLoader(
        train_dataset_noflip, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=4, drop_last=True)
    train_loader_flip = torch.utils.data.DataLoader(
        train_dataset_flip, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=4, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.val_batch_size, shuffle=True,
        pin_memory=True, num_workers=4, drop_last=True)

    return {'train': train_loader, 'train_flip': train_loader_flip, 'train_noflip': train_loader_noflip, 'val': valid_loader}
    # return {'train': train_loader, 'val': valid_loader}


def test_loader(args):
    test_dataset = train_valid_dataset_6channels_ST(args.test_path, args.img_size, args.img_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        pin_memory=True, num_workers=4, drop_last=True)
    return test_loader


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))  # C,H,W => H,W,C
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
