import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random


def is_image_file(filename):  # 定义一个判断是否是图片的函数
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def trans_to_tensor(pic):  # 定义一个转变图像格式的函数
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))  # transpose和reshape区别巨大
        return img.float().div(255)
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def data_augment(img1, img2, flip=1, ROTATE_90=1, ROTATE_180=1, ROTATE_270=1, add_noise=1):
    n = flip + ROTATE_90 + ROTATE_180 + ROTATE_270 + add_noise
    a = random.random()
    if flip == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if ROTATE_90 == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    if ROTATE_180 == 1:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    if ROTATE_270 == 1:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    if add_noise == 1:
        pass


def add_salt_and_pepper(image, amount=0.01):

    output = np.copy(np.array(image))

    # add salt
    nb_salt = np.ceil(amount * output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
    output[tuple(coords)] = 255

    # add pepper
    nb_pepper = np.ceil(amount * output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
    output[tuple(coords)] = 0

    return Image.fromarray(output)

## 2020/10/26
import torchvision.transforms as transforms

mean_std = ([0.485, 0.456, 0.406],  # imageNet均值和标准差
            [0.229, 0.224, 0.225])
# 将input变成tensor
input_transform = transforms.Compose([
    transforms.ToTensor(),  ##如果是numpy或者pil image格式，会将[0,255]转为[0,1]，并且(hwc)转为(chw)
    #transforms.Normalize(*mean_std)  # [0,1]  ---> 符合imagenet的范围
])


# 将label变成tensor
def function_label(x):
    if x > 0:
        return 1
    else:
        return 0


class RGBToGray(object):
    def __call__(self, mask):
        mask = mask.convert("L")
        mask = mask.point(function_label)
        return mask


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


target_transform = transforms.Compose([
    RGBToGray(),
    MaskToTensor()
])
palette = [0, 0, 0, 255, 0, 0, 0, 0, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(num_classes * label_true[mask].astype(int) +
                       label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


class train_valid_dataset_6channels(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0):
        super(train_valid_dataset_6channels, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/A/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        initial_path_before = os.path.join(self.data_path + '/A/', self.list[index])
        initial_path_after = os.path.join(self.data_path + '/B/', self.list[index])
        image_name = self.list[index].split('.')[0]
        semantic_path_blue = os.path.join(self.data_path + '/label1/', image_name + '.png')
        semantic_path_red = os.path.join(self.data_path + '/label2/', image_name + '.png')
        assert os.path.exists(semantic_path_blue)
        assert os.path.exists(semantic_path_red)
        try:
            # Image是PIL的一个库
            initial_image_before = Image.open(initial_path_before).convert('RGB')
            initial_image_after = Image.open(initial_path_after).convert('RGB')
            # 如果读入的图像不满足opt输入的尺寸要求，就resize
            initial_image_before = initial_image_before.resize((self.size_w, self.size_h), Image.NEAREST)
            initial_image_after = initial_image_after.resize((self.size_w, self.size_h), Image.NEAREST)
            #  读入两组标签数据(并进行resize)
            semantic_image_blue = Image.open(semantic_path_blue).convert('RGB').resize((self.size_w, self.size_h), Image.NEAREST)
            semantic_image_red = Image.open(semantic_path_red).convert('RGB').resize((self.size_w, self.size_h), Image.NEAREST)
            # 图像增强
            if self.flip == 1:
                a = random.random()
                if a < 1 / 3:
                    initial_image_before = initial_image_before.transpose(Image.FLIP_LEFT_RIGHT)
                    initial_image_after = initial_image_after.transpose(Image.FLIP_LEFT_RIGHT)
                    semantic_image_blue = semantic_image_blue.transpose(Image.FLIP_LEFT_RIGHT)
                    semantic_image_red = semantic_image_red.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    if a < 2 / 3:
                        initial_image_before = initial_image_before.transpose(Image.ROTATE_90)
                        initial_image_after = initial_image_after.transpose(Image.ROTATE_90)
                        semantic_image_blue = semantic_image_blue.transpose(Image.ROTATE_90)
                        semantic_image_red = semantic_image_red.transpose(Image.ROTATE_90)
            semantic_image_blue = np.array(semantic_image_blue, dtype=np.int32)
            semantic_image_red = np.array(semantic_image_red, dtype=np.int32)
            #  分别构建三组二值标签
            semantic_label_blue = np.zeros([semantic_image_blue.shape[0], semantic_image_blue.shape[1]])
            semantic_label_red = np.zeros([semantic_image_red.shape[0], semantic_image_red.shape[1]])
            semantic_label_blue_red = np.zeros([semantic_image_red.shape[0], semantic_image_red.shape[1]])

            semantic_label_blue[(semantic_image_blue[:, :] == (0, 0, 255)).min(2)] = 1
            semantic_label_red[(semantic_image_red[:, :] == (255, 0, 0)).min(2)] = 1

            semantic_image_blue_red = semantic_image_blue + semantic_image_red
            #semantic_label_blue_red[(semantic_label_blue_red_[:, :] == (0, 0, 255)).min(2)] = 2  # 蓝色部分标记为2类
            #semantic_label_blue_red[(semantic_label_blue_red_[:, :] == (255, 0, 0)).min(2)] = 1  # 红色部分标记为1类
            semantic_label_blue_red[(semantic_image_blue_red[:, :] == (255, 0, 255)).min(2)] = 1  # 重叠部分不做损失

        except OSError:
            return None, None, None
        # 转化成tensor，标准化
        initial_image_before = input_transform(initial_image_before)
        initial_image_after = input_transform(initial_image_after)
        #  标签转能输入Loss的Long类型
        semantic_blue_label = torch.from_numpy(semantic_label_blue).long()
        semantic_red_label = torch.from_numpy(semantic_label_red).long()
        semantic_blue_red_label = torch.from_numpy(semantic_label_blue_red).long()
        # 输入图像进行拼接
        initial_image = torch.cat((initial_image_before, initial_image_after), 0)
        # semantic_image = target_transform(semantic_image)
        return initial_image, semantic_blue_label, semantic_red_label,  semantic_blue_red_label, self.list[index]

    def __len__(self):
        return len(self.list)


class train_valid_dataset_6channels_single_label(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0):
        super(train_valid_dataset_6channels_single_label, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/A/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        initial_path_after = os.path.join(self.data_path + '/A/', self.list[index])
        initial_path_before = os.path.join(self.data_path + '/B/', self.list[index])
        image_name = self.list[index].split('.')[0]
        semantic_path = os.path.join(self.data_path + '/label2/', image_name + '.png')
        assert os.path.exists(semantic_path)
        try:
            initial_image_before = Image.open(initial_path_before).convert('RGB')
            initial_image_after = Image.open(initial_path_after).convert('RGB')
            # 如果读入的图像不满足opt输入的尺寸要求，就resize
            initial_image_before = initial_image_before.resize((self.size_w, self.size_h), Image.NEAREST)
            initial_image_after = initial_image_after.resize((self.size_w, self.size_h), Image.NEAREST)
            #  读入标签数据(并进行resize)
            semantic_image = Image.open(semantic_path).convert('RGB').resize((self.size_w, self.size_h), Image.NEAREST)
            # 图像增强
            if self.flip == 1:
                a = random.random()
                if a < 1.0 / 5:
                    contrast = ImageEnhance.Contrast(initial_image_before)
                    initial_image_before = contrast.enhance(2.0)
                    # contrast = ImageEnhance.Contrast(initial_image_after)
                    # initial_image_after = contrast.enhance(2.0)
                elif a >= 1.0 / 5 and a < 2.0 / 5:
                    initial_image_before = ImageOps.equalize(initial_image_before)
                    initial_image_after = ImageOps.equalize(initial_image_after)
                elif a >= 2.0 / 5 and a < 3.0 / 5:
                    b = random.random()
                    if b < 1.0 / 2:
                        initial_image_before = initial_image_before.filter(ImageFilter.GaussianBlur(radius=3))
                        # initial_image_after = initial_image_after.filter(ImageFilter.GaussianBlur(radius=3))
                    else:
                        initial_image_before = initial_image_before.filter(ImageFilter.GaussianBlur(radius=5))
                        # initial_image_after = initial_image_after.filter(ImageFilter.GaussianBlur(radius=5))
                elif a >= 3.0 / 5 and a < 4.0 / 5:
                    initial_image_before = add_salt_and_pepper(initial_image_before)
                    # initial_image_after = add_salt_and_pepper(initial_image_after)
                else:
                    pass

                b = random.random()
                if b < 1.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.FLIP_LEFT_RIGHT)
                    initial_image_after = initial_image_after.transpose(Image.FLIP_LEFT_RIGHT)
                    semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
                elif b >= 1.0 / 6 and b < 2.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.FLIP_TOP_BOTTOM)
                    initial_image_after = initial_image_after.transpose(Image.FLIP_TOP_BOTTOM)
                    semantic_image = semantic_image.transpose(Image.FLIP_TOP_BOTTOM)
                elif b >= 2.0 / 6 and b < 3.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.ROTATE_90)
                    initial_image_after = initial_image_after.transpose(Image.ROTATE_90)
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)
                elif b >= 3.0 / 6 and b < 4.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.ROTATE_180)
                    initial_image_after = initial_image_after.transpose(Image.ROTATE_180)
                    semantic_image = semantic_image.transpose(Image.ROTATE_180)
                elif b >= 4.0 / 6 and b < 5.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.ROTATE_270)
                    initial_image_after = initial_image_after.transpose(Image.ROTATE_270)
                    semantic_image = semantic_image.transpose(Image.ROTATE_270)
                elif b >= 5.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.TRANSPOSE)
                    initial_image_after = initial_image_after.transpose(Image.TRANSPOSE)
                    semantic_image = semantic_image.transpose(Image.TRANSPOSE)
                else:
                    pass
            semantic_image = np.array(semantic_image, dtype=np.int32)
            semantic_label = np.zeros([semantic_image.shape[0], semantic_image.shape[1]])
            semantic_label[(semantic_image[:, :] == (255, 0, 0)).min(2)] = 1

            semantic_label = torch.from_numpy(semantic_label).long()

        except OSError:
            return None, None, None

        initial_image_before = input_transform(initial_image_before)  # 转化成tensor，标准化
        initial_image_after = input_transform(initial_image_after)  # 转化成tensor，标准化
        # initial_image = torch.cat((initial_image_before, initial_image_after), 0)
        # semantic_image = target_transform(semantic_image)

        return {'name': self.list[index], 'A': initial_image_after, 'B': initial_image_before, 'L': semantic_label}

    def __len__(self):
        return len(self.list)


class train_valid_dataset_6channels_ST(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0):
        super(train_valid_dataset_6channels_ST, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/A/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        initial_path_after = os.path.join(self.data_path + '/A/', self.list[index])
        initial_path_before = os.path.join(self.data_path + '/B/', self.list[index])
        image_name = self.list[index].split('.')[0]
        semantic_path = os.path.join(self.data_path + '/label/', image_name + '.png')
        if not os.path.exists(semantic_path):
            semantic_path = os.path.join(self.data_path + '/label/', image_name + '.jpg')
        assert os.path.exists(semantic_path)
        try:
            initial_image_before = Image.open(initial_path_before).convert('RGB')
            initial_image_after = Image.open(initial_path_after).convert('RGB')
            # 如果读入的图像不满足opt输入的尺寸要求，就resize
            initial_image_before = initial_image_before.resize((self.size_w, self.size_h), Image.NEAREST)
            initial_image_after = initial_image_after.resize((self.size_w, self.size_h), Image.NEAREST)
            #  读入标签数据(并进行resize)
            semantic_image = Image.open(semantic_path).convert('RGB').resize((self.size_w, self.size_h), Image.NEAREST)
            # 图像增强
            if self.flip == 1:
                # a = random.random()
                # if a < 1.0 / 5:
                #     contrast = ImageEnhance.Contrast(initial_image_before)
                #     initial_image_before = contrast.enhance(2.0)
                #     # contrast = ImageEnhance.Contrast(initial_image_after)
                #     # initial_image_after = contrast.enhance(2.0)
                # elif a >= 1.0 / 5 and a < 2.0 / 5:
                #     initial_image_before = ImageOps.equalize(initial_image_before)
                #     initial_image_after = ImageOps.equalize(initial_image_after)
                # elif a >= 2.0 / 5 and a < 3.0 / 5:
                #     b = random.random()
                #     if b < 1.0 / 2:
                #         initial_image_before = initial_image_before.filter(ImageFilter.GaussianBlur(radius=3))
                #         # initial_image_after = initial_image_after.filter(ImageFilter.GaussianBlur(radius=3))
                #     else:
                #         initial_image_before = initial_image_before.filter(ImageFilter.GaussianBlur(radius=5))
                #         # initial_image_after = initial_image_after.filter(ImageFilter.GaussianBlur(radius=5))
                # elif a >= 3.0 / 5 and a < 4.0 / 5:
                #     initial_image_before = add_salt_and_pepper(initial_image_before)
                #     # initial_image_after = add_salt_and_pepper(initial_image_after)
                # else:
                #     pass
                #
                b = random.random()
                if b < 1.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.FLIP_LEFT_RIGHT)
                    initial_image_after = initial_image_after.transpose(Image.FLIP_LEFT_RIGHT)
                    semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
                elif b >= 1.0 / 6 and b < 2.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.FLIP_TOP_BOTTOM)
                    initial_image_after = initial_image_after.transpose(Image.FLIP_TOP_BOTTOM)
                    semantic_image = semantic_image.transpose(Image.FLIP_TOP_BOTTOM)
                elif b >= 2.0 / 6 and b < 3.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.ROTATE_90)
                    initial_image_after = initial_image_after.transpose(Image.ROTATE_90)
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)
                elif b >= 3.0 / 6 and b < 4.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.ROTATE_180)
                    initial_image_after = initial_image_after.transpose(Image.ROTATE_180)
                    semantic_image = semantic_image.transpose(Image.ROTATE_180)
                elif b >= 4.0 / 6 and b < 5.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.ROTATE_270)
                    initial_image_after = initial_image_after.transpose(Image.ROTATE_270)
                    semantic_image = semantic_image.transpose(Image.ROTATE_270)
                elif b >= 5.0 / 6:
                    initial_image_before = initial_image_before.transpose(Image.TRANSPOSE)
                    initial_image_after = initial_image_after.transpose(Image.TRANSPOSE)
                    semantic_image = semantic_image.transpose(Image.TRANSPOSE)
                else:
                    pass
            semantic_image = np.array(semantic_image, dtype=np.int32)
            semantic_label = np.zeros([semantic_image.shape[0], semantic_image.shape[1]])
            semantic_label[(semantic_image[:, :] == (255, 255, 255)).min(2)] = 1

            semantic_label = torch.from_numpy(semantic_label).long()

        except OSError:
            return None, None, None

        initial_image_before = input_transform(initial_image_before)  # 转化成tensor，标准化
        initial_image_after = input_transform(initial_image_after)  # 转化成tensor，标准化
        # initial_image = torch.cat((initial_image_before, initial_image_after), 0)
        # semantic_image = target_transform(semantic_image)

        return {'name': self.list[index], 'A': initial_image_after, 'B': initial_image_before, 'L': semantic_label}

    def __len__(self):
        return len(self.list)
