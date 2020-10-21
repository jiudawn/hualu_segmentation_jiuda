"""
AIIR loader modified
"""
import os, glob
import torch
import numpy as np
import pandas as pd
import collections
import torch
import random
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
from torch.utils.tensorboard import summary
import matplotlib.pyplot as plt
import math

import cv2

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, Cutout, Rotate
)


#  Dataset Class
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, images_path, image_size, mode, masks_path=None):
        """
        Args:
            param df_path: csv文件的路径
            img_dir: 训练样本图片的存放路径
            image_size: 模型的输入图片尺寸
        """
        super(MyDataset).__init__()

        self.class_num = 2
        self.image_size = image_size
        # 是否使用数据增强
        # self.mean = (0.490, 0.490, 0.490)
        # self.std = (0.229, 0.229, 0.229)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # 所有样本和掩膜的名称
        self.images_path = images_path
        self.masks_path = masks_path
        self.mode = mode

    def __getitem__(self, idx):
        """得到样本与其对应的mask
        Return:
            img: 经过预处理的样本图片
            mask: 值为0/1，0表示属于背景，1表示属于目标类
        """
        # 依据idx读取样本图片
        img_path = self.images_path[idx]
        img = Image.open(img_path).convert("RGB")
        # 依据idx读取掩膜
        sample_name = os.path.splitext(img_path)[0].split('/')[-1]
        if self.mode == "train" or self.mode == "validation":
            mask_path = self.masks_path[idx]
            mask = Image.open(mask_path)
            img, mask = self.augmentation(img, mask,self.mode)
            # 对图片和mask同时进行转换
            img = self.image_transform(img)
            mask = self.mask_transform(mask)

            return img, mask

        elif self.mode == 'test':
            img = self.image_transform(img)
            return img,sample_name

    def image_transform(self, image):
        """对样本进行预处理
        """
        resize = transforms.Resize(self.image_size)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(self.mean, self.std)

        transform_compose = transforms.Compose([resize, to_tensor,normalize])

        return transform_compose(image)

    def mask_transform(self, mask):
        """对mask进行预处理
        """
        mask = mask.resize((self.image_size, self.image_size))
        # 将255转换为1， 0转换为0
        mask = np.around(np.array(mask.convert('L')) / 256.)
        # # mask = mask[:, :, np.newaxis] # Wrong, will convert range
        # mask = np.reshape(mask, (np.shape(mask)[0],np.shape(mask)[1],1)).astype("float32")
        # to_tensor = transforms.ToTensor()

        # transform_compose = transforms.Compose([to_tensor])
        # mask = transform_compose(mask)
        # mask = torch.squeeze(mask)
        mask = torch.from_numpy(mask)
        return mask.float()

    def augmentation(self, image, mask,mode):
        """进行数据增强
        Args:
            image: 原始图像，Image图像
            mask: 原始掩膜，Image图像
        Return:
            image_aug: 增强后的图像，Image图像
            mask: 增强后的掩膜，Image图像
        """
        image = np.asarray(image)
        mask = np.asarray(mask)
        image_aug, mask_aug = data_augmentation(image, mask,mode)

        image_aug = Image.fromarray(image_aug)
        mask_aug = Image.fromarray(mask_aug)

        return image_aug, mask_aug

    def __len__(self):
        return len(self.images_path)


def data_augmentation(original_image, original_mask, mode):
    """进行样本和掩膜的随机增强

    Args:
        original_image: 原始图片
        original_mask: 原始掩膜
    Return:
        image_aug: 增强后的图片
        mask_aug: 增强后的掩膜
    """
    original_height, original_width = original_image.shape[:2]
    augmentations = Compose([
        RandomRotate90(p=0.3),
        HorizontalFlip(p=0.3),
        Rotate(limit=15, p=0.3),

        CLAHE(p=0.3),
        HueSaturationValue(20,5,5,p=0.7),
        # 亮度、对比度
        RandomGamma(gamma_limit=(80, 120), p=0.4),
        RandomBrightnessContrast(p=0.4),
        #
        # # 模糊
        # OneOf([
        #     # MotionBlur(p=0.1),
        #     MedianBlur(blur_limit=3, p=0.1),
        #     Blur(blur_limit=3, p=0.1),
        # ], p=0.3),
        #
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),
        # ], p=0.2)
    ])

    augmentations2 = Compose([
        # HorizontalFlip(p=0.2),
        # HueSaturationValue(p=1),

        Rotate(limit=15, p=0.2),
        # CenterCrop(p=0.3, height=original_height, width=original_width),
        # 直方图均衡化
        # CLAHE(p=0.4),
    ])

    if mode=='train':
        augmented = augmentations(image=original_image, mask=original_mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        return image_aug, mask_aug

    elif mode == 'validation':
        augmented = augmentations2(image=original_image, mask=original_mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        return image_aug, mask_aug


def exist_mask(mask):
    """判断是否存在掩膜
    """
    pix_max = torch.max(mask)
    if pix_max == 1:
        flag = 1
    elif pix_max == 0:
        flag = 0

    return flag


def weight_mask(dataset, weights_sample=[1, 3]):
    """计算每一个样本的权重

    Args:
        dataset: 数据集
        weight_sample: 正负类样本对应的采样权重

    Return:
        weights: 每一个样本对应的权重
    """
    print('Start calculating weights of sample...')
    weights = list()
    tbar = tqdm(dataset)
    for index, (image, mask) in enumerate(tbar):
        flag = exist_mask(mask)
        # 存在掩膜的样本的采样权重为3，不存在的为1
        if flag:
            weights.append(weights_sample[1])
        else:
            weights.append(weights_sample[0])
        descript = 'Image %d, flag: %d' % (index, flag)
        tbar.set_description(descript)

    print('Finish calculating weights of sample...')
    return weights


def get_loader(train_images_path, train_masks_path, val_images_path, val_masks_path,
               image_size=224, batch_size=2, num_workers=2, weights_sample=None):
    """Builds and returns Dataloader."""
    # train loader
    dataset_train = MyDataset(train_images_path, image_size, mode='train', masks_path=train_masks_path)
    # val loader, 验证集要保证augmentation_flag为False
    dataset_val = MyDataset(val_images_path,  image_size, mode='validation', masks_path=val_masks_path)

    # 依据weigths_sample决定是否对训练集的样本进行采样
    if weights_sample:
        if os.path.exists('weights_sample.pkl'):
            print('Extract weights of sample from: weights_sample.pkl...')
            with open('weights_sample.pkl', 'rb') as f:
                weights = pickle.load(f)
        else:
            print('Calculating weights of sample...')
            weights = weight_mask(dataset_train, weights_sample)
            with open('weights_sample.pkl', 'wb') as f:
                pickle.dump(weights, f)
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset_train), replacement=True)
        train_data_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                       pin_memory=True)
    else:
        train_data_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                       pin_memory=True)

    val_data_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 pin_memory=True)

    return train_data_loader, val_data_loader

def get_test_loader(test_images_path,image_size=224, batch_size=2, num_workers=2):
    dataset_test = MyDataset(test_images_path, image_size, mode='test')
    test_data_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return test_data_loader

class DatasetsStatic(object):
    def __init__(self, images_path, masks_path, sort_flag=False):
        """
        Args:
            data_root: 数据集的根目录
            image_folder: 样本文件夹名
            mask_folder: 掩膜文件夹名
            sort_flag: bool，是否对样本路径进行排序
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.sort_flag = sort_flag

    def mask_static_level(self, level=16):
        """ 依照掩膜的大小，按照指定的等级数对各样本包含的掩膜进行分级
        """
        masks_path = self.masks_path
        if self.sort_flag:
            masks_path = sorted(masks_path)
        masks_pixes_num = list()
        for index, mask_path in enumerate(masks_path):
            mask_pixes_num = self.cal_mask_pixes(mask_path)
            masks_pixes_num.append(mask_pixes_num)

        masks_pixes_num_np = np.asarray(masks_pixes_num)

        # 最大掩膜和最小掩膜
        mask_max = np.max(masks_pixes_num_np)
        mask_min = np.min(masks_pixes_num_np)
        # 相邻两个等级之间相差的掩膜大小，采用向上取证以保证等级数不会超出level
        step = math.ceil((mask_max - mask_min) / level)
        # 每一个元素表示对应掩膜大小所属的等级
        masks_level = np.zeros_like(masks_pixes_num_np)
        for index, start in enumerate(range(mask_min, mask_max, step)):
            end = start + step
            mask_index = np.where((masks_pixes_num_np >= start) & (masks_pixes_num_np < end))
            masks_level[mask_index] = index

        return masks_level

    def statistical_pixel(self):
        """按像素点计算所有掩模中正负样本的比例
        """
        masks_path = self.masks_path

        if self.sort_flag:
            masks_path = sorted(masks_path)

        masks_pixes_num, backgrounds_pixes_num, all_negative, masks_bool = list(), list(), list(), list()
        for index, mask_path in enumerate(masks_path):
            mask_pixes_num, background_pixes_num = self.cal_mask_pixes(mask_path)
            if mask_pixes_num:
                masks_pixes_num.append(mask_pixes_num)
            else:
                masks_pixes_num.append(0)
                all_negative.append(background_pixes_num)
            backgrounds_pixes_num.append(background_pixes_num)
            masks_bool.append(bool(mask_pixes_num))
        positive_sum = np.sum(masks_pixes_num)
        negative_sum = np.sum(backgrounds_pixes_num)
        negative_sum_mask = np.sum(backgrounds_pixes_num) - np.sum(all_negative)
        return positive_sum, negative_sum, negative_sum / positive_sum, sum(
            masks_bool), negative_sum_mask / positive_sum

    def mask_pixes_average_num(self):
        """统计每个样本所包含的掩膜的像素的平均数目
        Return:
            average: 每个样本所包含的像素的平均数目
        """
        masks_path = self.masks_path
        # 有掩膜的样本的总数
        mask_num = 0
        # 掩膜像素总数
        mask_pixes_sum = 0
        for index, mask_path in enumerate(masks_path):
            mask_pixes_num = self.cal_mask_pixes(mask_path)[0]
            mask_pixes_sum += mask_pixes_num
            if mask_pixes_num:
                mask_num += 1

        average = mask_pixes_sum / mask_num
        return average

    def mask_num_static(self):
        """统计数据集掩膜分布情况
        """
        masks_path = self.masks_path
        # 各样本掩膜的像素数目
        mask_pix_num = list()

        for index, mask_path in enumerate(masks_path):
            mask_pix_per_image = self.cal_mask_pixes(mask_path)[0]
            if mask_pix_per_image:
                mask_pix_num.append(mask_pix_per_image)

        mask_pix_num_np = np.asarray(mask_pix_num)
        # 掩膜像素数目的最小值
        pix_num_min = np.min(mask_pix_num_np)
        # 掩膜像素数目的最大值
        pix_num_max = np.max(mask_pix_num_np)
        # 具有最小的掩膜像素数目的样本个数
        mask_num_pix_min = np.sum(mask_pix_num_np == pix_num_min)
        # 具有最大的掩膜像素数目的样本个数
        mask_num_pix_max = np.sum(mask_pix_num_np == pix_num_max)

        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 6))
        ax0.hist(mask_pix_num, 100, histtype='bar', facecolor='yellowgreen', alpha=0.75, log=True)
        ax0.set_title('Pixes Num of Mask')
        # ax0.text(pix_num_min, mask_num_pix_min, 'min: %d, number: %d'%(pix_num_min, mask_num_pix_min))
        # ax0.text(pix_num_max, mask_num_pix_max, 'max: %d, number: %d'%(pix_num_max, mask_num_pix_max))

        ax0.annotate('min: %d, number: %d' % (pix_num_min, mask_num_pix_min),
                     xy=(pix_num_min, mask_num_pix_min), xytext=(pix_num_min, mask_num_pix_min + 5),
                     arrowprops=dict(facecolor='blue', shrink=0.0005))
        ax0.annotate('max: %d, number: %d' % (pix_num_max, mask_num_pix_max),
                     xy=(pix_num_max, mask_num_pix_max), xytext=(pix_num_max - 20000, mask_num_pix_max + 5),
                     arrowprops=dict(facecolor='green', shrink=0.0005))

        ax1.hist(mask_pix_num, 100, histtype='bar', facecolor='pink', alpha=0.75, cumulative=True, rwidth=0.8)
        ax1.set_title('Accumlation Pixes Num of Mask')
        fig.subplots_adjust(hspace=0.4)
        plt.savefig('./dataset_static.png')

    def cal_mask_pixes(self, mask_path):
        """计算样本的标记的掩膜所包含的像素的总数
        Args:
            mask_path: 标记存放路径
        Return:
            mask_pixes: 掩膜的像素总数
        """
        mask_img = cv2.imread(mask_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        mask_np = np.asarray(mask_img)
        mask_np = mask_np > 0
        mask_pixes = np.sum(mask_np)
        x, y = mask_img.shape
        background_pixes = x * y - mask_pixes
        return mask_pixes, background_pixes


def visualize_aug(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

        plt.show()


def find_image(paths, names):
    files = []
    for i in paths:
        for j in names:
            if j in i:
                files.append(i)
    return files

def find_path(input_set, keyword):
    path = list()
    for directory in input_set:
        p = glob.glob(directory + '/*' + keyword)
        path.extend(p)
    return path


from torch.utils.tensorboard import SummaryWriter
if __name__ == "__main__":
    dataset_root = '/media/totem_disk/totem/weitang/competition/trainData/'
    images_folder = 'image'
    masks_folder = 'mask'
    images_path = glob.glob(dataset_root+images_folder+'/*jpg')
    masks_path = glob.glob(dataset_root+masks_folder+'/*jpg')
    #
    # ds = DatasetsStatic(images_path, masks_path)
    #
    # ds.mask_num_static()
    # average_num = ds.mask_pixes_average_num()
    # print('average num: %d' % (average_num))
    #
    # positive_sum, negative_sum, ratio_all, masks_sum, ratio_mask = ds.statistical_pixel()
    # print(
    #     'positive_sum:{}, negative_sum:{}, ratio_all:{}, mask_sum:{}, ratio_mask:{}'.format(positive_sum, negative_sum,
    #                                                                                         ratio_all, masks_sum,
    #                                                                                         ratio_mask))
    images_path = os.listdir('/media/totem_disk/totem/weitang/competition/data2/image')
    masks_path = os.listdir('/media/totem_disk/totem/weitang/competition/data2/mask')
    img_dir = '/media/totem_disk/totem/weitang/competition/trainData/image'
    ms_dir = '/media/totem_disk/totem/weitang/competition/trainData/mask'
    #
    # random.seed(123)
    # val_id = random.sample(range(700), 100)
    # result = {}
    # image_name = sorted(list(set([i.split('/')[-1].split('_')[0] for i in images_path])))
    # random.seed(123)
    #
    # val_id = random.sample(range(700), 100)
    # imid = [image_name[i] for i in val_id]
    # masks_path = find_path([ms_dir], 'jpg')
    # images_path = find_path([img_dir], 'jpg')
    # val_paths = find_image(images_path, imid)
    # val_paths_masks = find_image(masks_path, imid)
    # for i in range(len(val_paths)):
    #     #     print(val_files[i])
    #     images_path.remove(val_paths[i])
    #     masks_path.remove(val_paths_masks[i])

    images_path = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/data2/image/*jpg'))
    masks_path = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/data2/mask/*jpg'))
    image_name = sorted(list(set([i.split('/')[-1].split('_')[0] for i in images_path])))
    random.seed(123)
    val_id = random.sample(range(700), 100)
    imid = [image_name[i] for i in val_id]
    val_files = pd.DataFrame(data=imid)
    val_files.to_csv('val.csv')
    val_files = find_image(images_path, imid)
    val_masks_files = find_image(masks_path, imid)
    for i in range(len(val_files)):
        images_path.remove(val_files[i])
        masks_path.remove(val_masks_files[i])

    index = 1
    train_loader, val_loader = get_loader(images_path, masks_path,
                                          val_files, val_masks_files,
                                          256,
                                          1, 2, # weights_sample=config.weight_sample
                                          )

    # for i, (images, masks) in enumerate(val_loader):
    #     print(images.shape)
    #     print(masks.shape)
    #     masks = masks.view((masks.shape[0], -1, masks.shape[1], masks.shape[2]))

    for i in range(2):
        image = cv2.imread(val_files[22+i])
        mask = cv2.imread(val_masks_files[22+i], 0)
        augmentations = Compose([
            # RandomRotate90(p=0.5),
            # HorizontalFlip(p=0.5),
            # Rotate(limit=15, p=0.4),
            # # 直方图均衡化
            # CLAHE(p=0.4),
            # HueSaturationValue(p=1),
            # 亮度、对比度
            # RandomGamma(gamma_limit=(80, 120), p=1),
            # RandomBrightnessContrast(p=0.1),

            # 模糊
            # OneOf([
            #     MotionBlur(p=0.1),
            #     MedianBlur(blur_limit=3, p=0.1),
            #     Blur(blur_limit=3, p=0.1),
            # ], p=0.3),

            # OneOf([
            #     IAAAdditiveGaussianNoise(),
            #     GaussNoise(),
            # ], p=0.2)
        ])
        # augmented = Compose([HueSaturationValue(p=1)])(image=image, mask=mask)
        # augmented = Compose([Blur(blur_limit=3, p=1)])(image=image, mask=mask)
        # augmented = Compose([HorizontalFlip(p=0.5)])(image=image, mask=mask)
        # augmented = Compose([CLAHE(p=0.4)])(image=image, mask=mask)
        # augmented = Compose([IAAAdditiveGaussianNoise(p=1)])(image=image, mask=mask)

        # image_aug = augmented['image']
        # mask_aug = augmented['mask']
        # visualize_aug(image_aug, mask_aug, original_image=image, original_mask=mask)
