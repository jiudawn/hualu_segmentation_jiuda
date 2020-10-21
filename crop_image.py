import cv2
import os,glob
import numpy as np
from tqdm import tqdm
import zipfile


def resize_pad(img_path, scale, image_size, value=255):
    img = cv2.imread(img_path)
    # resize in 0.25 scale
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    img_height, img_width = img.shape[0], img.shape[1]
    # pad image size to 512*512
    if img_height < image_size:
        img = np.pad(img, (
        ((image_size - img_height) // 2, image_size - img_height - (image_size - img_height) // 2), (0, 0), (0, 0)),
                     constant_values=(value, value))
    if img_width < image_size:
        img = np.pad(img, (
        (0, 0), ((image_size - img_width) // 2, image_size - img_width - (image_size - img_width) // 2), (0, 0)),
                     constant_values=(value, value))

    return img


def crop(ori_path, crop_set, mask_paths=None, scale=0.25, image_size=512, mode='test'):
    if not os.path.isdir(crop_set):
        os.makedirs(crop_set)
    if mode == 'train':
        os.makedirs(crop_set + '_mask', exist_ok=True)
        save_mask_path = crop_set + '_mask'
    save_path = crop_set
    print("crop images...")
    for img_path in tqdm(ori_path):
        name = img_path.split('/')[-1]
        if ('mask' in img_path) or ('black' in img_path):
            continue

        img_root_name, img_type = os.path.splitext(img_path)
        img_name = img_root_name.split('/')[-1]

        img = resize_pad(img_path, scale, image_size)
        img_height, img_width = img.shape[0], img.shape[1]

        if mode == 'train':
            mask_path = os.path.split(mask_paths[0])[0]+'/'+img_name + '_mask.jpg'
            mask = resize_pad(mask_path, scale, image_size, value=0)

        # crop with step= image_size/2
        step_size = image_size // 2
        for j in range(0, img_height, step_size):
            for k in range(0, img_width, step_size):
                x_start, y_start = k, j
                x_end, y_end = k + image_size, j + image_size
                # drop if size-start<step
                if img_width - x_start < step_size:
                    continue
                if img_height - y_start < step_size:
                    continue
                # if size-end<step
                if (img_width - x_end <= step_size) and (k != 0):
                    x_end = img_width
                    x_start = img_width - image_size

                if (img_height - y_end <= step_size) and (j != 0):
                    y_end = img_height
                    y_start = img_height - image_size

                x = img[y_start:y_end, x_start:x_end]
                # drop some images
                bm1 = np.all(x > np.reshape([215, 215, 215], [1, 1, 3]), -1)
                if bm1.sum(dtype=np.int) > bm1.shape[0] * bm1.shape[1] * 0.95:
                    continue

                assert x.shape[0] == x.shape[1] == image_size
                cv2.imwrite('{}/{}_{}_{}.jpg'.format(save_path, img_name, y_start, x_start), x)
                if mode == 'train':
                    x_mask = mask[y_start:y_end, x_start:x_end]
                    cv2.imwrite('{}/{}_{}_{}_mask.jpg'.format(save_mask_path, img_name, y_start, x_start), x_mask)
    print("crop finished")
if __name__ == '__main__':
    ori_path = glob.glob('/media/totem_disk/totem/weitang/competition/trainData_Big/image/*jpg')
    mask_paths = glob.glob('/media/totem_disk/totem/weitang/competition/trainData_Big/mask/*jpg')
    crop_set = '/media/totem_disk/totem/weitang/competition/0.4data768/image'
    crop(ori_path, crop_set, mask_paths=mask_paths, scale=0.4, image_size=768, mode='train')


    print("Stage 5: ")
    #zip masks
