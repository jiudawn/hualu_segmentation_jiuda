import sys, os
import cv2
import torch, glob
import numpy as np
import random
import time
from crop_image import crop
import ttach as tta
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import nn
import segmentation_models_pytorch as smp
import zipfile
from dataset2 import get_test_loader
from tqdm import tqdm
from medpy import metric
import matplotlib.pyplot as plt
from statistics import mean
from image_dataset_viz import render_datapoint
import yaml

def find_image(paths_list, names_list):
    files = []
    for i in paths_list:
        for j in names_list:
            if j in i:
                files.append(i)
    return files

def find_path(input_set, keyword):
    path = list()
    for directory in input_set:
        p = glob.glob(directory + '/*' + keyword)
        path.extend(p)
    return path


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn

def test(test_loader,output_save_path,model):
    time_begin = time.time()
    model.eval()

    with torch.no_grad():
        print("Predicting cropped images...")
        for input,samples_name in tqdm(test_loader):
            input = input.cuda()
            d1 = model(input)
            batch_out = d1[:,0,:,:]
            batch_out = batch_out.cpu().numpy()
            for out,sample_name in zip(batch_out,samples_name):
                ori_name = sample_name.split('_')[0]
                real_y = sample_name.split('_')[1]
                real_x = sample_name.split('_')[2]
                # if not os.path.exists(output_save_path + '/{}'.format(ori_name)):
                #     os.makedirs(output_save_path + '/{}'.format(ori_name))
                # save_path = output_save_path + '/{}/{}_{}.png'.format(ori_name, real_y, real_x)
                if not os.path.exists(output_save_path):
                    os.makedirs(output_save_path)
                save_path = output_save_path + '/{}_{}_{}.png'.format(ori_name, real_y, real_x)
                out = np.asarray((out * 255).clip(0, 255), np.uint8)
                assert len(out.shape) == 2
                cv2.imwrite(save_path,out)
                # print("processing",ori_name)

    print('Predicting cropped images took {} mins.'.format((time.time() - time_begin) // 60))

def merge_hot_pic(imgs_path,hot_pic_path,scale,save_dir,img_size=512):
    time_begin = time.time()
    print("Merging images begin...")
    for img_path in tqdm(imgs_path):
        img_name = os.path.splitext(img_path)[0].split('/')[-1]
        hot_imgs = glob.glob(hot_pic_path + '/' + img_name + '_*.png')
        ori_img = cv2.imread(img_path)
        hot_pic_height = int(ori_img.shape[0] * scale)
        hot_pic_width = int(ori_img.shape[1] * scale)
        to_img = np.zeros((hot_pic_height + img_size * 2, hot_pic_width + img_size * 2), dtype=np.uint16)
        mask = np.zeros((hot_pic_height + img_size * 2, hot_pic_width + img_size * 2), dtype=np.uint16) + 1e-8
        for i, each_img_path in enumerate(hot_imgs):
            img_y, img_x = int(os.path.splitext(each_img_path)[0].split('/')[-1].split('_')[1]), int(
                os.path.splitext(each_img_path)[0].split('/')[-1].split('_')[2])
            from_img = cv2.imread(each_img_path, 0)
            if hot_pic_height < img_size:
                from_img = from_img[(img_size-hot_pic_height)//2:hot_pic_height+(img_size-hot_pic_height)//2,:]
            if hot_pic_width < img_size:
                from_img = from_img[:,(img_size-hot_pic_width)//2:hot_pic_width+(img_size-hot_pic_width)//2]
            from_img_height,from_img_width = from_img.shape[0],from_img.shape[1]
            from_img = from_img.astype(np.uint16)
            roi = to_img[img_y + img_size:(img_y + img_size + from_img_height), img_x + img_size:(img_x + img_size + from_img_width)].astype(np.uint16)
            little_merge_img = from_img + roi
            mask[img_y + img_size:img_y + img_size + from_img_height, img_x + img_size:img_x + img_size + from_img_width] += 1
            to_img[img_y + img_size:img_y + img_size + from_img_height, img_x + img_size:img_x + img_size + from_img_width] = little_merge_img
        to_img = (to_img / mask).astype(np.uint8)
        new_to_img = to_img[img_size:hot_pic_height + img_size, img_size:hot_pic_width + img_size].astype(np.uint8)
        ori_height,ori_width = ori_img.shape[0],ori_img.shape[1]
        new_to_img = cv2.resize(new_to_img,(ori_width,ori_height))
        save_path = save_dir +'/'+ img_name + '.png'
        assert len(new_to_img.shape) == 2
        assert new_to_img.shape[0] == ori_height,new_to_img.shape[1] == ori_width
        cv2.imwrite(save_path,new_to_img)

    print('Merging images took:{} mins'.format((time.time() - time_begin) // 60))

def prob_to_mask(prob_dir, mask_save_dir, th=0.5, pixelth=1000,pad_white=False):

    time_begin = time.time()
    print("Converting probs to masks ...")
    all_hot_pic = glob.glob(prob_dir+'/*png')
    for hot_pic in tqdm(all_hot_pic):
        img_name = os.path.split(hot_pic)[-1].split('.png')[0]
        mask = cv2.imread(hot_pic,0)
        # mask = np.where(mask < 255*0.33, 0, mask)
        # mask = np.where(mask >= 255*0.33, 255, mask)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
        # mask = cv2.dilate(mask, kernel)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

        mask = np.where(mask < 255*th, 0, mask)
        mask = np.where(mask >= 255*th, 255, mask)
        #
        cnts, _ = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
        cont = []
        cont2 = []
        for i in range(len(cnts)):
            if cv2.contourArea(cnts[i]) < mask.shape[0]*mask.shape[1]*0.008:
                cont.append(cnts[i])
            else:
                cont2.append(cnts[i])

        mask_bgr = mask.copy()
        mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask_bgr, cont, -1, (0, 0, 0), -1)
        cv2.drawContours(mask_bgr, cont2, -1, (255, 255, 255), -1)
        new_mask = cv2.cvtColor(mask_bgr,cv2.COLOR_BGR2GRAY)
        # new_mask = mask
        if pad_white:
            compute_mask_area = np.sum(new_mask)
            count_black= 0
            if compute_mask_area < 500:
                new_mask = np.ones_like(new_mask)*255   # blank white
                count_black += 1
        save_path = mask_save_dir + '/' + img_name + '_mask.png'
        cv2.imwrite(save_path,new_mask)

    print("Finding {} black masks".format(count_black))
    print("Postprocessing images finished and took:{} mins".format((time.time() - time_begin) // 60))


def choose_threshold(probs_dir, masks_dir):

    # 先大概选取阈值范围
    dices_big = []
    thrs_big = np.arange(0.1, 1, 0.1)  # 阈值列表
    for th in thrs_big:
        all_hot_pic_list = glob.glob(probs_dir + '/*png')
        dice ={}
        for hot_pic in tqdm(all_hot_pic_list):
            img_name = os.path.split(hot_pic)[-1].split('.png')[0]
            hot_picture = cv2.imread(hot_pic, 0)
            hot_picture = np.where(hot_picture < 255 * th, 0, hot_picture)
            pre_mask = np.where(hot_picture >= 255 * th, 255, hot_picture)
            for mask_path in masks_dir:
                if img_name in mask_path:
                    mask = cv2.imread(mask_path)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    dice_score = metric.binary.dc(pre_mask, mask)
                    dice[img_name] = dice_score
        print("The mean dice is {} when threshold = {}".format(mean(dice.values()),th))
        dices_big.append(mean(dice.values()))
    dices_big = np.array(dices_big)
    best_thrs_big = thrs_big[dices_big.argmax()]

    # 精细选取范围
    dices_little = []
    thrs_little = np.arange(best_thrs_big - 0.05, best_thrs_big + 0.05, 0.01)  # 阈值列表
    for th in thrs_little:
        all_hot_pic_list = glob.glob(probs_dir + '/*png')
        dice = {}
        for hot_pic in tqdm(all_hot_pic_list):
            img_name = os.path.split(hot_pic)[-1].split('.png')[0]
            hot_picture = cv2.imread(hot_pic, 0)
            hot_picture = np.where(hot_picture < 255 * th, 0, hot_picture)
            pre_mask = np.where(hot_picture >= 255 * th, 255, hot_picture)
            for mask_path in masks_dir:
                if img_name in mask_path:
                    mask = cv2.imread(mask_path)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    dice_score = metric.binary.dc(pre_mask, mask)
                    dice[img_name] = dice_score
        print("The mean dice is {} when threshold = {}".format(mean(dice.values()), th))
        dices_little.append(mean(dice.values()))
    dices_little = np.array(dices_little)
    # score = dices.max()
    best_thr = thrs_little[dices_little.argmax()]

    dices_pixel = []
    pixel_thrs = np.arange(0, 4096, 256)  # 阈值列表
    for pixel_thr in pixel_thrs:
        for hot_pic in tqdm(all_hot_pic_list):
            img_name = os.path.split(hot_pic)[-1].split('.png')[0]
            hot_picture = cv2.imread(hot_pic, 0)
            hot_picture = np.where(hot_picture < 255 * th, 0, hot_picture)
            pre_mask = np.where(hot_picture >= 255 * th, 255, hot_picture)
            for mask_path in masks_dir:
                if img_name in mask_path:
                    mask = cv2.imread(mask_path)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    cnts, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                    cont = []

                    for i in range(len(cnts)):
                        if cv2.contourArea(cnts[i]) < pixel_thr:
                            cont.append(cnts[i])

                    mask_bgr = mask.copy()
                    mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(mask_bgr, cont, -1, (0, 0, 0), -1)
                    new_mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                    dice_score = metric.binary.dc(pre_mask, new_mask)
                    dice[img_name] = dice_score
        print("The mean dice is {} when pixel threshold = {}".format(mean(dice.values()), pixel_thr))
        dices_pixel.append(mean(dice.values()))
    dices_pixel = np.array(dices_pixel)
    score = dices_pixel.max()
    best_pixel_thr = pixel_thrs[dices_pixel.argmax()]

    print('best_thr:{}, best_pixel_thr:{}, score:{}'.format(best_thr, best_pixel_thr, score))

    plt.figure(figsize=(10.4, 4.8))
    plt.subplot(1, 3, 1)
    plt.title('Large-scale search')
    plt.plot(thrs_big, dices_big)
    plt.subplot(1, 3, 2)
    plt.title('Little-scale search')
    plt.plot(thrs_little, dices_little)
    plt.subplot(1, 3, 3)
    plt.title('pixel thrs search')
    plt.plot(pixel_thrs, dices_pixel)
    plt.savefig('./threshold_{}_{}.jpg'.format(best_thr,round(score,2)))
    # plt.show()

    plt.close()
    return float(best_thr), float(best_pixel_thr), float(score)

def calculate_dice(pres_path_list,masks_path_list):
    print("Calcalating dice...")
    dice = {}
    for pre_path in tqdm(pres_path_list):
        name = os.path.splitext(pre_path)[0].split('/')[-1]
        pre = cv2.imread(pre_path)
        pre = cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY)
        for mask_path in masks_path_list:
            if name in mask_path:
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if np.sum(mask)<20:
                    continue
                dice_score = metric.binary.dc(pre,mask)
                if 'mask' in name:
                    name = name.split('_')[0]
                dice[name] = dice_score
    print("The mean dice is",mean(dice.values()))
    return dice

def find_imagenames(directory, names, sorted_flag=True):
    paths = glob.glob(directory + '/*jpg')
    if len(paths) == 0:
        paths = glob.glob(directory + '/*png')
    assert len(paths) != 0
    files = []
    for i in paths:
        for j in names:
            if j in i:
                files.append(i)
    if sorted_flag:
        files = sorted(files)
    return files

def visualize(image_name, image_dir, mask_dir, pre_dir,save_name,dice=None):
    print("The {} visualization".format(save_name))
    fontsize = 8
    if 'mask' in image_name[0]:
        image_name = [i.split('_')[0] for i in image_name]
    length = len(image_name)
    images_path = find_imagenames(image_dir,image_name)
    masks_path = find_imagenames(mask_dir,image_name)
    pres_path = find_imagenames(pre_dir,image_name)

    f, ax = plt.subplots(length, 3, figsize=(10, 40))
    if dice:
        for i in range(length):
            name = images_path[i].split('/')[-1].split('.')[0]
            original_image = cv2.imread(images_path[i])
            original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
            dice_score = dice[name]
            w,h = original_image.shape[:2]
            ax[i, 0].imshow(original_image)
            ax[i, 0].set_title('Original image '+name +' '+str(w)+'*'+str(h), fontsize=fontsize)
            original_mask = cv2.imread(masks_path[i])
            # original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2GRAY)
            rimg = render_datapoint(original_mask, original_image, blend_alpha=0.8)
            ax[i, 1].imshow(rimg)
            ax[i, 1].set_title('Original mask', fontsize=fontsize)
            pre_mask = cv2.imread(pres_path[i])
            # pre_mask = cv2.cvtColor(pre_mask,cv2.COLOR_BGR2GRAY)
            rimg2 = render_datapoint(pre_mask, original_image, blend_alpha=0.8)
            ax[i, 2].imshow(rimg2)
            ax[i, 2].set_title('pred mask'+' dice:'+str(round(dice_score,3)), fontsize=fontsize)
        plt.savefig(save_name+'.png')
        plt.show()

def predict_test(CropStage=False,TestStage=True,toMask=True,toZip=True,newTH=0.05):
    # os.environ["CUDA_VISIBLE_DIVICES"] ="1"

    root_path = '/media/totem_disk/totem/weitang/project'
    # model = smp.Unet('se_resnext101_32x4d', activation=None).cuda()
    # i_size=512
    # i_scale=0.25
    # dir_model = root_path + '/model/unet_se_resnext101_32x4d_2_1_best.pth'
    # model.load_state_dict(torch.load(dir_model)['state_dict'])

    model = smp.Unet('densenet161', activation=None).cuda()
    i_size=512
    i_scale = 0.25
    dir_model = root_path + '/model/unet_densenet161_2_1_best_0.73.pth'
    model.load_state_dict(torch.load(dir_model)['state_dict'])
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        # tta.Scale(scales=[1,2,4])
        # tta.Rotate90(angles=[0,180])
    ])
    tta_model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')


    # model = smp.Unet('resnet34', activation=None).cuda()
    # dir_model = root_path + '/model/unet_resnet34_1_1_best.pth'

    # 裁切测试集路径
    # crop_test_images_path = root_path + '/temp_data_test/0.4crop_test_set1024'
    crop_test_images_path = root_path + '/temp_data_test/crop_test_set'

    test_path_list = glob.glob('/media/totem_disk/totem/weitang/competition/test2/test/*jpg')
    print("Total {} images for testing.".format(len(test_path_list)))
    # 裁切程序
    if CropStage==True:
        print("Stage 1: ")
        #crop images
        crop(test_path_list, crop_test_images_path,scale=i_scale,image_size=i_size,mode="test")

    # prob_save_path = root_path + '/temp_data_test/resprob1024'
    prob_save_path = root_path + '/temp_data_test/dense101_t'
    crop_predict = root_path + '/temp_data_test/predict512_resnet101_t'

    # crop_predict = root_path + '/temp_data_test/crop_predict_1024'
    if TestStage==True:
        print("Stage 2: ")
        #predict cropped images
        test_images_path_list = glob.glob(crop_test_images_path + '/*.jpg')
        os.makedirs(crop_predict, exist_ok=True)
        test_loader = get_test_loader(test_images_path_list, image_size=i_size, batch_size=2)
        test(test_loader, crop_predict, model=tta_model)

        print("Stage 3: ")
        #merge predicted images
        # prob_save_path = root_path + '/temp_data_test/prob'
    # 缩放倍率:0.25，即将原图*0.25再进行裁切
        os.makedirs(prob_save_path, exist_ok=True)
        merge_hot_pic(test_path_list, crop_predict, i_scale, prob_save_path)

    mask_save_path = root_path + '/temp_data_test/mask/'
    if toMask == True:
        print("Stage 4: ")
        #convert probs to masks
        os.makedirs(mask_save_path, exist_ok=True)
        prob_to_mask(prob_save_path, mask_save_dir=mask_save_path,th=newTH,pad_white=True)

    if toZip == True:
        print("Stage 5: ")
        #zip masks
        zf = zipfile.ZipFile(f'{root_path}/result/result.zip', 'w')
        for i in glob.glob(f"{mask_save_path}/*.png"):
            basename = os.path.split(i)[1]
            zf.write(i, f'result/{basename}')
        zf.close()

def find_highest_lowest10(dice_sorted):
    sorted_name = []
    for i in dice_sorted:
        sorted_name.append(i[0])
    lowest10 = sorted_name[:10]
    highest10 = sorted_name[-10:]
    return highest10,lowest10

def sortedsize(masks_path_list):
    print("Sorting masks by size...")
    size = {}
    for mask_path in tqdm(masks_path_list):
        name = os.path.splitext(mask_path)[0].split('/')[-1]
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        area = mask.shape[0]*mask.shape[1]
        size[name] = area
    size_sorted = sorted(size.items(), key=lambda item: item[1])
    sorted_name = []
    for i in size_sorted:
        sorted_name.append(i[0])
    lowest10 = sorted_name[:10]
    highest10 = sorted_name[-10:]
    return highest10,lowest10

def check_validation(CropStage=False,TestStage=True,chooseTH=True,toMask=True):
    print("Checking validation dataset and finding the best threshold...")
    root_path = '/media/totem_disk/totem/weitang/project'
    # model = smp.Unet('densenet161', activation=None).cuda()
    i_size=512
    i_scale=0.25
    model = smp.Unet('se_resnext101_32x4d', activation=None).cuda()
    dir_model = root_path + '/model/unet_se_resnext101_32x4d_2_1_best.pth'
    # dir_model = root_path + '/model/unet_densenet161_1_1_best0.70.pth'
    model.load_state_dict(torch.load(dir_model)['state_dict'])
    # model = smp.Unet('densenet161', activation=None).cuda()
    # dir_model = root_path + '/model/unet_densenet161_1_1_best0.65.pth'
    # model.load_state_dict(torch.load(dir_model)['state_dict'])

    images_path = os.listdir('/media/totem_disk/totem/weitang/competition/data3/image')
    masks_path = os.listdir('/media/totem_disk/totem/weitang/competition/data3/mask')
    img_dir = '/media/totem_disk/totem/weitang/competition/trainData_Big/image'
    ms_dir = '/media/totem_disk/totem/weitang/competition/trainData_Big/mask'

    random.seed(123)
    val_id = random.sample(range(585), 80)
    image_name = sorted(list(set([i.split('/')[-1].split('_')[0] for i in images_path])))
    imid = [image_name[i] for i in val_id]
    masks_path = find_path([ms_dir], 'jpg')
    images_path = find_path([img_dir], 'jpg')
    val_paths = find_image(images_path, imid)
    val_paths_masks = find_image(masks_path, imid)
    for i in range(len(val_paths)):
        images_path.remove(val_paths[i])
        masks_path.remove(val_paths_masks[i])

    crop_test_images_path = root_path + '/temp_data/crop_train_set'
    test_path_list = images_path
    # 裁切测试集路径
    print("Total {} images for test.".format(len(test_path_list)))
    # 裁切程序
    if CropStage==True:
        print("Stage 1: ")
        #crop images
        crop(test_path_list, crop_test_images_path,image_size=i_size,scale=i_scale,mode="test")

    prob_save_path = root_path + '/temp_data/prob_train'
    crop_predict = root_path + '/temp_data/crop_predict_train'
    if TestStage==True:
        print("Stage 2: ")
        #predict cropped images
        test_images_path_list = glob.glob(crop_test_images_path + '/*.jpg')
        os.makedirs(crop_predict, exist_ok=True)
        test_loader = get_test_loader(test_images_path_list, image_size=i_size, batch_size=4)
        test(test_loader, crop_predict, model=model)

        print("Stage 3: ")
        #merge predicted images
        # prob_save_path = root_path + '/temp_data_test/prob'
        os.makedirs(prob_save_path, exist_ok=True)
        merge_hot_pic(test_path_list, crop_predict, i_scale, prob_save_path,img_size=i_size)

    mask_save_path = root_path + '/temp_data/mask_train/'
    mask_ori_path = ms_dir
    if chooseTH == True:
        bestTH, pixelTH, dice_score = choose_threshold(prob_save_path, glob.glob(mask_ori_path+'/*jpg'))
    else:
        bestTH = 0.5
        pixelTH = 40
    if toMask == True:
        print("Stage 4: ")
        #convert probs to masks
        os.makedirs(mask_save_path, exist_ok=True)
        prob_to_mask(prob_save_path, mask_save_dir=mask_save_path, th = bestTH, pixelth =pixelTH, pad_white=True)

    dice = calculate_dice(glob.glob(mask_save_path+'*png'),glob.glob(mask_ori_path+'/*jpg'))
    # dice2 = calculate_dice(glob.glob(crop_predict + '/*png'), glob.glob(root_path + '/temp_data/crop_test_set_mask/*jpg'))
    # dice_sorted2 = sorted(dice2.items(), key=lambda item: item[1])
    dice_sorted = sorted(dice.items(),key=lambda item:item[1])
    # h10,l10 = find_highest_lowest10(dice_sorted2)
    highest10,lowest10 = find_highest_lowest10(dice_sorted)
    # big10,small10 = sortedsize(glob.glob(mask_save_path+'/*png'))

    image_dir = '/media/totem_disk/totem/weitang/competition/trainData/image'
    # visualize(l10, crop_test_images_path, root_path + '/temp_data/crop_test_set_mask/', crop_predict, save_name='lowest_dice',dice=dice2)
    # visualize(h10, crop_test_images_path, root_path + '/temp_data/crop_test_set_mask/', crop_predict, save_name='highest_dice',dice=dice2)
    mask_save_path = root_path + '/temp_data/mask'
    # visualize(lowest10, image_dir, mask_ori_path, mask_save_path, save_name='lowest_dice',dice=dice)
    # visualize(highest10, image_dir, mask_ori_path, mask_save_path, save_name= 'highest_dice',dice=dice)
    # visualize(big10, image_dir, mask_ori_path, mask_save_path, save_name= 'biggest_size_dice',dice=dice)
    # visualize(small10, image_dir, mask_ori_path, mask_save_path, save_name='smallest_dice', dice=dice)
    yaml.safe_dump(dice_sorted,open('dice_sorted.yml','w'))
    print(dice_sorted)

if __name__ == '__main__':

    predict_test(CropStage=False, TestStage=True, toMask=False, toZip=False,newTH=0.3)
    # check_validation(CropStage=False, TestStage=False, chooseTH=False, toMask=False)