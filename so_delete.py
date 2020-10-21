import sys, os
import cv2
import torch, glob
import numpy as np
import random
import time
import copy
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import nn
# import torch.
import segmentation_models_pytorch as smp
import zipfile
from dataset import get_test_loader
from tqdm import tqdm
from medpy import metric
import matplotlib.pyplot as plt
from statistics import mean
from image_dataset_viz import render_datapoint
import openslide as opsl
import lxml.etree as ET


def find_image(paths_list, names_list):
    files = []
    for i in paths_list:
        for j in names_list:
            if j in i:
                files.append(i)
    return files


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

def find_path(input_set, keyword):
    path = list()
    for directory in input_set:
        p = glob.glob(directory + '/*' + keyword)
        path.extend(p)
    return path


def test(test_loader,output_save_path,model):
    time_begin = time.time()
    model.eval()

    with torch.no_grad():
        print("Predicting cropped images...")
        for input,samples_name in tqdm(test_loader):
            input = input.cuda()
            d1 = model(input)
            batch_out = d1[:,0,:,:]
            batch_out = torch.sigmoid(batch_out)
            batch_out = batch_out.cpu().numpy()
            for out,sample_name in zip(batch_out,samples_name):
                ori_name = sample_name.split('_')[0]
                real_y = sample_name.split('_')[1]
                real_x = sample_name.split('_')[2]
                if not os.path.exists(output_save_path):
                    os.makedirs(output_save_path )
                save_path = output_save_path + '/{}_{}_{}.png'.format(ori_name, real_y, real_x)
                out = np.asarray((out * 255).clip(0, 255), np.uint8)

                mask = np.where(out < 255 * 0.1, 0, out)
                mask = np.where(mask >= 255 * 0.1, 255, mask)
                if np.sum(mask)==0:
                    continue
                assert len(out.shape) == 2
                cv2.imwrite(save_path,out)
                # print("processing",ori_name)
    print('Predicting crop images took {} mins'.format((time.time() - time_begin) // 60))

def prob_to_mask(prob_dir, mask_save_dir, th=0.5, pixelth=50,pad_white=False):

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
        # cont2 = []
        for i in range(len(cnts)):
            if cv2.contourArea(cnts[i]) < pixelth:
                cont.append(cnts[i])
            # else:
            #     cont2.append(cnts[i])

        mask_bgr = mask.copy()
        mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask_bgr, cont, -1, (0, 0, 0), -1)
        # cv2.drawContours(mask_bgr, cont2, -1, (255, 255, 255), -1)
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

    # print("Finding {} black masks".format(count_black))
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
    plt.savefig('./resnet34_threshold_{}_{}.jpg'.format(best_thr,round(score,2)))
    plt.show()

    plt.close()
    return float(best_thr), float(best_pixel_thr), float(score)


def crop_svs(svs_paths_list,save_dir,image_size=512,step=256):
    for svs_path in tqdm.tqdm(svs_paths_list):
        print("processing: ", svs_path)
        slide = opsl.OpenSlide(svs_path)
        patch_size = image_size
        w_count = int(slide.level_dimensions[0][0] // step)
        h_count = int(slide.level_dimensions[0][1] // step)
        for w in range(w_count - 3):
            for h in range(h_count - 3):
                subHIC1 = np.array(slide.read_region((w * step, h * step), 0, (patch_size, patch_size)))[:, :, :3]
                subHIC = cv2.cvtColor(subHIC1, cv2.COLOR_RGB2BGR)
                name = svs_path.split('/')[-1].split('.')[0] + '_' + str(w * step) + '_' + str(h * step)
                cv2.imwrite(save_dir + name + '.png', subHIC)

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
                dice_score = metric.binary.dc(pre,mask)
                if 'mask' in name:
                    name = name.split('_')[0]
                dice[name] = dice_score
    print("The mean dice is",mean(dice.values()))
    return dice

def visualize(image_name, image_dir, mask_dir, pre_dir,save_name,dice=None):

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
            rimg = render_datapoint(original_mask, original_image, blend_alpha=0.7)
            ax[i, 1].imshow(rimg)
            ax[i, 1].set_title('Original mask', fontsize=fontsize)
            pre_mask = cv2.imread(pres_path[i])
            pre_mask = cv2.cvtColor(pre_mask,cv2.COLOR_BGR2GRAY)
            ax[i, 2].imshow(pre_mask)
            ax[i, 2].set_title('pred mask'+' dice:'+str(round(dice_score,3)), fontsize=fontsize)
        plt.savefig(save_name+'.png')
        plt.show()

def xml_to_region(xml_file, color="16744448",level_downsample=1):
    """
    parse XML label file and get the points
    :param xml_file: xml file
    :param color: kind of labeled color
    :param shape: kind of labeled shape
    :return: region list,region_class
    """

    tree = ET.parse(xml_file)
    region_list = []
    region_class = []
    for col in tree.findall('.//Annotation'):
        if col.attrib['LineColor'] in color:
            for region in col.findall('./Regions/Region'):
                vertex_list = []
                # region.attrib.get('Type')=='0':
                region_class.append(region.attrib.get('Type'))
                for vertex in region.findall('.//Vertices/Vertex'):
                    # parse the 'X' and 'Y' for the vertex
                    vertex_list.append(vertex.attrib)
                region_list.append(vertex_list)
    point_list = []
    for __, region in enumerate(region_list):
        reg = []
        for __, point in enumerate(region):
            X, Y = int(float(point['X']) / level_downsample), int(float(point['Y']) / level_downsample)
            reg.append([X, Y])
        point_list.append(reg)
    return point_list

def statistic(image_path_list, xml_list, name_list):
    # tp_l,tn_l,fp_l,fn_l = [],[],[],[]
    result = []
    for xml_file in xml_list:
        for name in name_list:
            if name in xml_file:
                print("Processing,",name)
                find_label = 0 #
                not_find_label = 0 #
                find_wrong_label = 0 #

                pred_point = []
                pl = xml_to_region(xml_file)
                cen_point = []
                for list_area in pl:
                    np_area = np.array(list_area)
                    x, y, w, h = cv2.boundingRect(np_area)  # dawn rectangle in the image, return (top left piont, width, height)
                    x_c, y_c = int(x + (w / 2)), int(y + (h / 2))
                    cen_point.append([x_c, y_c])
                print("Total {} positive labels in {}".format(len(cen_point),name))
                for image_path in image_path_list:
                    if name in image_path:
                        img_name = image_path.split('/')[-1].split('.')[0]
                        assert img_name.split('_')[0] == name
                        start_x = int(img_name.split('_')[1])
                        start_y = int(img_name.split('_')[2])
                        point_status = 1
                        for c_point in cen_point:
                            x_c,y_c = c_point[0],c_point[1]
                            if start_x<x_c and x_c<start_x+256 and start_y<y_c and y_c<start_y+256:
                                pred_point.append([x_c,y_c])
                                point_status = 0
                                find_label+=1
                                continue
                        if point_status:
                                find_wrong_label+=1
                pred_point2 = list(set([tuple(t) for t in pred_point]))
                dup = len(pred_point)- len(pred_point2)
                not_find_label = len(cen_point) - len(pred_point2)
                result.append([find_label,find_wrong_label/2,not_find_label,dup])
                break
    return result

def mask_to_image(mask_list,image_dir,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    for mask in tqdm(mask_list):
        img = cv2.imread(image_dir+'/'+mask.split('/')[-1])
        cv2.imwrite(save_dir+'/'+mask.split('/')[-1],img)


def predict_test(CropStage=False,TestStage=True,toMask=True,toZip=True,Statistic=True,newTH=0.05):

    root_path = '/media/totem_disk/totem/weitang/MyProject'
    #
    model = smp.Unet('resnet34', activation=None).cuda()
    dir_model = root_path + '/unet_resnet34_1_1_best.pth'
    model.load_state_dict(torch.load(dir_model)['state_dict'])
    val_name_list = ['1024531', '1024624', '1023965']
    svs_list = glob.glob('/media/totem_disk/totem/weitang/data_handlabel/*svs')
    test_path_list = find_image(svs_list,val_name_list)
    crop_images_path = '/media/totem_disk/totem/weitang/MyProject/temp_data/crop_image'

    print("Total {} images for testing".format(len(test_path_list)))
    # 裁切程序
    if CropStage==True:
        print("Stage 1: ")
        #crop images
        crop_svs(test_path_list, crop_images_path,image_size=512,step=256)
    crop_images_path_list = glob.glob('/media/totem_disk/totem/weitang/MyProject/temp_data/crop_image/*png')

    crop_predict = root_path + '/temp_data_test/crop_predict'
    if TestStage==True:
        print("Stage 2: ")
        #predict cropped images
        test_images_path_list = crop_images_path_list
        os.makedirs(crop_predict, exist_ok=True)
        test_loader = get_test_loader(test_images_path_list, image_size=512, batch_size=4)
        test(test_loader, crop_predict, model=model)

    mask_save_path = root_path + '/temp_data_test/mask/'
    if toMask == True:
        print("Stage 4: ")
        #convert probs to masks
        os.makedirs(mask_save_path, exist_ok=True)
        prob_to_mask(prob_save_path, mask_save_dir=mask_save_path,th=newTH,pad_white=True)

    crop_preds_path_list = glob.glob('/media/totem_disk/totem/weitang/MyProject/temp_data_test/crop_predict/*png')
    crop_pred_toimage_path = '/media/totem_disk/totem/weitang/MyProject/temp_data_test/crop_predict_image'
    mask_to_image(crop_preds_path_list,image_dir = crop_images_path,save_dir = crop_pred_toimage_path)
    data_xml_list = glob.glob(r'/media/totem_disk/totem/weitang/data_handlabel/*.xml')
    if Statistic == True:
        result = statistic(crop_preds_path_list,data_xml_list,val_name_list)
        print(result)
    if toZip == True:
        print("Stage 5: ")
        #zip masks
        zf = zipfile.ZipFile(f'{root_path}/result/result.zip', 'w')
        for i in glob.glob(f"{mask_save_path}/*.png"):
            basename = os.path.split(i)[1]
            zf.write(i, f'result/{basename}')
        zf.close()

def check_validation(CropStage=False,TestStage=True,chooseTH=True,toMask=True):
    root_path = '/media/totem_disk/totem/weitang/MyProject'
    #
    model = smp.Unet('resnet34', activation=None).cuda()
    # model = U_Net2(img_ch=3, output_ch=1).cuda()
    # dir_model = root_path + '/U_Net2-30-0.1000-5-0.2494.pkl'
    dir_model = root_path + '/unet_resnet34_1_1_best.pth'
    model.load_state_dict(torch.load(dir_model)['state_dict'])
    # model.load_state_dict(torch.load(dir_model))

    dataset_path = '/media/totem_disk/totem/weitang/data_handlabel/sample_abnormal_256'
    path_list = glob.glob(dataset_path + '/*png')
    images_path_list = copy.deepcopy(path_list)
    masks_path_list = []
    for p in path_list:
        if 'mask' in p:
            masks_path_list.append(p)
            images_path_list.remove(p)

    val_name_list = ['1024531', '1024624', '1023965']
    val_files = find_image(images_path_list, val_name_list)
    val_masks_files = find_image(masks_path_list, val_name_list)
    for i in range(len(val_files)):
        images_path_list.remove(val_files[i])
        masks_path_list.remove(val_masks_files[i])
    test_path_list = val_files
    # 裁切测试集路径
    print("Total {} images for testing".format(len(test_path_list)))

    prob_save_path = root_path + '/temp_data/prob'
    crop_predict = root_path + '/temp_data/crop_predict'
    if TestStage==True:
        print("Stage 1: ")
        #predict cropped images
        os.makedirs(crop_predict, exist_ok=True)
        test_loader = get_test_loader(test_path_list, image_size=512, batch_size=1)
        test(test_loader, crop_predict, model=model)

    mask_save_path = root_path + '/temp_data/mask/'
    mask_ori_path = val_masks_files
    if chooseTH == True:
        bestTH, pixelTH, dice_score = choose_threshold(crop_predict, mask_ori_path)
    else:
        bestTH = 0.05
        pixelTH = 5
    if toMask == True:
        print("Stage 4: ")
        #convert probs to masks
        os.makedirs(mask_save_path, exist_ok=True)
        prob_to_mask(prob_save_path, mask_save_dir=mask_save_path, th = bestTH, pixelth =pixelTH)

    dice = calculate_dice(glob.glob(mask_save_path+'*png'),mask_ori_path)
    dice_sorted = sorted(dice.items(),key=lambda item:item[1])
    sorted_name = []
    for i in dice_sorted:
        sorted_name.append(i[0])
    lowest10 = sorted_name[:10]
    highest10 = sorted_name[-10:]
    image_dir = '/media/totem_disk/totem/weitang/competition/trainData/image'
    visualize(lowest10, image_dir, mask_ori_path, mask_save_path, save_name='lowest_dice',dice=dice)
    visualize(highest10, image_dir, mask_ori_path, mask_save_path, save_name= 'highest_dice',dice=dice)
    # visualize(highest10, image_dir, mask_ori_path, mask_save_path, save_name= 'biggest_size_dice',dice=dice)
    # visualize(highest10, image_dir, mask_ori_path, mask_save_path, save_name='smallest_dice', dice=dice)

def detect_abnormal(svs_paths_list,image_size=512,step=396):
    root_path = '/media/totem_disk/totem/weitang/MyProject'
    model = smp.Unet('resnet34', activation=None).cuda()
    # model = U_Net2(img_ch=3, output_ch=1).cuda()
    # dir_model = root_path + '/U_Net2-30-0.1000-5-0.2494.pkl'
    dir_model = root_path + '/unet_resnet34_1_1_best.pth'
    model.load_state_dict(torch.load(dir_model)['state_dict'])
    crop_predict = root_path + '/temp_data/crop_predict'


if __name__ == '__main__':

    predict_test(CropStage=False, TestStage=False, toMask=False, toZip=False,Statistic=True,newTH=0.05)
    # check_validation(TestStage=True, chooseTH=False, toMask=False)