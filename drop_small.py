import glob
import cv2
import os

image_dir ='/media/totem_disk/totem/weitang/competition/trainData_Big/image/'
mask_dir ='/media/totem_disk/totem/weitang/competition/trainData_Big/mask/'
root_path = '/media/totem_disk/totem/weitang/competition/trainData_Big'
images_path = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/trainData/image/*jpg'))
masks_path_list = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/trainData/mask/*jpg'))
image_name = sorted(list(set([i.split('/')[-1].split('_')[0] for i in images_path])))
image_name_samll = []
for mask_path in masks_path_list:
        name = os.path.splitext(mask_path)[0].split('/')[-1]
#     if name in mask_path:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[0]*mask.shape[1]>1349120:
            cv2.imwrite(image_dir+name+'_mask.png',mask)
            image = cv2.imread(root_path + '/image/'+name+'.png')
            cv2.imwrite(image_dir+name+'.png',image)
        else:
            image_name_samll.append(name)