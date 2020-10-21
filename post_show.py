import numpy as np
import glob
from PIL import Image
import cv2
import os
from PIL import Image
import cv2
import random
import pandas as pd
# pip install image-dataset-viz
from image_dataset_viz import render_datapoint
def find_image(train_files1, name1):
    val_files = []
    for i in train_files1:
        for j in name1:
            if j in i:
                val_files.append(i)
    return val_files

def read_image(path):
    img = cv2.imread(path)
    # mask = os.path.split()
    # mask = os.path.split(path)[-1]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    name = os.path.split(path)[-1].split('.')[0]
    mask = cv2.imread('/media/totem_disk/totem/weitang/competition/trainData_Big/mask/'+name+'_mask.jpg')
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
    mask[:,:,:2] = 0
    img2 = cv2.addWeighted(img,0.7,mask,0.3,0)
    return img2

def read_mask(path):
    # img = Image.open(path)
    # bk = Image.new('L', size=img.size)
    # g = Image.merge('RGB', (bk, img.convert('L'), bk))
    img = cv2.imread(path)
    # mask = os.path.split()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

from image_dataset_viz import DatasetExporter
# images_path = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/data3/image/*jpg'))
# masks_path = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/data3/mask/*jpg'))
# image_name = sorted(list(set([i.split('/')[-1].split('_')[0] for i in images_path])))
# random.seed(123)
# val_id = random.sample(range(585), 80)
# imid = [image_name[i] for i in val_id]
# val_files = pd.DataFrame(data=imid)
# val_files = find_image(images_path, imid)
# val_masks_files = find_image(masks_path, imid)
# for i in range(len(val_files)):
#     images_path.remove(val_files[i])
#     masks_path.remove(val_masks_files[i])
hard_sample = ['2450','2440','2469', '2412', '2505', '2593','2694','2511','2444','2524','2635','2669','2627','2520','2506', '2704',
'2216','2320','2456','2622','2280', '2688','2218','2289', '2690','2698','2467', '2534','2668','2287', '2393',
 '2557', '2720', '2023', '2572', '2422','2225','2033','2709','2370','2386','2537','2367', '2657', '2501', '2673'
, '2310','2701','2401', '2430','2309','2578','2067','2553','2389','2721','2573','2465','2294','2623','2490','2396',
 '2101','2702','2200','2439','2383','2728']

# predict_dir = '/media/totem_disk/totem/weitang/MyProject/temp_data/crop_predict2/'
prob_dir = '/media/totem_disk/totem/weitang/project/temp_data/prob_train/'
# predict_dir = '/media/totem_disk/totem/weitang/project/temp_data/prob'
# data_dir = '/media/totem_disk/totem/weitang/data_handlabel/sample_abnormal_512/'
data_dir = '/media/totem_disk/totem/weitang/competition/trainData_Big/image/'
mask_dir = '/media/totem_disk/totem/weitang/competition/trainData_Big/mask/'
images_list = glob.glob(data_dir+'*jpg')
images_sample_list = find_image(images_list, hard_sample)
# masks_sample_list = find_image(mask_dir, hard_sample)
# image_paths_list = glob.glob(data_dir+'/*png')
originals,predicts,masks = [],[],[]
for image_path in images_sample_list:
    name = os.path.split(image_path)[-1].split('.')[0]
    mask_path = mask_dir + name+'_mask.jpg'
    pred_path = prob_dir + name + '.png'
    # mask_path = data_dir + name.split('.png')[0]+'_mask.png'
    masks.append(pred_path)
    # masks.append(mask_path)
    originals.append(image_path)

de = DatasetExporter(read_image, read_mask, blend_alpha=0.3, n_cols=20, max_output_img_size=(512, 512))
de.export(originals, masks, "result_hard")