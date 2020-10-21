import argparse
import os,glob
import random
import json,codecs
from pprint import pprint
from argparse import Namespace
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
import pickle
from datetime import datetime
from torch.backends import cudnn
from solver import Train
from dataset import DatasetsStatic,get_loader
import pandas as pd

def find_image(train_files1, name1):
    val_files = []
    for i in train_files1:
        for j in name1:
            if j == i.split('/')[-1].split('_')[0]:
                val_files.append(i)
    return val_files

def main(config):
    cudnn.benchmark = True
    config.save_path = config.model_path + '/' + config.model_type
    if not os.path.exists(config.save_path):
        print('Making pth folder...')
        os.makedirs(config.save_path)

    # 打印配置参数，并输出到文件中
    pprint(config)
    if 'choose_threshold' not in config.mode:
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        with codecs.open(config.save_path + '/' + TIMESTAMP + '.json', 'w', "utf-8") as json_file:
            json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

    images_path = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/data3/image/*jpg'))
    masks_path = sorted(glob.glob('/media/totem_disk/totem/weitang/competition/data3/mask/*jpg'))
    image_name = sorted(list(set([i.split('/')[-1].split('_')[0] for i in images_path])))
    random.seed(123)
    val_id = random.sample(range(585), 80)
    imid = [image_name[i] for i in val_id]
    val_files = pd.DataFrame(data=imid)
    val_files.to_csv('val.csv')
    val_files = find_image(images_path, imid)
    val_masks_files = find_image(masks_path, imid)
    for i in range(len(val_files)):
        images_path.remove(val_files[i])
        masks_path.remove(val_masks_files[i])

    hard_sample = ['2450','2440','2469', '2412', '2505', '2593','2694','2511','2444','2524','2635','2669','2627','2520','2506', '2704',
'2216','2320','2456','2622','2280', '2688','2218','2289', '2690','2698','2467', '2534','2668','2287', '2393',
 '2557', '2720', '2023', '2572', '2422','2225','2033','2709','2370','2386','2537','2367', '2657', '2501', '2673'
, '2310','2701','2401', '2430','2309','2578','2067','2553','2389','2721','2573','2465','2294','2623','2490','2396',
 '2101','2702','2200','2439','2383','2728']
    images_sample_list = find_image(images_path,hard_sample)
    masks_sample_list = find_image(masks_path, hard_sample)
    index = 1
    # train_loader, val_loader = get_loader(images_path, masks_path,
    #                                       val_files, val_masks_files,
    #                                       config.image_size_stage1,
    #                                       config.batch_size_stage1, config.num_workers,
    #                                        # weights_sample=config.weight_sample
    #                                       )
    train_loader, val_loader = get_loader(images_sample_list, masks_sample_list,
                                          val_files, val_masks_files,
                                          config.image_size_stage1,
                                          config.batch_size_stage1, config.num_workers,
                                           # weights_sample=config.weight_sample
                                          )
    solver = Train(config, train_loader, val_loader)
    # 针对不同mode，在第一阶段的处理方式
    if config.mode == 'train' or config.mode == 'train_stage1':
        solver.train(index)

    # 对于第二个阶段的处理方法
    train_loader_stage2, val_loader_stage2 = get_loader(images_path, masks_path,
                                          val_files, val_masks_files,
                                                        config.image_size_stage2,
                                                        config.batch_size_stage2, config.num_workers,
                                                        # weights_sample=config.weight_sample
                                                        )
    # 更新类的训练集以及验证集
    solver.train_loader, solver.valid_loader = train_loader_stage2, val_loader_stage2
    # 针对不同mode，在第二阶段的处理方式
    if config.mode == 'train' or config.mode == 'train_stage2' :
        solver.train_stage2(index)

    del train_loader_stage2, val_loader_stage2

    print('save the result')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # stage 1 hyper-parameters
    parser.add_argument('--image_size_stage1', type= int,default=512, help='image size in the 1st stage')
    parser.add_argument('--batch_size_stage1',type =int, default=6, help= 'batch size in the 1st stage')
    parser.add_argument('--epoch_stage1',type = int,default=30,help='how many epoch in the 1st stage')
    parser.add_argument('--epoch_stage1_freeze',type=int,default=0,help='how many epoch freezes the encoder layer in the 1st stage')

    #stage 2 hyper-parameters
    parser.add_argument('--image_size_stage2',type=int,default=512,help='image size in the 2nd stage')
    parser.add_argument('--batch_size_stage2',type=int,default=3,help='batch size in the 2nd stage')
    parser.add_argument('--epoch_stage2',type=int,default=200,help='how many epoch in the 2nd stage')
    parser.add_argument('--epoch_stage2_accumulation',type=int,default=0,help='how many epoch gradients accumulation in the 2nd stage')
    parser.add_argument('--accumulation_steps',type=int,default=10,help='how many steps do you add up to the gradient in the 2nd stage')

    #model set'][p;/l.omk
    parser.add_argument('--resume',type=str,default='unet_densenet161_1_2_best.pth',help='if has value, must be the name of Weight file')

    parser.add_argument('--mode',type=str,default='train_stage2',\
                        help = 'train/train_stage1/train_stage2/')
    parser.add_argument('--model_type',type=str,default='unet_densenet161',\
                        help = 'unet_resnet34/linknet/unet_densenet161/pspnet/resnet34/unet_densenet121/unet_efficientnet-b4/unet_se_resnext101_32x4d')

    #model hyper-parameters
    parser.add_argument('--t',type=int,default=3,help='t for recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--img_ch',type=int,default=3)
    parser.add_argument('--output_ch',type=int,default=1)
    parser.add_argument('--num_workers',type=int,default=3)
    parser.add_argument('--lr',type=float,default=2e-4,help='initialize in stage 1')
    parser.add_argument('--lr-stage2',type=float,default=5e-5,help='init lr in stage 2')
    parser.add_argument('--weight_decay',type = float,default=1e-4,help= 'weight_decay in optimizer')
    parser.add_argument('--momentum',type = float,default=0.9,help= 'momentum in optimizer')

    #dataset
    parser.add_argument('--model_path',type=str,default='./checkpoints')

    parser.add_argument('--dataset_root',type=str,default='/media/totem_disk/totem/weitang/competition/data3')
    parser.add_argument('--train_dir',type=str,default='/media/totem_disk/totem/weitang/competition/data3/image')
    parser.add_argument('--mask_dir',type=str,default='/media/totem_disk/totem/weitang/competition/data3/mask')
    parser.add_argument('--weight_sample',type=list,default=0,help='sample weight of class')

    config = parser.parse_args()
    main(config)