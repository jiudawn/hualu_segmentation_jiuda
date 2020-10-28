import os, shutil
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils import write_txt
import csv
import matplotlib.pyplot as plt
from utils import NoamLR

plt.switch_backend('agg')
import seaborn as sns
import tqdm
from loss import GetLoss, RobustFocalLoss2d, BCEDiceLoss, SoftBCEDiceLoss, SoftBceLoss,SoftDiceLoss
from Losses import ComboLoss,LovaszLossSigmoid,LovaszLoss,JaccardLoss
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
import pandas as pd

# os.environ["CUDA_VISIBLE_DIVECES"] = "1,2"

class Train(object):
    def __init__(self, config, train_loader, valid_loader):
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = SoftBCEDiceLoss(weight=[0.33, 0.67])
        # self.criterion = LovaszLossSigmoid()

        # self.criterion = torch.nn.BCELoss()
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(50))

        # self.criterion_stage2 = SoftDiceLoss(weight=[0.33, 0.67])
        self.criterion_stage2 = SoftBCEDiceLoss(weight=[0.3, 0.7])
        # self.criterion_stage2 = LovaszLossSigmoid()
        self.model_type = config.model_type
        self.t = config.t

        self.mode = config.mode
        self.resume = config.resume

        # Hyper-parameters
        self.lr = config.lr
        self.lr_stage2 = config.lr_stage2
        self.start_epoch, self.max_dice = 0, 0
        self.weight_decay = config.weight_decay
        self.weight_decay_stage2 = config.weight_decay
        self.momentum = config.momentum
        # save set
        self.save_path = config.save_path
        if 'choose_threshold' not in self.mode:
            TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
            self.writer = SummaryWriter(log_dir=self.save_path + '/' + TIMESTAMP)

        # 配置参数
        self.epoch_stage1 = config.epoch_stage1
        self.epoch_stage1_freeze = config.epoch_stage1_freeze
        self.epoch_stage2 = config.epoch_stage2
        self.epoch_stage2_accumulation = config.epoch_stage2_accumulation
        self.accumulation_steps = config.accumulation_steps

        # 模型初始化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        print("Using model: {}".format(self.model_type))
        """Build generator and discriminator."""

        if self.model_type == 'unet_resnet34':
            # self.unet = Unet(backbone_name='resnet34', pretrained=True, classes=self.output_ch)
            self.unet = smp.Unet('resnet34', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_resnet50':
            self.unet = smp.Unet('resnet50', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_se_resnext50_32x4d':
            self.unet = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_densenet121':
            self.unet = smp.Unet('densenet121', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_densenet161':
            self.unet = smp.Unet('densenet161', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_efficientnet-b4':
            self.unet = smp.Unet('efficientnet-b4', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_xception':
            self.unet = smp.Unet('xception', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_vgg19_bn':
            self.unet = smp.Unet('vgg19_bn', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'pspnet_resnet34':
            self.unet = smp.PSPNet('resnet34', encoder_weights='imagenet', classes=1, activation=None)
        elif self.model_type == 'unet_resnet50_32x4d':
            self.unet = smp.Unet('resnext50_32x4d', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_densenet201':
            self.unet = smp.Unet('densenet201', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_timm-efficientnet-b1':
            self.unet = smp.Unet('timm-efficientnet-b1', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_se_resnext101_32x4d':
            self.unet = smp.Unet('se_resnext101_32x4d', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_timm-efficientnet-b6':
            self.unet = smp.Unet('timm-efficientnet-b6', encoder_weights='imagenet', activation=None)
        elif self.model_type == 'unet_dpn98':
            self.unet = smp.Unet('dpn98', encoder_weights='imagenet', activation=None)

        if torch.cuda.is_available():
            self.unet = torch.nn.DataParallel(self.unet)
            # os.environ["CUDA_VISIBLE_DIVECES"] = "2"
            self.criterion = self.criterion.cuda()
            self.criterion_stage2 = self.criterion_stage2.cuda()
        self.unet.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def save_checkpoint(self, state, stage, index, is_best):
        # 保存权重，每一epoch均保存一次，若为最优，则复制到最优权重；index可以区分不同的交叉验证
        pth_path = os.path.join(self.save_path, '%s_%d_%d.pth' % (self.model_type, stage, index))
        torch.save(state, pth_path)
        if is_best:
            print('Saving Best Model.')
            write_txt(self.save_path, 'Saving Best Model.')
            shutil.copyfile(os.path.join(self.save_path, '%s_%d_%d.pth' % (self.model_type, stage, index)),
                            os.path.join(self.save_path, '%s_%d_%d_best.pth' % (self.model_type, stage, index)))

    def load_checkpoint(self, load_optimizer=True):
        # Load the pretrained Encoder
        weight_path = os.path.join(self.save_path, self.resume)
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path)
            # 加载模型的参数，学习率，优化器，开始的epoch，最小误差等
            if torch.cuda.is_available:
                self.unet.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.unet.load_state_dict(checkpoint['state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.max_dice = checkpoint['max_dice']
            if load_optimizer:
                self.lr = checkpoint['lr']
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            print('%s is Successfully Loaded from %s' % (self.model_type, weight_path))
            write_txt(self.save_path, '%s is Successfully Loaded from %s' % (self.model_type, weight_path))
        else:
            raise FileNotFoundError("Can not find weight file in {}".format(weight_path))

    def train(self, index):
        # for param in self.unet.module.encoder.parameters():
        #     param.requires_grad = False
        # self.optimizer = optim.Adam(filter(lambda p:p.requires_grad, self.unet.module.parameters()),self.lr,weight_decay=self.weight_decay)

        self.optimizer = optim.Adam(self.unet.module.parameters(), self.lr, weight_decay=self.weight_decay)
        # 若训练到一半暂停了，则需要加载之前训练的参数，并加载之前学习率 TODO:resume学习率没有接上，所以resume暂时无法使用
        # if self.resume:
        #     self.load_checkpoint(load_optimizer=True)
        #     '''
        #     CosineAnnealingLR：若存在['initial_lr']，则从initial_lr开始衰减；
        #     若不存在，则执行CosineAnnealingLR会在optimizer.param_groups中添加initial_lr键值，其值等于lr
        #     重置初始学习率，在load_checkpoint中会加载优化器，但其中的initial_lr还是之前的，所以需要覆盖为self.lr，让其从self.lr衰减
        #     '''
        #     self.optimizer.param_groups[0]['initial_lr'] = self.lr

        stage1_epoches = self.epoch_stage1 - self.start_epoch
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,30)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=5, verbose=True,
        #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                            eps=1e-08)
        # 防止训练到一半暂停重新训练，日志被覆盖
        global_step_before = self.start_epoch * len(self.train_loader)

        for epoch in range(self.start_epoch, self.epoch_stage1):
            epoch += 1
            self.unet.train(True)

            # 学习率重启

            if epoch>=18:
                def set_bn_eval(m):
                    classname = m.__class__.__name__
                    if classname.find('BatchNorm') != -1:
                        m.eval()
                self.unet.apply(set_bn_eval)
            # if epoch == 25:
            #     self.optimizer.param_groups[0]['initial_lr'] = 0.00005
            #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 25)
            epoch_loss = 0
            tbar = tqdm.tqdm(self.train_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                masks = masks.to(self.device)

                # SR : Segmentation Result
                net_output = self.unet(images)
                net_output_flat = net_output.view(net_output.size(0), -1)
                masks_flat = masks.view(masks.size(0), -1)
                loss_set = self.criterion(net_output_flat, masks_flat)

                try:
                    loss_num = len(loss_set)
                except:
                    loss_num = 1
                # 依据返回的损失个数分情况处理
                if loss_num > 1:
                    for loss_index, loss_item in enumerate(loss_set):
                        if loss_index > 0:
                            loss_name = 'stage1_loss_%d' % loss_index
                            self.writer.add_scalar(loss_name, loss_item.item(), global_step_before + i)
                    loss = loss_set[0]
                else:
                    loss = loss_set
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (
                    param_group['lr'])

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('Stage1_train_loss', loss.item(), global_step_before + i)

                descript = "Train Loss: %.7f, lr: %s" % (loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)
            # 更新global_step_before为下次迭代做准备
            global_step_before += len(tbar)

            # Print the log info
            print(
                'Finish Stage1 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage1, epoch_loss / len(tbar)))
            write_txt(self.save_path, 'Finish Stage1 Epoch [%d/%d], Average Loss: %.7f' % (
            epoch, self.epoch_stage1, epoch_loss / len(tbar)))

            # 验证模型，保存权重，并保存日志
            loss_mean, dice_mean = self.validation(stage=1)
            if dice_mean > self.max_dice:
                is_best = True
                self.max_dice = dice_mean
            else:
                is_best = False

            self.lr = lr_scheduler.get_lr()
            state = {'epoch': epoch,
                     'state_dict': self.unet.module.state_dict(),
                     'max_dice': self.max_dice,
                     'optimizer': self.optimizer.state_dict(),
                     'lr': self.lr}

            self.save_checkpoint(state, 1, index, is_best)

            self.writer.add_scalars('Stage1_val_loss_dice', {'val_loss':loss_mean,'val_dice':dice_mean}, epoch)
            self.writer.add_scalar('Stage1_lr', self.lr[0], epoch)

            # 学习率衰减
            lr_scheduler.step()
    #
    def train_stage2(self, index):
        # for param in self.unet.module.encoder.parameters():
        #     param.requires_grad = False
        # self.optimizer = optim.Adam(filter(lambda p:p.requires_grad, self.unet.module.parameters()),self.lr_stage2,weight_decay=self.weight_decay)

        # # 冻结BN层， see https://zhuanlan.zhihu.com/p/65439075 and https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/100736591271 for more information
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        # self.optimizer = optim.Adam([{'params': self.unet.decoder.parameters(), 'lr': 1e-5}, {'params': self.unet.encoder.parameters(), 'lr': 1e-7},])
        # self.optimizer = optim.Adam(self.unet.module.parameters(), self.lr_stage2, weight_decay=self.weight_decay_stage2
        #                             )
        # self.optimizer = NoamLR(self.unet.module.parameters(),10)
        self.optimizer = optim.SGD(self.unet.module.parameters(),lr=self.lr_stage2,momentum=self.momentum,
                                   weight_decay=self.weight_decay_stage2)
        # 加载的resume分为两种情况：之前没有训练第二个阶段，现在要加载第一个阶段的参数；第二个阶段训练了一半要继续训练
        if self.resume:
            # 若第二个阶段训练一半，要重新加载 TODO
            if self.resume.split('_')[-3] == '2':
                self.load_checkpoint(load_optimizer=False)  # 当load_optimizer为True会重新加载学习率和优化器
                '''
                CosineAnnealingLR：若存在['initial_lr']，则从initial_lr开始衰减；
                若不存在，则执行CosineAnnealingLR会在optimizer.param_groups中添加initial_lr键值，其值等于lr

                重置初始学习率，在load_checkpoint中会加载优化器，但其中的initial_lr还是之前的，所以需要覆盖为self.lr，让其从self.lr衰减
                '''
                self.optimizer.param_groups[0]['initial_lr'] = self.lr

            # 若第一阶段结束后没有直接进行第二个阶段，中间暂停了
            elif self.resume.split('_')[-3] == '1':
                self.load_checkpoint(load_optimizer=False)
                self.start_epoch = 0
                self.max_dice = 0

        # 第一阶段结束后直接进行第二个阶段，中间并没有暂停
        else:
            self.start_epoch = 0
            self.max_dice = 0

        # 防止训练到一半暂停重新训练，日志被覆盖
        global_step_before = self.start_epoch * len(self.train_loader)

        stage2_epoches = self.epoch_stage2 - self.start_epoch
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 30)
        d_h,d_l = 0,0
        for epoch in range(self.start_epoch, self.epoch_stage2):
            # if self.epoch >= 20:
            # self.unet.apply(set_bn_eval)
            epoch += 1
            self.unet.train(True)
            epoch_loss = 0

            self.reset_grad()  # 梯度累加的时候需要使用

            tbar = tqdm.tqdm(self.train_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                masks = masks.to(self.device)
                assert images.size(2) == 512

                # SR : Segmentation Result
                net_output = self.unet(images)
                net_output_flat = net_output.view(net_output.size(0), -1)
                masks_flat = masks.view(masks.size(0), -1)
                loss_set = self.criterion_stage2(net_output_flat, masks_flat)

                try:
                    loss_num = len(loss_set)
                except:
                    loss_num = 1
                # 依据返回的损失个数分情况处理
                if loss_num > 1:
                    for loss_index, loss_item in enumerate(loss_set):
                        if loss_index > 0:
                            loss_name = 'stage2_loss_%d' % loss_index
                            self.writer.add_scalar(loss_name, loss_item.item(), global_step_before + i)
                    loss = loss_set[0]
                else:
                    loss = loss_set
                epoch_loss += loss.item()

                # Backprop + optimize, see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20 for Accumulating Gradients
                if epoch <= self.epoch_stage2 - self.epoch_stage2_accumulation:
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    # loss = loss / self.accumulation_steps                # Normalize our loss (if averaged)
                    loss.backward()  # Backward pass
                    if (i + 1) % self.accumulation_steps == 0:  # Wait for several backward steps
                        self.optimizer.step()  # Now we can do an optimizer step
                        self.reset_grad()

                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (
                    param_group['lr'])

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('Stage2_train_loss', loss.item(), global_step_before + i)

                descript = "Train Loss: %.7f, lr: %s" % (loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)
            # 更新global_step_before为下次迭代做准备
            global_step_before += len(tbar)

            # Print the log info
            print(
                'Finish Stage2 Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch_stage2, epoch_loss / len(tbar)))
            write_txt(self.save_path, 'Finish Stage2 Epoch [%d/%d], Average Loss: %.7f' % (
            epoch, self.epoch_stage2, epoch_loss / len(tbar)))

            # 验证模型，保存权重，并保存日志
            loss_mean, dice_mean = self.validation(stage=2)
            if dice_mean > self.max_dice:
                is_best = True
                self.max_dice = dice_mean
            else:
                is_best = False

            self.lr = lr_scheduler.get_lr()
            state = {'epoch': epoch,
                     'state_dict': self.unet.module.state_dict(),
                     'max_dice': self.max_dice,
                     'optimizer': self.optimizer.state_dict(),
                     'lr': self.lr}

            self.save_checkpoint(state, 2, index, is_best)

            self.writer.add_scalar('Stage2_val_loss', loss_mean, epoch)
            self.writer.add_scalar('Stage2_val_dice', dice_mean, epoch)
            self.writer.add_scalar('Stage2_lr', self.lr[0], epoch)

            # 学习率衰减
            lr_scheduler.step()

    # stage3, 接着stage2的训练，只训练有mask的样本
    def dice_overall(self, preds, targs):
        n = preds.shape[0]  # batch size:
        preds = preds.view(n, -1)
        targs = targs.view(n, -1)
        # preds, targs = preds.to(self.device), targs.to(self.device)
        preds, targs = preds.cpu(), targs.cpu()
        intersect = (preds * targs).sum(-1).float()
        union = (preds + targs).sum(-1).float()
        u0 = union==0
        intersect[u0] = 1
        union[u0] = 2
        return (2. * intersect / union)

    def validation(self, stage=1):
        d_h,d_l=0,0
        # 验证的时候，train(False)是必须的0，设置其中的BN层、dropout等为eval模式
        # with torch.no_grad(): 可以有，在这个上下文管理器中，不反向传播，会加快速度，可以使用较大batch size
        self.unet.eval()
        tbar = tqdm.tqdm(self.valid_loader)
        loss_sum, dice_sum = 0, 0
        if stage == 1:
            criterion = self.criterion
        elif stage == 2:
            criterion = self.criterion_stage2
        elif stage == 3:
            criterion = self.criterion_stage3
        with torch.no_grad():
            for i, (images, masks) in enumerate(tbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                net_output = self.unet(images)
                net_output_flat = net_output.view(net_output.size(0), -1)
                masks_flat = masks.view(masks.size(0), -1)

                loss_set = criterion(net_output_flat, masks_flat)
                try:
                    loss_num = len(loss_set)
                except:
                    loss_num = 1

                # 依据返回的损失个数分情况处理
                if loss_num > 1:
                    loss = loss_set[0]
                else:
                    loss = loss_set
                loss_sum += loss.item()

                # 计算dice系数，预测出的矩阵要经过sigmoid含义以及阈值，阈值默认为0.5
                net_output_flat_sign = (torch.sigmoid(net_output_flat) > 0.5).float()
                dice = self.dice_overall(net_output_flat_sign, masks_flat).mean()

                # if dice>0.8:
                #
                #     masks_i = masks.view((masks.shape[0],-1,masks.shape[1],masks.shape[2]))
                #     # img_show = torch.cat((images,masks,net_output))
                #     self.writer.add_images('images_Dice>0.8',images, d_h)
                #     self.writer.add_images('masks_Dice>0.8',masks_i, d_h)
                #     self.writer.add_images('preds_Dice>0.8',net_output, d_h)
                #     d_h += 1
                #
                # if dice<0.15:
                #
                #     masks_i = masks.view((masks.shape[0],-1,masks.shape[1],masks.shape[2]))
                #     # img_show = torch.cat((images, masks, net_output))
                #     self.writer.add_images('images_Dice<0.15', images, d_l)
                #     self.writer.add_images('masks_Dice<0.15', masks_i, d_l)
                #     self.writer.add_images('preds_Dice<0.15', net_output, d_l)
                #     d_l += 1
                dice_sum += dice.item()

                descript = "Val Loss: {:.7f}, dice: {:.7f}".format(loss.item(), dice.item())
                tbar.set_description(desc=descript)

        loss_mean, dice_mean = loss_sum / len(tbar), dice_sum / len(tbar)
        print("Val Loss: {:.7f}, dice: {:.7f}".format(loss_mean, dice_mean))
        write_txt(self.save_path, "Val Loss: {:.7f}, dice: {:.7f}".format(loss_mean, dice_mean))
        return loss_mean, dice_mean


    def pred_mask_count(self, model_path, masks_bool, val_index, best_thr, best_pixel_thr):
        '''加载模型，根据最优阈值和最优像素阈值，得到在验证集上的分类准确率。适用于训练的第二阶段使用 dice 选完阈值，查看分类准确率
        Args:
            model_path: 当前模型的权重路径
            masks_bool: 全部数据集中的每个是否含有mask
            val_index: 当前验证集的在全部数据集的下标
            best_thr: 选出的最优阈值
            best_pixel_thr: 选出的最优像素阈值

        Return: None, 打印出有多少个真实情况有多少个正样本，实际预测出了多少个样本。但是不是很严谨，因为这不能代表正确率。
        '''
        count_true, count_pred = 0, 0
        for index1 in val_index:
            if masks_bool[index1]:
                count_true += 1

        self.unet.module.load_state_dict(torch.load(model_path)['state_dict'])
        print('Loaded from %s' % model_path)
        self.unet.eval()

        with torch.no_grad():
            tmp = []
            tbar = tqdm.tqdm(self.valid_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                net_output = torch.sigmoid(self.unet(images))
                preds = (net_output > best_thr).to(self.device).float()  # 大于阈值的归为1
                preds[preds.view(preds.shape[0], -1).sum(-1) < best_pixel_thr, ...] = 0.0  # 过滤噪声点

                n = preds.shape[0]  # batch size为多少
                preds = preds.view(n, -1)

                for index2 in range(n):
                    pred = preds[index2, ...]
                    if torch.sum(pred) > 0:
                        count_pred += 1

                tmp.append(self.dice_overall(preds, masks).mean())
            print('score:', sum(tmp) / len(tmp))

        print('count_true:{}, count_pred:{}'.format(count_true, count_pred))

    def grid_search(self, thrs_big, pixel_thrs):
        '''利用网格法搜索最优阈值和最优像素阈值

        Args:
            thrs_big: 网格法搜索时的一系列阈值
            pixel_thrs: 网格搜索时的一系列像素阈值

        Return: 最优阈值，最优像素阈值，最高得分，网络矩阵中每个位置的得分
        '''
        with torch.no_grad():
            # 先大概选取阈值范围和像素阈值范围
            dices_big = []  # 存放的是二维矩阵，每一行为每一个阈值下所有像素阈值得到的得分
            for th in thrs_big:
                dices_pixel = []
                for pixel_thr in pixel_thrs:
                    tmp = []
                    tbar = tqdm.tqdm(self.valid_loader)
                    for i, (images, masks) in enumerate(tbar):
                        # GT : Ground Truth
                        images = images.to(self.device)
                        net_output = torch.sigmoid(self.unet(images))
                        preds = (net_output > th).to(self.device).float()  # 大于阈值的归为1
                        preds[preds.view(preds.shape[0], -1).sum(-1) < pixel_thr, ...] = 0.0  # 过滤噪声点
                        tmp.append(self.dice_overall(preds, masks).mean())
                        # tmp.append(self.classify_score(preds, masks))
                    dices_pixel.append(sum(tmp) / len(tmp))
                dices_big.append(dices_pixel)
            dices_big = np.array(dices_big)
            print('粗略挑选最优阈值和最优像素阈值，dices_big_shape:{}'.format(np.shape(dices_big)))
            re = np.where(dices_big == np.max(dices_big))
            # 如果有多个最大值的处理方式
            if np.shape(re)[1] != 1:
                re = re[0]
            best_thrs_big, best_pixel_thr = thrs_big[int(re[0])], pixel_thrs[int(re[1])]
            best_thr, score = best_thrs_big, dices_big.max()
        return best_thr, best_pixel_thr, score, dices_big

    def choose_threshold_grid(self, model_path, index):
        '''利用网格法搜索当前模型的最优阈值和最优像素阈值，分为粗略搜索和精细搜索两个过程；并保存热力图

        Args:
            model_path: 当前模型权重的位置
            index: 当前为第几个fold

        Return: 最优阈值，最优像素阈值，最高得分
        '''
        self.unet.module.load_state_dict(torch.load(model_path)['state_dict'])
        stage = eval(model_path.split('/')[-1].split('_')[2])
        print('Loaded from %s, using choose_threshold_grid!' % model_path)
        self.unet.eval()

        thrs_big1 = np.arange(0.35, 0.7, 0.015)  # 阈值列表
        pixel_thrs1 = np.arange(100, 512, 100)  # 像素阈值列表
        best_thr1, best_pixel_thr1, score1, dices_big1 = self.grid_search(thrs_big1, pixel_thrs1)
        print('best_thr1:{}, best_pixel_thr1:{}, score1:{}'.format(best_thr1, best_pixel_thr1, score1))

        thrs_big2 = np.arange(best_thr1 - 0.015, best_thr1 + 0.015, 0.0075)  # 阈值列表
        pixel_thrs2 = np.arange(best_pixel_thr1 - 50, best_pixel_thr1 + 50, 50)  # 像素阈值列表
        best_thr2, best_pixel_thr2, score2, dices_big2 = self.grid_search(thrs_big2, pixel_thrs2)
        print('best_thr2:{}, best_pixel_thr2:{}, score2:{}'.format(best_thr2, best_pixel_thr2, score2))

        if score1 < score2:
            best_thr, best_pixel_thr, score, dices_big = best_thr2, best_pixel_thr2, score2, dices_big2
        else:
            best_thr, best_pixel_thr, score, dices_big = best_thr1, best_pixel_thr1, score1, dices_big1

        print('best_thr:{}, best_pixel_thr:{}, score:{}'.format(best_thr, best_pixel_thr, score))

        f, (ax1, ax2) = plt.subplots(figsize=(14.4, 4.8), ncols=2)

        cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
        data1 = pd.DataFrame(data=dices_big1, index=np.around(thrs_big1, 3), columns=pixel_thrs1)
        sns.heatmap(data1, linewidths=0.05, ax=ax1, vmax=np.max(dices_big1), vmin=np.min(dices_big1), cmap=cmap,
                    annot=True, fmt='.4f')
        ax1.set_title('Large-scale search')

        data2 = pd.DataFrame(data=dices_big2, index=np.around(thrs_big2, 3), columns=pixel_thrs2)
        sns.heatmap(data2, linewidths=0.05, ax=ax2, vmax=np.max(dices_big2), vmin=np.min(dices_big2), cmap=cmap,
                    annot=True, fmt='.4f')
        ax2.set_title('Little-scale search')
        f.savefig(os.path.join(self.save_path, 'stage{}'.format(stage) + '_fold' + str(index)))
        # plt.show()
        plt.close()
        return float(best_thr), float(best_pixel_thr), float(score)

    def get_dice_onval(self, model_path, best_thr, pixel_thr):
        '''已经训练好模型，并且选完阈值后。根据当前模型，best_thr, pixel_thr得到在验证集的表现

        Args:
            model_path: 要加载的模型路径
            best_thr: 选出的最优阈值
            pixel_thr: 选出的最优像素阈值

        Return: None
        '''
        self.unet.module.load_state_dict(torch.load(model_path)['state_dict'])
        stage = eval(model_path.split('/')[-1].split('_')[2])
        print('Loaded from %s, using get_dice_onval!' % model_path)
        self.unet.eval()

        with torch.no_grad():
            # 选最优像素阈值
            tmp = []
            tbar = tqdm.tqdm(self.valid_loader)
            for i, (images, masks) in enumerate(tbar):
                # GT : Ground Truth
                images = images.to(self.device)
                net_output = torch.sigmoid(self.unet(images))
                preds = (net_output > best_thr).to(self.device).float()  # 大于阈值的归为1
                if stage != 3:
                    preds[preds.view(preds.shape[0], -1).sum(-1) < pixel_thr, ...] = 0.0  # 过滤噪声点
                tmp.append(self.dice_overall(preds, masks).mean())
                # tmp.append(self.classify_score(preds, masks))
            score = sum(tmp) / len(tmp)
        print('best_thr:{}, best_pixel_thr:{}, score:{}'.format(best_thr, pixel_thr, score))
