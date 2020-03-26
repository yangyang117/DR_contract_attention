import os
import sys
sys.path.append('/home/yyang2/data/yyang2/pycharm/DR_contract_attention')
from utils.data_loader import DatasetGenerator
from utils.Focal_Loss import FocalLoss
from utils import hyperparam_scheduler
from utils.c_utils import save_checkpoint, train, validate
import cfg.config as cfg
import csv
import shutil
import time
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from tensorboardX import SummaryWriter
writer = SummaryWriter('/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/tensorboard/res101_1')


transform_list1 = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
    ])


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model = models.resnet101(num_classes=cfg.n_class)
# 已有训练结果时进行加载
# path = '/home/yyang2/data/yyang2/Data/EyeQ-master/results/isbi/vy1/checkpoint.pth.tar'
# loaded_model = torch.load(path)
# model.load_state_dict(loaded_model['state_dict'])

module = torch.nn.DataParallel(model)
model = model.cuda()

criterion = FocalLoss(5)

optimizer = torch.optim.SGD(model.parameters(), lr= cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
cycle_scheduler = hyperparam_scheduler.make_1cycle(optimizer, cfg.opt_time, cfg.opt_min_lr, cfg.max_mo)

global best_acc1
global best_kappa
best_acc1 = 0
best_kappa = 0


train_dataset = DatasetGenerator(list_file=cfg.train_file,
                                 transform1=transform_list1,
                              transform2=transformList2, n_class=5, set_name='train',mode = 'normal')

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=12,
                                           shuffle=True,
                                           num_workers=1 * 4,
                                           pin_memory=True
                                           )


test_dataset = DatasetGenerator(list_file=cfg.test_file,
                                transform1=transform_list_val1,
                             transform2=transformList2, n_class=5, set_name='val',mode = 'normal')

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=12,
                                         shuffle=False,
                                         num_workers=1 * 4 ,
                                         pin_memory=True)

val_dataset = DatasetGenerator(list_file=cfg.val_file,
                                transform1=transform_list_val1,
                             transform2=transformList2, n_class=5, set_name='val',mode = 'normal')

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=12,
                                         shuffle=False,
                                         num_workers=1 * 4 ,
                                         pin_memory=True)



for epoch in range(cfg.start_epoches,cfg.epoches):
    epoch_start = time.time()

    # adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch


    acc_t1, kappa_t, loss_t = train(train_loader, model, criterion, optimizer, epoch, cycle_scheduler)


    # evaluate on validation set
    acc_v, kappa_v, loss_v = validate(val_loader, model, criterion)
    if (epoch + 1) % 10 == 0:
        acc_t, kappa_t, loss_t = validate(test_loader, model, criterion)
    writer.add_scalars('data/acc', {'train': acc_t1, 'val': acc_v}, epoch)
    writer.add_scalars('data/kappa', {'train': kappa_t, 'val': kappa_v}, epoch)
    writer.add_scalars('data/loss', {'train': loss_t, 'val': loss_v}, epoch)
    # remember best acc@1 and save checkpoint
    is_best_acc = acc_v > best_acc1
    best_acc1 = max(acc_v, best_acc1)

    is_best_kappa = kappa_v > best_kappa
    best_kappa = max(kappa_v, best_kappa)
    if is_best_acc:
        print('开始新的验证:\n')
        acc_t, kappa_t, loss_t = validate(test_loader, model, criterion)

    '''
    if is_best_kappa:
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.module.state_dict(), '/home/yyang2/data/yyang2/pycode/isbi_dr/results/best_kappa.pkl')
        print('保存最好的kappa模型：', best_kappa, 'epoch:', epoch)
    model.load_state_dict(best_model_wts)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    '''
    epoch_end = time.time()

    path_save = '/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/model_file/'
    filename = path_save + 'checkpoint.pth.tar'
    file_best_acc_name = path_save + 'acc/' + str(epoch + 1) + '_best_acc.pth.tar'
    file_best_kappa_name = path_save + 'kappa/' + str(epoch + 1) + '_best_acc.pth.tar'

    save_checkpoint(
        {
            'epoch': epoch + 1,
            'arch': 'densenet121',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_kappa':best_kappa,
        }, is_best_acc,is_best_kappa,filename,file_best_acc_name,file_best_kappa_name)







