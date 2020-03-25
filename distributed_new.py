import os
import sys
root = '/home/yyang2/data/yyang2/pycharm/isbi_dr'
sys.path.append('/home/yyang2/data/yyang2/pycharm/isbi_dr')
from data_loader import DatasetGenerator
from Focal_Loss import FocalLoss
import csv

import argparse
import os
import random
import shutil
import time
import warnings
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import hyperparam_scheduler
import torchvision
import copy
import numpy as np
from tensorboardX import SummaryWriter
data_root = '/home/yyang2/data/yyang2/Data/EyeQ-master/results/isbi'
writer = SummaryWriter(os.path.join(data_root, 'results/tensorboard/vy1'))


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



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


#
# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--data', metavar='DIR', default='/home/yyang2/data/yyang2/Data/EyeQ-master/', help='path to dataset')
# parser.add_argument('-a',
#                     '--arch',
#                     metavar='ARCH',
#                     default='densenet121',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) + ' (default: densenet121)')
# parser.add_argument('-j',
#                     '--workers',
#                     default=1,
#                     type=int,
#                     metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# parser.add_argument('-b',
#                     '--batch-size',
#                     default=12,
#                     type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 3200), this is the total '
#                     'batch size of all GPUs on the current node when '
#                     'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--lr',
#                     '--learning-rate',
#                     default=7 * 1e-4,
#                     type=float,
#                     metavar='LR',
#                     help='initial learning rate',
#                     dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
# parser.add_argument('--local_rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--wd',
#                     '--weight-decay',
#                     default=1e-4,
#                     type=float,
#                     metavar='W',
#                     help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=300, type=int, metavar='N', help='print frequency (default: 80)')
# parser.add_argument('-pre', '--pretrained', default=False, help='pretrained (default: False)')
#
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# # parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
global best_acc1
global best_kappa
best_acc1 = 0
best_kappa = 0
epoches = 300
start_epoches = 24
# def main():
#     # args = parser.parse_args()
#
#     # if args.seed is not None:
#     #     random.seed(args.seed)
#     #     torch.manual_seed(args.seed)
#     #     cudnn.deterministic = True
#     #     warnings.warn('You have chosen to seed training. '
#     #                   'This will turn on the CUDNN deterministic setting, '
#     #                   'which can slow down your training considerably! '
#     #                   'You may see unexpected behavior when restarting '
#     #                   'from checkpoints.')
#
#     main_worker()
#
#
# def main_worker():
#     global best_acc1
#     global kappa_v
#     global best_kappa
#
#     dist.init_process_group(backend='nccl')
#     # create model

    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
model = models.__dict__['densenet121'](num_classes=5)
path = '/home/yyang2/data/yyang2/Data/EyeQ-master/results/isbi/vy1/checkpoint.pth.tar'
loaded_model = torch.load(path)
model.load_state_dict(loaded_model['state_dict'])

module = torch.nn.DataParallel(model)
model = model.cuda()

criterion = FocalLoss(5)

optimizer = torch.optim.SGD(model.parameters(), lr= 7 * 1e-4, momentum=0.9, weight_decay=1e-4)
cycle_scheduler = hyperparam_scheduler.make_1cycle(optimizer, 7739 * 100, 7 * 1e-4, 0.9)

train_dataset = DatasetGenerator(list_file='/home/yyang2/data/yyang2/Data/EyeQ-master/y1_df_train_over.csv',
                                 transform1=transform_list1,
                              transform2=transformList2, n_class=5, set_name='train')

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=12,
                                           shuffle=True,
                                           num_workers=1 * 4,
                                           pin_memory=True
                                           )


test_dataset = DatasetGenerator(list_file='/home/yyang2/data/yyang2/Data/isbs/all.csv',
                                transform1=transform_list_val1,
                             transform2=transformList2, n_class=5, set_name='val')

val_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=8,
                                         shuffle=False,
                                         num_workers=1 * 4 ,
                                         pin_memory=True)


log_csv = "/home/yyang2/data/yyang2/Data/EyeQ-master/distributed_y1.csv"


def train(train_loader, model, criterion, optimizer, epoch, cycle_scheduler):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    '''
    if epoch == 0:
        model.cuda()
        path = '/home/yyang2/data/yyang2/pycode/isbi_dr/results/model_best.pth.tar'
        loaded_model = torch.load(path)
        model.load_state_dict(loaded_model['state_dict'])
        print('加载完成')
    else:
        pass
    '''
    model.train()
    end = time.time()
    y_tru = None
    y_p = None
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)
        if y_tru is None:
            y_tru = np.array(target)
        else:
            y_tru = np.append(y_tru, np.array(target))

        images = images.cuda()
        target = target.cuda().long()

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 3))


        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cycle_scheduler.batch_step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        y_pre = output.argmax(dim=1).detach().cpu().numpy()

        if y_p is None:
            y_p = np.array(y_pre)
        else:
            y_p = np.append(y_p, np.array(y_pre))
        if i % 300 == 0:
            progress.display(i)
            print(epoch, ':', optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['momentum'])

    y_tru = y_tru.reshape((-1))
    y_p = y_p.reshape((-1))
    kappa = cohen_kappa_score(y_tru, y_p)
    acc_t = accuracy_score(y_tru, y_p)
    confu = confusion_matrix(y_tru, y_p)
    print('训练的kappa：', kappa, '-----acc:', acc_t)
    print('confusio_nmatrix:\n', confu)
    return acc_t, kappa


def validate(val_loader, model, criterion):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    y_tru = None
    y_p = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if y_tru is None:
                y_tru = np.array(target)
            else:
                y_tru = np.append(y_tru, np.array(target))
            images = images.cuda()
            target = target.cuda().long()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 300 == 0:
                progress.display(i)

            y_pre = output.argmax(dim=1).detach().cpu().numpy()
            if y_p is None:
                y_p = np.array(y_pre)
            else:
                y_p = np.append(y_p, np.array(y_pre))
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    kappa = cohen_kappa_score(y_tru, y_p)
    acc_t = accuracy_score(y_tru, y_p)
    confu = confusion_matrix(y_tru, y_p)
    print('测试的kappa：', kappa, '-----acc:', acc_t)
    print('confusio_nmatrix:\n', confu)
    return acc_t, kappa


def save_checkpoint(state, is_best, filename='/home/yyang2/data/yyang2/Data/EyeQ-master/results/isbi/vy1/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/home/yyang2/data/yyang2/Data/EyeQ-master/results/isbi/vy1/model_best.pth.tar')
        print('保存完成')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['momentum'] = 0.9


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


for epoch in range(start_epoches,epoches):
    epoch_start = time.time()

    # adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch


    acc_t1, kappa_t = train(train_loader, model, criterion, optimizer, epoch, cycle_scheduler)


    # evaluate on validation set
    acc_v1, kappa_v = validate(val_loader, model, criterion)
    writer.add_scalars('data/acc', {'train': acc_t1, 'val': acc_v1}, epoch)
    writer.add_scalars('data/kappa', {'train': kappa_t, 'val': kappa_v}, epoch)
    # remember best acc@1 and save checkpoint
    is_best = acc_v1 > best_acc1
    best_acc1 = max(acc_v1, best_acc1)

    is_best_kappa = kappa_v > best_kappa
    best_kappa = max(kappa_v, best_kappa)
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

    with open(log_csv, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)), epoch_end - epoch_start]
        csv_write.writerow(data_row)

    save_checkpoint(
        {
            'epoch': epoch + 1,
            'arch': 'densenet121',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best_kappa)







