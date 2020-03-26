import torchvision.transforms as transforms
import shutil
import time
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import torch
import numpy as np

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

def save_checkpoint(state, is_best_acc,is_best_kapp, filename,file_best_acc_name,file_best_kappa_name):
    torch.save(state, filename)
    if is_best_acc:
        shutil.copyfile(filename, file_best_acc_name)
        print('保存完成')
    else:
        pass
    if is_best_kapp:
        shutil.copyfile(filename, file_best_kappa_name)
        print('保存完成')
    else:
        pass

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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['momentum'] = 0.9

def train(train_loader, model, criterion, optimizer, epoch, cycle_scheduler):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))


    # 进入训练模式
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

