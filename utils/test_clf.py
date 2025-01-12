# test.py
import torch
from tqdm import tqdm
import numpy as np
from config import get_config
import time

from sklearn.metrics import accuracy_score  # 可以删除，因为不再需要
Config = get_config()

device = Config.DEVICE

class AverageMeter:
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
        fmtstr = '{name} {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test(model, dataloader):
    model.eval()

    # 初始化 AverageMeter
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        for img, labels in dataloader:
            img, labels = img.to(device), labels.to(device)
            output = model(img)

            # 计算 top-1 准确率
            prec1 = accuracy(output, labels, topk=(1,))[0]
            top1.update(prec1.item(), img.size(0))

    accuracy_avg = top1.avg

    print(f'Test Accuracy: {accuracy_avg:.2f}%')

    return accuracy_avg
