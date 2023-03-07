import os
import sys
import json
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def read_split_data(root='train_label.txt', train_rate=0.9, randomseed=42):
    random.seed(randomseed)
    df = pd.read_csv(root, header=None, sep=' ', index_col=False)
    n = len(df)
    indices = np.random.permutation(n)
    split_point = int(train_rate * n)
    train_df = df.iloc[indices[:split_point], :]
    val_df = df.iloc[indices[split_point:], :]
    train_images_path, train_images_label = list(train_df.iloc[:, 0]), list(train_df.iloc[:, 1])
    val_images_path, val_images_label = list(val_df.iloc[:, 0]), list(val_df.iloc[:, 1])
    print("{} images were found in the dataset.".format(len(train_images_path)+len(val_images_path)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    return train_images_path, train_images_label, val_images_path, val_images_label


class FocalLoss(nn.Module):
    def __init__(self, class_nums, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.Tensor(class_nums, 1).fill_(1.0)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
        self.class_num = class_nums
        self.size_average = size_average

    def forward(self, inputs, targets):
        inputs = inputs.to(targets.device)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = torch.zeros((N, C), device=targets.device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        
        alpha = self.alpha.to(targets.device)
        alpha = alpha[ids.view(-1)].squeeze()
        probs = (P * class_mask).sum(1).view(-1, 1).to(targets.device)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * probs.log()

        loss = torch.where(torch.isnan(batch_loss), torch.zeros_like(batch_loss), batch_loss)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  
    accu_num = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(epoch,accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num,
                                                                               optimizer.param_groups[0]['lr'])


        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    model.eval()
    accu_num = torch.zeros(1).to(device) 
    accu_loss = torch.zeros(1).to(device)  

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class MyDataSet(Dataset):
    """ 
    Custom Dataset 
    """
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        """        
        The official implementation of default collate can be referred to:
        https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        """
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


if __name__ == '__main__':
    print()
    



