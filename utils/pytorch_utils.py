import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
import argparse
import torch
import random
import os
import math


def print_args(args: argparse.ArgumentParser):
    """
    print the hyperparameters
    :param args: hyperparameters
    :return: None
    """
    s = "=========================================================\n"
    for arg, concent in args.__dict__.items():
        s += "{}:{}\n".format(arg, concent)
    return s


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)

def init_mask(model: nn.Module):
    classname = model.__class__.__name__
    if classname.find('Mask') != -1:
        nn.init.kaiming_uniform_(model.neuron_mask, a=math.sqrt(5))


def bca_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def standard_normalize(x_train, x_test, clip_range=None):
    mean, std = np.mean(x_train), np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    if clip_range is not None:
        x = np.clip(x_train, a_min=clip_range[0], a_max=clip_range[1])
        x = np.clip(x_test, a_min=clip_range[0], a_max=clip_range[1])
    return x_train, x_test


def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer: nn.Module, epoch: int):
    """decrease the learning rate"""
    if epoch == 50 or epoch == 100:
        r = 0.1
    else:
        r = 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= r


class CustomTensorDataset(Dataset):
    """ TnsorDataset with support of transforms. """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def weight_for_balanced_classes(y: torch.Tensor):
    count = [0.0] * len(np.unique(y.numpy()))
    for label in y:
        count[label] += 1.0
    count = [len(y) / x for x in count]
    weight = [0.0] * len(y)
    for idx, label in enumerate(y):
        weight[idx] = count[label]

    return weight


class DataIterator:
    def __init__(self,
                 dataloader: DataLoader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __next__(self):
        try:
            data = next(self.iter)
        except Exception:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.dataloader)