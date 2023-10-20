import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
from models import EEGNet
from utils.pytorch_utils import init_weights
from torch.utils.data import TensorDataset, DataLoader



def FGSM_cha(model: nn.Module,
         x: torch.Tensor,
         y: torch.Tensor,
         eps: Optional[float] = 0.05,
         distance: Optional[str] = 'inf',
         target: Optional[bool] = False):
    """ FGSM attack """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)

    data_loader = DataLoader(dataset=TensorDataset(x, y),
                             batch_size=128,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False)

    model.eval()
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.clone().detach().to(device)
        batch_y = batch_y.clone().detach().to(device)
        batch_x.requires_grad = True

        with torch.enable_grad():
            loss = criterion(model(batch_x), batch_y)
        grad = torch.autograd.grad(loss,
                                   batch_x,
                                   retain_graph=False,
                                   create_graph=False)[0]

        if distance == 'inf':
            delta = grad.detach().sign()
        else:
            raise 'No such distance.'

        cha_std = batch_x.std(axis=-1)[:,:,:,None].detach()
        if target:
            batch_adv_x = batch_x.detach() - eps * delta * cha_std
        else:
            batch_adv_x = batch_x.detach() + eps * delta * cha_std

        if step == 0: adv_x = batch_adv_x
        else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

    return adv_x


def PGD_cha(model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: Optional[float] = 0.05,
        alpha: Optional[float] = 0.005,
        steps: Optional[int] = 20,
        distance: Optional[str] = 'inf',
        target: Optional[bool] = False):
    """ PGD attack """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss().to(device)

    data_loader = DataLoader(dataset=TensorDataset(x, y),
                             batch_size=128,
                             shuffle=False,
                             drop_last=False,num_workers=1)

    model.eval()
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.clone().detach().to(device)
        batch_y = batch_y.clone().detach().to(device)

        cha_std = batch_x.std(axis=-1)[:,:,:,None].detach()
        # craft adversarial examples
        batch_adv_x = batch_x.clone().detach() + torch.empty_like(
            batch_x).uniform_(-eps, eps).detach() * cha_std
        for _ in range(steps):
            batch_adv_x.requires_grad = True
            with torch.enable_grad():
                loss = criterion(model(batch_adv_x), batch_y)
            grad = torch.autograd.grad(loss,
                                       batch_adv_x,
                                       retain_graph=False,
                                       create_graph=False)[0]

            if distance == 'inf':
                delta = grad.detach().sign()
            else:
                raise 'No such distance'

            if target:
                batch_adv_x = batch_adv_x.detach() - alpha * delta * cha_std
            else:
                batch_adv_x = batch_adv_x.detach() + alpha * delta * cha_std

            # projection
            if distance == 'inf':
                delta = torch.clamp(batch_adv_x - batch_x, min=-eps*cha_std, max=eps*cha_std)

            batch_adv_x = (batch_x + delta).detach()

        if step == 0: adv_x = batch_adv_x
        else: adv_x = torch.cat([adv_x, batch_adv_x], dim=0)

    return adv_x


def PGD_batch_cha(model: nn.Module,
              x: torch.Tensor,
              y: torch.Tensor,
              eps: Optional[float] = 0.05,
              alpha: Optional[float] = 0.005,
              steps: Optional[int] = 20,
              label_free: Optional[bool] = False):
    """ PGD attack """
    device = next(model.parameters()).device
    if label_free:
        criterion =nn.KLDivLoss(reduction='batchmean').to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)

    cha_std = x.std(axis=-1)[:,:,:,None].detach()
    # craft adversarial examples
    adv_x = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps).detach() * cha_std
    for _ in range(steps):
        adv_x.requires_grad = True
        with torch.enable_grad():
            if label_free:
                loss = criterion(F.log_softmax(model(adv_x), dim=1), F.softmax(model(x), dim=1))
            else:
                loss = criterion(model(adv_x), y)
        grad = torch.autograd.grad(loss,
                                   adv_x,
                                   retain_graph=False,
                                   create_graph=False)[0]
        adv_x = adv_x.detach() + alpha * cha_std * grad.detach().sign()
        # projection
        delta = torch.clamp(adv_x - x, min=-eps*cha_std, max=eps*cha_std)
        adv_x = (x + delta).detach()

    return adv_x

def FGSM_batch_cha(model: nn.Module,
              x: torch.Tensor,
              y: torch.Tensor,
              eps: Optional[float] = 0.05,
              label_free: Optional[bool] = False):
    """ FGSM attack """
    device = next(model.parameters()).device
    if label_free:
        criterion =nn.KLDivLoss(reduction='batchmean').to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)

    cha_std = x.std(axis=-1)[:,:,:,None].detach()
    # craft adversarial examples
    x.requires_grad = True
    with torch.enable_grad():
        if label_free:
            loss = criterion(F.log_softmax(model(x), dim=1), F.softmax(model(x), dim=1))
        else:
            loss = criterion(model(x), y)
    grad = torch.autograd.grad(loss,
                                x,
                                retain_graph=False,
                                create_graph=False)[0]
    adv_x = x.detach() + eps * cha_std * grad.detach().sign() 

    return adv_x

