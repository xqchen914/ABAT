import torch
import logging
import attack_lib
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.pytorch_utils import bca_score, adjust_learning_rate
from autoattack import AutoAttack

class Trainer:
    def __init__(self, modelF, modelC, optimizer, args, **kwargs):
        self.modelF = modelF
        self.modelC = modelC
        self.optimizer = optimizer
        self.device = args.device
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.CE_loss = nn.CrossEntropyLoss().to(self.device)
        self.AT_eps = args.AT_eps
        self.seed = kwargs['seed']
        self.train_type = args.train
        

    def train(self, train_loader, test_loader):
        for epoch in range(self.epochs):
            # model training
            adjust_learning_rate(optimizer=self.optimizer, epoch=epoch + 1)
            self.modelF.train()
            self.modelC.train()
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(
                    self.device)
                
                if self.train_type == 'ATchastd':
                    if epoch >= 20:
                        batch_x = attack_lib.PGD_batch_cha(nn.Sequential(self.modelF, self.modelC),
                                                    batch_x,
                                                    batch_y,
                                                    eps=self.AT_eps,#e,
                                                    alpha=self.AT_eps/5,#torch.round(e/5, decimals=4).to(self.device),
                                                    steps=10,
                                                    label_free=False)
                        self.modelF.train()
                        self.modelC.train()
                elif self.train_type == 'ATchastd_fgsm':
                    if epoch >= 20:
                        batch_x = attack_lib.FGSM_batch_cha(nn.Sequential(self.modelF, self.modelC),
                                                    batch_x,
                                                    batch_y,
                                                    eps=self.AT_eps,
                                                    label_free=False)
                        self.modelF.train()
                        self.modelC.train()


                out = self.modelC(self.modelF(batch_x))
                loss = self.CE_loss(out, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.modelF.MaxNormConstraint()
                self.modelC.MaxNormConstraint()

            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs:
                train_loss, train_acc, train_bca = self.test(train_loader)
                test_loss, test_acc, test_bca = self.test(test_loader)

                logging.info(
                    'Epoch {}/{}: train_loss: {:.4f} train acc: {:.4f} train bca: {:.2f} | test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
                    .format(epoch + 1, self.epochs, train_loss, train_acc, train_bca, test_loss,
                            test_acc, test_bca))
        torch.save(self.modelC.state_dict(),
                    self.model_path + '/modelC.pt')
        torch.save(self.modelF.state_dict(),
                    self.model_path + '/modelF.pt')

    def test(self, data_loader):
        self.modelF.eval()
        self.modelC.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)
        correct = 0
        loss = 0
        labels, preds = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.modelC(self.modelF(x))
                pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
                loss += criterion(out, y).item()
                correct += pred.eq(y.cpu().view_as(pred)).sum().item()
                labels.extend(y.cpu().tolist())
                preds.extend(pred.tolist())
        acc = correct / len(data_loader.dataset)
        bca = bca_score(labels, preds)
        loss /= len(data_loader.dataset)

        return loss, acc, bca
    
    def adv_test(self,
                    data_loader,
                    attack='FGSM',
                    eps=0.01,
                    distance='Linf'):
        x, y = data_loader.dataset.tensors[0], data_loader.dataset.tensors[1]
        if attack == 'FGSM':
            adv_x = attack_lib.FGSM_cha(nn.Sequential(self.modelF, self.modelC),
                                    x,
                                    y,
                                    eps=eps)
        elif attack == 'PGD':
            adv_x = attack_lib.PGD_cha(nn.Sequential(self.modelF, self.modelC),
                                   x,
                                   y,
                                   eps=eps,
                                   alpha=eps / 10,
                                   steps=20)
        elif attack == 'autoPGD':
            adversary = AutoAttack(nn.Sequential(self.modelF, self.modelC), norm=distance,seed=self.seed, eps=eps,
                version='',attacks_to_run=['apgd-ce'],device=self.device,n_iter=20)
            adv_x = adversary.run_standard_evaluation_individual(x,
                y, bs=data_loader.batch_size)
            adv_x = adv_x['apgd-ce']
        adv_loader = DataLoader(dataset=TensorDataset(adv_x.cpu(),
                                                      y),
                                batch_size=data_loader.batch_size,
                                shuffle=False,
                                drop_last=False,num_workers=1)
        adv_loss, adv_acc, adv_bca = self.test(adv_loader)

        return adv_loss, adv_acc, adv_bca
