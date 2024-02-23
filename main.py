import os
import logging
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
import numpy as np
import openpyxl
import pandas as pd
from pandas import Series,DataFrame
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from models import LoadModel
from utils.data_loader import *
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes
import train



def run(x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor,
        y_test: torch.Tensor, args):
    # initialize the model
    args.num_classes = len(np.unique(y.numpy()))
    modelF, modelC, modelD, args.embed_dim = LoadModel(
        model_name=args.model,
        Classes=args.num_classes,
        Chans=x.shape[2],
        Samples=x.shape[3])
    modelF.apply(init_weights).to(args.device)
    modelC.apply(init_weights).to(args.device)
    modelD.apply(init_weights).to(args.device)


    if args.FT:
        
        model_save_path = os.path.join(model_load_path, f'{r}/{t}')
        modelF.load_state_dict(
            torch.load(model_save_path +
                        '/modelF.pt',
                        map_location=lambda storage, loc: storage))
        modelC.load_state_dict(
            torch.load(model_save_path +
                        '/modelC.pt',
                        map_location=lambda storage, loc: storage))



    # trainable parameters
    params = []
    for _, v in modelF.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    for _, v in modelC.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    sample_weights = weight_for_balanced_classes(y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    train_loader = DataLoader(dataset=TensorDataset(x, y),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False,num_workers=1)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False,num_workers=1)

    trainer = train.Trainer(modelF,modelC,
                                        optimizer,
                                        args,
                                        seed=r,sf=sf_dict[args.dataset])
    trainer.train(train_loader, test_loader)

    # normal test
    test_loss, test_acc, test_bca = trainer.test(test_loader)
    logging.info('test_loss: {:.4f} test acc: {:.4f} test bca: {:.2f}'
                    .format(test_loss, test_acc, test_bca))
    recorder_bca[round(r*(len(args.attacks)*len(args.epss)+1)),t] = test_bca
    recorder_acc[round(r*(len(args.attacks)*len(args.epss)+1)),t] = test_acc

    # robust test
    adv_accs, adv_bcas = [], []
    for attack in range(len(args.attacks)):
        for eps in range(len(args.epss)):
            adv_loss, adv_acc, adv_bca = trainer.adv_test(test_loader,
                                                   attack=args.attacks[attack],
                                                   eps=args.epss[eps])
            recorder_bca[round(r*(len(args.attacks)*len(args.epss)+1))+round((attack)*len(args.epss))+eps+1,t] = adv_bca
            recorder_acc[round(r*(len(args.attacks)*len(args.epss)+1))+round((attack)*len(args.epss))+eps+1,t] = adv_acc
            adv_accs.append(adv_acc)
            adv_bcas.append(adv_bca)
            logging.info(
                f'{args.attacks[attack]} {args.epss[eps]} adv loss: {adv_loss}, adv acc: {adv_acc}, adv bca: {adv_bca}')

    return test_acc, test_bca, adv_accs, adv_bcas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='7')
    parser.add_argument('--model', type=str, default='ShallowCNN') #ShallowCNN DeepCNN EEGNet
    parser.add_argument('--dataset', type=str, default='MI2014001')
    parser.add_argument('--cross_balance', type=int, default=0)
    
    parser.add_argument('--attacks', nargs='+',default=['FGSM', 'PGD'])#['FGSM', 'PGD']
    parser.add_argument('--epss', nargs='+',type=float,default=[0.001,0.003,0.005,0.01,0.03,0.05])#[0.001,0.003,0.005,0.01,0.03,0.05]


    parser.add_argument('--setup', type=str, default='within')
    parser.add_argument('--ea', type=str, default='sess') # no sub sess

    parser.add_argument('--train', type=str,
                        default='NT')  # NT AT TL Dist TAT ATreg ATchastd NTaug ATchastd_fgsm
    parser.add_argument('--FT', type=int,
                        default=0)
    parser.add_argument('--AT_eps', type=float,
                        default=0.01) 
    parser.add_argument('--transform', type=str,
                        default='freqshift') #'guassian'
    
                        
    parser.add_argument('--lambd', type=float, default=0.2)

    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--commit', type=str, default='test')

    args = parser.parse_args()


    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    if args.setup == 'cross':
        args.batch_size = 128 
    elif args.setup == 'within':
        args.batch_size = 32
    
    if args.dataset == 'EPFL':
        args.batch_size = 128 

    if args.train == 'NT':
        args.AT_eps = 0.0 

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8, 'MI2014001':9,  'MI2014001-2':9,\
                        'MI2015001':12, 'MI2014004':9, 'MI2015004':9, 'weibo':10, 'weibocsp':10,\
                            'physionet':109, 'physionet2c':109, '2014009':10,'20140091':10}
    if args.dataset[:-1] == 'weibo':
        subject_num_dict[args.dataset] = 10

    if args.dataset[:-1] == 'ERN':
        subject_num_dict[args.dataset] = 16
    

    sf_dict = {'MI2014001':250, 'weibo':200 ,'ERN': 200,'EPFL':128, '2014009':256,'20140091':256}
    if args.dataset[:-1] == 'weibo':
        sf_dict[args.dataset] = 200
    
    if args.dataset[:-1] == 'ERN':
        sf_dict[args.dataset] = 200

    # ========================model path=======================
    model_path = f'/data1/cxq/model_align/{args.train}/{args.AT_eps}/target/{args.dataset}/{args.model}/{args.setup}/{args.ea}'
    if args.FT:
        model_path = f'/data1/cxq/model_align/{args.train}/FT/{args.AT_eps}/target/{args.dataset}/{args.model}/{args.setup}/{args.ea}'
        model_load_path = f'/data1/cxq/model_align/ATchastd/0.05/target/{args.dataset}/{args.model}/cross/sess'
        model_load_path = f'/data1/cxq/model_align/NT/0.0/target/{args.dataset}/{args.model}/cross/{args.ea}'


    method_name = args.train
    if args.FT:
        method_name += 'FT'

    # ========================log name and excel name=======================
    log_path = f'/home/xqchen/attack_align/result/log/{method_name}/{args.AT_eps}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'{args.setup}_{args.dataset}_{args.ea}_{args.model}.log')
    # system time
    args.commit = datetime.datetime.now()
    log_name = log_name.replace('.log', f'_{args.commit}.log')

    excel_path = f'/home/xqchen/attack_align/result/excel/{method_name}/{args.AT_eps}'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    excel_name = os.path.join(excel_path,f'{args.setup}_{args.dataset}_{args.ea}_{args.model}.xlsx')
    excel_name = excel_name.replace('.xlsx', f'_{args.commit}.xlsx')



    recorder_bca = np.zeros(((len(args.attacks)*len(args.epss)+1)*args.repeat,subject_num_dict[args.dataset]))
    recorder_acc = np.zeros(((len(args.attacks)*len(args.epss)+1)*args.repeat,subject_num_dict[args.dataset]))

    
    # ========================logging========================
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')
    if args.FT:
        logging.info(model_load_path)


    # ========================model train========================
    r_acc, r_bca, r_adv_acc, r_adv_bca = [], [], [], []
    for r in range(args.repeat):
        seed(r)
        # model train
        acc_list = []
        bca_list = []
        adv_acc_list = []
        adv_bca_list = []
        for t in range(subject_num_dict[args.dataset]):
            # build model path
            model_save_path = os.path.join(model_path, f'{r}/{t}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            args.model_path = model_save_path

            logging.info(f'subject id: {t}')
            # load data

            if args.dataset == 'EPFL':
                x_train, y_train, x_test, y_test = epflLoad(id=t,
                                                            setup=args.setup, ea=args.ea, cross_balance=args.cross_balance)

            elif args.dataset == '2014009' or args.dataset == '20140091':
                x_train, y_train, x_test, y_test = p3002014009Load(id=t,
                                                            setup=args.setup, ea=args.ea, dataset=args.dataset ,cross_balance=args.cross_balance)
            
            elif args.dataset == 'MI2014001':
                x_train, y_train, x_test, y_test = MI2014001Load(id=t,
                                                            setup=args.setup, ea=args.ea)
            elif args.dataset[:-1] == 'weibo':
                x_train, y_train, x_test, y_test = weiboLoad(id=t,
                                                            setup=args.setup, ea=args.ea, p = int(args.dataset[-1])) 
            elif args.dataset == 'weibo':
                x_train, y_train, x_test, y_test = weiboLoad(id=t,
                                                            setup=args.setup, ea=args.ea)  
            elif args.dataset == 'bcimi':
                x_train, y_train, x_test, y_test = bcimiLoad(id=t,
                                                            setup=args.setup, ea=args.ea)  



            logging.info(f'train: {x_train.shape},{x_train.mean()},{x_train.std()}, test: {x_test.shape},{x_test.mean()},{x_test.std()}')
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            test_acc, test_bca, adv_acc, adv_bca = run(x_train, y_train,
                                                       x_test, y_test, args)
            acc_list.append(test_acc)
            bca_list.append(test_bca)
            adv_acc_list.append(adv_acc)
            adv_bca_list.append(adv_bca)

        r_acc.append(acc_list)
        r_bca.append(bca_list)
        r_adv_acc.append(adv_acc_list)
        r_adv_bca.append(adv_bca_list)

        

        logging.info(f'Repeat {r + 1}')
        logging.info(
            f'Mean acc: {np.mean(acc_list)} | Mean bca: {np.mean(bca_list)}')
        logging.info(
            f'Mean adv acc: {np.mean(adv_acc_list, axis=0)} | Mean adv bca: {np.mean(adv_bca_list, axis=0)}'
        )


    recorder_bca = DataFrame(recorder_bca,
               columns = [f's{g}' for g in range(subject_num_dict[args.dataset])],
               index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)],
                                                   ['normal']+[i+str(j) for i in args.attacks for j in args.epss]]))
    recorder_acc = DataFrame(recorder_acc,
            columns = [f's{g}' for g in range(subject_num_dict[args.dataset])],
            index = pd.MultiIndex.from_product([[f'repeat_{g}' for g in range(args.repeat)],
                                                ['normal']+[i+str(j) for i in args.attacks for j in args.epss]]))


    # 使用ExcelWriter创建Excel文件
    with pd.ExcelWriter(excel_name) as writer:
        # 将DataFrame保存到指定工作表
        recorder_bca.to_excel(writer, sheet_name='bca')
        recorder_acc.to_excel(writer, sheet_name='acc')

    r_acc, r_adv_acc = np.mean(r_acc, 1), np.mean(r_adv_acc, axis=1)
    r_bca, r_adv_bca = np.mean(r_bca, 1), np.mean(r_adv_bca, axis=1)
    r_adv_acc, r_adv_bca = np.array(r_adv_acc), np.array(r_adv_bca)
    logging.info('*' * 50)
    logging.info(
        f'Repeat mean acc | bca: {round(np.mean(r_acc), 4)}-{round(np.std(r_acc), 4)} | {round(np.mean(r_bca), 4)}-{round(np.std(r_bca), 4)}'
    )
    for k in range(len(args.attacks)):
        for i in range(len(args.epss)):
            logging.info(
                f'Repeat mean adv acc | bca ({args.attacks[k]} {args.epss[i]}): {round(np.mean(r_adv_acc[:, round(k*len(args.epss)+i)]), 4)}-{round(np.std(r_adv_acc[:, round(k*len(args.epss)+i)]), 4)} | {round(np.mean(r_adv_bca[:, round(k*len(args.epss)+i)]), 4)}-{round(np.std(r_adv_bca[:, round(k*len(args.epss)+i)]), 4)}'
            )
    # print params
