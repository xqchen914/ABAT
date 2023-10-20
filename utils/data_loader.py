from re import T
from typing import Optional
from scipy import signal
from scipy.signal import resample
import scipy.io as scio
import numpy as np
from utils.data_align import centroid_align


def split(x, y):
    idx = np.arange(len(x))
    train_size = 240

    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[
        idx[train_size:]]




def MI2014001Load(id: int, setup: Optional[str] = 'within',ea: Optional[str] = 'no', online: Optional[int] = 0):
    if  ea == 'sess':
        data_path = '/data1/cxq/data/processedMI2014001_4s_sea/'
    elif ea == 'no':
        data_path = '/data1/cxq/data/processedMI2014001/'
    
    if online:
        data_path = '/data1/cxq/data/processedMI2014001/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
        
        data = scio.loadmat(data_path + f's{id}E.mat')
        x_train, y_train = data['x'], data['y']
        y_train = np.squeeze(y_train.flatten())

        data = scio.loadmat(data_path + f's{id}T.mat')
        x_test, y_test = data['x'], data['y']
        y_test = np.squeeze(y_test.flatten())

    elif setup == 'cross':
        for i in range(9):

            data = scio.loadmat(data_path + f's{i}E.mat')
            x1, y1 = data['x'], data['y']
            y1 = np.squeeze(y1.flatten())

            # x1 = x1[:,None,:,:]

            data = scio.loadmat(data_path + f's{i}T.mat')
            x2, y2 = data['x'], data['y']
            y2 = np.squeeze(y2.flatten())

            # x2 = x2[:,None,:,:]
        
            x = np.concatenate((x1,x2))

            y = np.concatenate((y1,y2))
            if i == id:
                x_test, y_test = x, y
            else:
                x_train.append(x)
                y_train.append(y)

        x_train = np.concatenate(x_train)
        y_train = np.hstack(y_train)
        
        print(x_train.shape)
        print(y_train.shape)
    else:
        raise Exception('No such Experiment setup.')

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()



def weiboLoad(id: int, setup: Optional[str] = 'within',ea: Optional[str] = 'no',p=2):
    if ea == 'sub':
        data_path = '/data1/cxq/data/processed-weibo-2014subea/'
    elif ea == 'no':
        data_path = '/data1/cxq/data/processed-weibo-2014/'
    elif ea == 'sess':
        data_path = '/data1/cxq/data/processed-weibo-2014sea/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
            data = scio.loadmat(data_path + f's{id}.mat')
            x, y = data['x'], data['y']
            y = np.squeeze(np.array(y).flatten())
            # y -= 1

            x_train, x_test = x[:round(60*p)], x[round(60*p):]
            y_train, y_test = y[:round(60*p)], y[round(60*p):]
    elif setup == 'cross':
        for i in range(10):

            data = scio.loadmat(data_path + f's{i}.mat')
            x, y = data['x'], data['y']
            y = np.squeeze(np.array(y).flatten())
            # y -= 1
            
            if i == id:
                x_test, y_test = x, y
            else:
                x_train.append(x)
                y_train.append(y)

        x_train = np.concatenate(x_train)
        y_train = np.hstack(y_train)
        
        print(x_train.shape)
        print(y_train.shape)
    else:
        raise Exception('No such Experiment setup.')

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()



def epflLoad(id: int, setup: Optional[str] = 'within',ea: Optional[str] = 'no', cross_balance: Optional[int] = 0, online: Optional[int] = 0):
    if  ea == 'no':
        data_path = '/data1/cxq/data/processedepfl/'
    elif ea == 'sess':
        data_path = '/data1/cxq/data/processedepflsea/'
    if online:
        data_path = '/data1/cxq/data/processedepfl/'

    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
        
        data = scio.loadmat(data_path + f's{id}_0.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(np.array(y1).flatten())

        data = scio.loadmat(data_path + f's{id}_1.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(np.array(y2).flatten())

        x_train = np.concatenate((x1,x2))
        y_train = np.concatenate((y1,y2))

        data = scio.loadmat(data_path + f's{id}_2.mat')
        x3, y3 = data['x'], data['y']
        y3 = np.squeeze(np.array(y3).flatten())

        data = scio.loadmat(data_path + f's{id}_3.mat')
        x4, y4 = data['x'], data['y']
        y4 = np.squeeze(np.array(y4).flatten())

        x_test = np.concatenate((x3,x4))
        y_test = np.concatenate((y3,y4))

    elif setup == 'cross':
        for i in range(8):

            data = scio.loadmat(data_path + f's{i}_0.mat')
            x1, y1 = data['x'], data['y']
            y1 = np.squeeze(y1.flatten())

            data = scio.loadmat(data_path + f's{i}_1.mat')
            x2, y2 = data['x'], data['y']
            y2 = np.squeeze(y2.flatten())

            data = scio.loadmat(data_path + f's{id}_2.mat')
            x3, y3 = data['x'], data['y']
            y3 = np.squeeze(np.array(y3).flatten())

            data = scio.loadmat(data_path + f's{id}_3.mat')
            x4, y4 = data['x'], data['y']
            y4 = np.squeeze(np.array(y4).flatten())
        
            x = np.concatenate((x1,x2,x3,x4))
            y = np.concatenate((y1,y2,y3,y4))

            

            if i == id:
                x_test, y_test = x, y
            else:
                if cross_balance:
                    ind0 = np.where(y==0)[0]
                    ind1 = np.where(y==1)[0]
                    ind0 = np.random.choice(ind0,len(ind1),replace=False)

                    ind = np.concatenate((ind0,ind1))

                    x = x[ind]
                    y = y[ind]
                x_train.append(x)
                y_train.append(y)

        x_train = np.concatenate(x_train)
        y_train = np.hstack(y_train)
        
        print(x_train.shape)
        print(y_train.shape)
    else:
        raise Exception('No such Experiment setup.')

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()


def p3002014009Load(id: int, setup: Optional[str] = 'within',ea: Optional[str] = 'no',\
                     dataset: Optional[str] = '2014009', cross_balance: Optional[int] = 0, online: Optional[int] = 0):
    if  ea == 'no':
        data_path = '/data1/cxq/data/processed2014009/'
    elif ea == 'sess':
        data_path = '/data1/cxq/data/processed2014009sea/'

    if online:
        data_path = '/data1/cxq/data/processed2014009/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
        
        if dataset == '2014009':
            data = scio.loadmat(data_path + f's{id}_0.mat')
            x1, y1 = data['x'], data['y']
            y1 = np.squeeze(np.array(y1).flatten())

            data = scio.loadmat(data_path + f's{id}_1.mat')
            x2, y2 = data['x'], data['y']
            y2 = np.squeeze(np.array(y2).flatten())

            x_train = np.concatenate((x1,x2))
            y_train = np.concatenate((y1,y2))

            data = scio.loadmat(data_path + f's{id}_2.mat')
            x3, y3 = data['x'], data['y']
            y3 = np.squeeze(np.array(y3).flatten())


            x_test = x3
            y_test = y3
        elif dataset == '20140091':
            data = scio.loadmat(data_path + f's{id}_0.mat')
            x1, y1 = data['x'], data['y']
            y1 = np.squeeze(np.array(y1).flatten())

            x_train = x1
            y_train = y1

            data = scio.loadmat(data_path + f's{id}_1.mat')
            x2, y2 = data['x'], data['y']
            y2 = np.squeeze(np.array(y2).flatten())

            data = scio.loadmat(data_path + f's{id}_2.mat')
            x3, y3 = data['x'], data['y']
            y3 = np.squeeze(np.array(y3).flatten())

            x_test = np.concatenate((x2,x3))
            y_test = np.concatenate((y2,y3))


    elif setup == 'cross':
        for i in range(10):

            data = scio.loadmat(data_path + f's{i}_0.mat')
            x1, y1 = data['x'], data['y']
            y1 = np.squeeze(y1.flatten())

            data = scio.loadmat(data_path + f's{i}_1.mat')
            x2, y2 = data['x'], data['y']
            y2 = np.squeeze(y2.flatten())

            data = scio.loadmat(data_path + f's{id}_2.mat')
            x3, y3 = data['x'], data['y']
            y3 = np.squeeze(np.array(y3).flatten())

        
            x = np.concatenate((x1,x2,x3))
            y = np.concatenate((y1,y2,y3))

            

            if i == id:
                x_test, y_test = x, y
            else:
                if cross_balance:
                    ind0 = np.where(y==0)[0]
                    ind1 = np.where(y==1)[0]
                    ind0 = np.random.choice(ind0,len(ind1),replace=False)

                    ind = np.concatenate((ind0,ind1))

                    x = x[ind]
                    y = y[ind]
                x_train.append(x)
                y_train.append(y)

        x_train = np.concatenate(x_train)
        y_train = np.hstack(y_train)
        
        print(x_train.shape)
        print(y_train.shape)
    else:
        raise Exception('No such Experiment setup.')

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()