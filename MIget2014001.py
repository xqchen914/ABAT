from scipy.io import loadmat, savemat
import scipy.io as io
import numpy as np
from scipy import signal
from scipy.signal import resample
import os
import mne
from re import T
from typing import Optional
from scipy import signal
import scipy.io as scio
import numpy as np
from utils.data_align import centroid_align

data_path = "/data1/cxq/data/2014001/BCICIV_2a_gdf/"
fs = 250
MI_duration = 4 #4s
Fstop1 = 8
Fstop2 = 32
save_file = "/data1/cxq/data/processedMI2014001/s{}{}.mat"

if not os.path.exists("/data1/cxq/data/processedMI2014001/"):
    os.makedirs("/data1/cxq/data/processedMI2014001/")

def get_one_sub(sid):
    """
    extract mi period
    """
    X = []
    Y = []

    tdata = data_path + f"A0{sid}T.gdf"
    tdata = mne.io.read_raw_gdf(tdata, preload=True, exclude=['EOG-left', 'EOG-central', 'EOG-right'])

    event_position = tdata.annotations.onset  # 事件位置列表
    event_type = tdata.annotations.description  # 事件名称
    x = tdata.to_data_frame()
    
    x = x.T.to_numpy()[:22,:]
    for j, t in enumerate(event_type):
        if t=='769' or t=='770' or t=='771' or t=='772':
            X.append(x[None, :, round((event_position[j]) * fs):round((event_position[j]+4) * fs)])
            Y.append(int(t)-769)

    X = np.concatenate(X)
    Y = np.array(Y)

    b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')  # 5阶巴特沃斯滤波器
    X = signal.filtfilt(b, a, X, axis=-1)

    print('x_train',X.shape,Y.shape)
    io.savemat(save_file.format(sid-1,"E"), {'x': X[:, None, :, :], 'y': Y})

    X = []
    Y = []
    edata = data_path + f"A0{sid}E.gdf"
    edata = mne.io.read_raw_gdf(edata, preload=True, exclude=['EOG-left', 'EOG-central', 'EOG-right'])
    label_pathE = "/data1/cxq/data/2014001//true_labels/" + "A0" + str(sid) + "E.mat"
    true_labelsE = io.loadmat(label_pathE)
    event_position = edata.annotations.onset  # 事件位置列表
    event_type = edata.annotations.description  # 事件名称

    x = edata.to_data_frame().T.to_numpy()[:22,:]

    index = 0
    for j, t in enumerate(event_type):
        label_nowE = int(np.array(true_labelsE['classlabel'][index]))
        if t == '783':
            if label_nowE == 1 or label_nowE == 2 or label_nowE == 3 or label_nowE == 4:
                index += 1
                X.append(x[None, :, round((event_position[j]) * fs):round((event_position[j] + 4) * fs)])
                Y.append(int(label_nowE) - 1)

    X = np.concatenate(X)
    Y = np.array(Y)

    b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')  # 5阶巴特沃斯滤波器
    X = signal.filtfilt(b, a, X, axis=-1)

    print('x_t',X.shape,Y.shape)
    io.savemat(save_file.format(sid-1,"T"), {'x': X[:, None, :, :], 'y': Y})





def MI2014001save():
    data_path = '/data1/cxq/data/processedMI2014001/'
    save_file = "/data1/cxq/data/processedMI2014001_4s_sea/s{}.mat"

    if not os.path.exists("/data1/cxq/data/processedMI2014001_4s_sea/"):
        os.makedirs("/data1/cxq/data/processedMI2014001_4s_sea/")

    ca = centroid_align(center_type='euclid', cov_type='lwf')
    for i in range(9):

        data = scio.loadmat(data_path + f's{i}E.mat')
        x1, y1 = data['x'], data['y']
        y1 = np.squeeze(y1.flatten())

        cov_new, x1 = ca.fit_transform(np.squeeze(x1))
        x1 = x1[:,None,:,:]

        print(i,x1.shape,x1.std(),x1.mean())
        print(y1.shape)

        save_file = "/data1/cxq/data/processedMI2014001_4s_sea/s{}E.mat"
        io.savemat(save_file.format(i), {'x': x1, 'y': y1.reshape(-1)})


        data = scio.loadmat(data_path + f's{i}T.mat')
        x2, y2 = data['x'], data['y']
        y2 = np.squeeze(y2.flatten())

        cov_new, x2 = ca.fit_transform(np.squeeze(x2))
        x2 = x2[:,None,:,:]

        
        print(i,x2.shape,x2.std(),x2.mean())
        print(y2.shape)

        save_file = "/data1/cxq/data/processedMI2014001_4s_sea/s{}T.mat"
        io.savemat(save_file.format(i), {'x': x2, 'y': y2.reshape(-1)})

    return #x_train, y_train.squeeze(), x_test, y_test.squeeze()



if __name__ == "__main__":
    for i in range(1,10):
        get_one_sub(i)
    MI2014001save()
    