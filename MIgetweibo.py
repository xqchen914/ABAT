from scipy.io import loadmat, savemat
import scipy.io as io
import numpy as np
from scipy import signal
from scipy.signal import resample
import os
import mne
from utils.data_align import centroid_align

data_path = "/data1/cxq/data/MNE-weibo-2014/subject_{}.mat"
fs = 200
MI_duration = 4 #4s
Fstop1 = 4
Fstop2 = 32
save_file = "/data1/cxq/data//processed-weibo-2014/s{}.mat"

if not os.path.exists("/data1/cxq/data/processed-weibo-2014/"):
    os.makedirs("/data1/cxq/data//processed-weibo-2014/")

for sub in range(10):

    data = loadmat(data_path.format(sub+1))
    x = data['data'].transpose((2,0,1))[:-80,None,:,600:1400]
    y = data['label'].reshape(-1) - 1
    y = y[:-80]

    b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')  # 5阶巴特沃斯滤波器
    x = signal.filtfilt(b, a, x, axis=-1)
    
    print(sub,'x',x.shape,y.shape,x.mean(),x.std())
    io.savemat(save_file.format(sub), {'x': x, 'y': y})







save_file = "/data1/cxq/data/processed-weibo-2014subea/s{}.mat"

if not os.path.exists("/data1/cxq/data/processed-weibo-2014subea/"):
    os.makedirs("/data1/cxq/data//processed-weibo-2014subea/")
ca = centroid_align(center_type='euclid', cov_type='lwf')
for sub in range(10):

    data = loadmat(data_path.format(sub+1))
    x = data['data'].transpose((2,0,1))[:-80,None,:,600:1400]
    y = data['label'].reshape(-1) - 1
    y = y[:-80]

    # x = signal.detrend(x, axis=-1, type='linear', )
    #
    b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')  # 5阶巴特沃斯滤波器
    x = signal.filtfilt(b, a, x, axis=-1)

    cov_new, xx = ca.fit_transform(np.squeeze(x))
    x = xx[:,None,:,:]
    #
    # x = resample(x, int(x.shape[-1] * 128 / fs),axis=-1)

    print(sub,'x',x.shape,y.shape,x.mean(),x.std())
    io.savemat(save_file.format(sub), {'x': x, 'y': y})