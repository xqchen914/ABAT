from mne.io import concatenate_raws, read_raw_gdf
import os
import scipy.io as io
from scipy import signal
from scipy.signal import resample
import numpy as np

path = '/data1/cxq/data/BCI_Database/Signals/DATA A/'
files = os.listdir('/data1/cxq/data/BCI_Database/Signals/DATA A/')
from scipy.signal import resample


fs = 512
re_fs = 128
Fstop1 = 8
Fstop2 = 30
MI_duration = 4
b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')  # 5阶巴特沃斯滤波器

save_file = '/data1/cxq/data/bci_mi_processed/'
if not os.path.exists(save_file):
    os.makedirs(save_file)


for u in range(60):
    # if '.' in f:
    #     continue
    p = path + f'A{u+1}/'
    
    acq_x = []
    acq_y = []
    for i in range(2):
        raw_data = read_raw_gdf(p+f'A{u+1}_R{i+1}_acquisition.gdf',preload=False)
        data = raw_data.get_data()
        data = data[np.concatenate([np.arange(11),np.arange(16,32)]),:]
        
        data = signal.filtfilt(b, a, data, axis=-1)
        data = resample(data, round(data.shape[-1] / fs * re_fs),axis=-1)
        for l,t in zip(raw_data.annotations.description,raw_data.annotations.onset):
            if l=='769' or l=='770':
                acq_x.append(data[None,:,round(re_fs*t):round(re_fs*(t+MI_duration))])
                acq_y.append(int(l)-769)
        
    acq_x = np.concatenate(acq_x)
    acq_y = np.array(acq_y)
    
    
    acq_x *= 1e5
    print(u,'x_train',acq_x.shape,acq_y.shape,acq_x.mean(),acq_x.std())
    io.savemat(save_file+f's{u}_e.mat', {'x': acq_x[:, None, :, :], 'y': acq_y})
    
    t_x = []
    t_y = []
    for i in range(2,6):
        if not os.path.exists(p+f'A{u+1}_R{i+1}_onlineT.gdf'):
            continue

        
        raw_data = read_raw_gdf(p+f'A{u+1}_R{i+1}_onlineT.gdf',preload=False)
        data = raw_data.get_data()
        data = data[np.concatenate([np.arange(11),np.arange(16,32)]),:]
        
        data = signal.filtfilt(b, a, data, axis=-1)
        data = resample(data, round(data.shape[-1] / fs * re_fs),axis=-1)
        for l,t in zip(raw_data.annotations.description,raw_data.annotations.onset):
            if l=='769' or l=='770':
                t_x.append(data[None,:,round(re_fs*t):round(re_fs*(t+MI_duration))])
                t_y.append(int(l)-769)
        
    t_x = np.concatenate(t_x)
    t_y = np.array(t_y)

    t_x *= 1e5
    print(u,'x_train',t_x.shape,t_y.shape,t_x.mean(),t_x.std())
    io.savemat(save_file+f's{u}_t.mat', {'x': t_x[:, None, :, :], 'y': t_y})

        # print(1)
# raw=read_raw_edf("Affaf Ikram 20121020 1839.L1.edf",preload=False)
from utils.data_align import centroid_align
save_file = '/data1/cxq/data/bci_mi_processedsea/'
if not os.path.exists(save_file):
    os.makedirs(save_file)
ca = centroid_align(center_type='euclid', cov_type='lwf')

for u in range(60):
    # if '.' in f:
    #     continue
    p = path + f'A{u+1}/'
    
    acq_x = []
    acq_y = []
    for i in range(2):
        raw_data = read_raw_gdf(p+f'A{u+1}_R{i+1}_acquisition.gdf',preload=False)
        data = raw_data.get_data()
        data = data[np.concatenate([np.arange(11),np.arange(16,32)]),:]
        
        data = signal.filtfilt(b, a, data, axis=-1)
        data = resample(data, round(data.shape[-1] / fs * re_fs),axis=-1)
        for l,t in zip(raw_data.annotations.description,raw_data.annotations.onset):
            if l=='769' or l=='770':
                acq_x.append(data[None,:,round(re_fs*t):round(re_fs*(t+MI_duration))])
                acq_y.append(int(l)-769)
        
    acq_x = np.concatenate(acq_x)
    acq_y = np.array(acq_y)
    
    
    acq_x *= 1e5
    # print(u,'x_train',acq_x.shape,acq_y.shape,acq_x.mean(),acq_x.std())
    # print(u,'x_train',acq_x.shape,acq_y.shape)
    # io.savemat(save_file+f's{u}_e.mat', {'x': acq_x[:, None, :, :], 'y': acq_y})
    
    t_x = []
    t_y = []
    for i in range(2,6):
        if not os.path.exists(p+f'A{u+1}_R{i+1}_onlineT.gdf'):
            continue

        
        raw_data = read_raw_gdf(p+f'A{u+1}_R{i+1}_onlineT.gdf',preload=False)
        data = raw_data.get_data()
        data = data[np.concatenate([np.arange(11),np.arange(16,32)]),:]
        
        data = signal.filtfilt(b, a, data, axis=-1)
        data = resample(data, round(data.shape[-1] / fs * re_fs),axis=-1)
        for l,t in zip(raw_data.annotations.description,raw_data.annotations.onset):
            if l=='769' or l=='770':
                t_x.append(data[None,:,round(re_fs*t):round(re_fs*(t+MI_duration))])
                t_y.append(int(l)-769)
        
    t_x = np.concatenate(t_x)
    t_y = np.array(t_y)
    t_x *= 1e5

    ca.fit(np.squeeze(np.concatenate((acq_x,t_x))))
    
    _, acq_x = ca.transform(acq_x)
    _, t_x = ca.transform(t_x)
    
    print(u,'x_train',acq_x.shape,acq_y.shape,acq_x.mean(),acq_x.std())
    io.savemat(save_file+f's{u}_e.mat', {'x': acq_x[:, None, :, :], 'y': acq_y})
    
    
    print(u,'x_train',t_x.shape,t_y.shape,t_x.mean(),t_x.std())
    io.savemat(save_file+f's{u}_t.mat', {'x': t_x[:, None, :, :], 'y': t_y})
