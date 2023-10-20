from scipy.io import loadmat, savemat
import scipy.io as io
import numpy as np
from scipy import signal
import os
from utils.data_align import centroid_align

data_path = "/data1/cxq/data/epfl/p300soft/"

save_file = "/data1/cxq/data/processedepfl/s{}_{}.mat"
if not os.path.exists("/data1/cxq/data/processedepfl/"):
    os.makedirs("/data1/cxq/data/processedepfl/")

for sub in range(4):
    sub_data = []
    sub_label = []
    for sess in range(4):
        data = data_path + f"s{sub+1}_{sess+1}.mat"
        data = loadmat(data)

        sess_data = []
        sess_label = []
        for run in range(6):
            x = data['runs'][0,run]['x'][0,0].transpose((2,0,1))[:,None,:,:]
            y = data['runs'][0,run]['y'][0,0].reshape(-1)
            sess_data.append(x)
            sess_label.append(y)
        sess_data = np.concatenate(sess_data)
        sess_label = np.concatenate(sess_label)
        print(sub,sess_data.shape,sess_label.shape,sess_data.mean(),sess_data.std())
        io.savemat(save_file.format(sub,sess), {'x': sess_data, 'y': sess_label})

for sub in range(5,9):
    sub_data = []
    sub_label = []
    for sess in range(4):
        data = data_path + f"s{sub+1}_{sess+1}.mat"
        data = loadmat(data)

        sess_data = []
        sess_label = []
        for run in range(6):
            x = data['runs'][0,run]['x'][0,0].transpose((2,0,1))[:,None,:,:]
            y = data['runs'][0,run]['y'][0,0].reshape(-1)
            sess_data.append(x)
            sess_label.append(y)
        sess_data = np.concatenate(sess_data)
        sess_label = np.concatenate(sess_label)
        print(sub-1,sess_data.shape,sess_label.shape,sess_data.mean(),sess_data.std())
        io.savemat(save_file.format(sub-1,sess), {'x': sess_data, 'y': sess_label})



save_file = "/data1/cxq/data/processedepflsea/s{}_{}.mat"
if not os.path.exists("/data1/cxq/data/processedepflsea/"):
    os.makedirs("/data1/cxq/data/processedepflsea/")
ca = centroid_align(center_type='euclid', cov_type='lwf')
for sub in range(4):
    sub_data = []
    sub_label = []
    for sess in range(4):
        data = data_path + f"s{sub+1}_{sess+1}.mat"
        data = loadmat(data)

        sess_data = []
        sess_label = []
        for run in range(6):
            x = data['runs'][0,run]['x'][0,0].transpose((2,0,1))[:,None,:,:]
            y = data['runs'][0,run]['y'][0,0].reshape(-1)
            sess_data.append(x)
            sess_label.append(y)
        sess_data = np.concatenate(sess_data)
        cov_new, sess_data = ca.fit_transform(np.squeeze(sess_data))
        sess_data = sess_data[:,None,:,:]
        sess_label = np.concatenate(sess_label)
        print(sub,sess_data.shape,sess_label.shape,sess_data.mean(),sess_data.std())
        io.savemat(save_file.format(sub,sess), {'x': sess_data, 'y': sess_label})

for sub in range(5,9):
    sub_data = []
    sub_label = []
    for sess in range(4):
        data = data_path + f"s{sub+1}_{sess+1}.mat"
        data = loadmat(data)

        sess_data = []
        sess_label = []
        for run in range(6):
            x = data['runs'][0,run]['x'][0,0].transpose((2,0,1))[:,None,:,:]
            y = data['runs'][0,run]['y'][0,0].reshape(-1)
            sess_data.append(x)
            sess_label.append(y)
        sess_data = np.concatenate(sess_data)
        cov_new, sess_data = ca.fit_transform(np.squeeze(sess_data))
        sess_data = sess_data[:,None,:,:]
        sess_label = np.concatenate(sess_label)
        print(sub-1,sess_data.shape,sess_label.shape,sess_data.mean(),sess_data.std())
        io.savemat(save_file.format(sub-1,sess), {'x': sess_data, 'y': sess_label})


