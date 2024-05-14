import moabb
from moabb.datasets import BNCI2014009
from moabb.paradigms import P300
import os
import scipy.io as io
import numpy as np
import pickle
import pandas as pd

dataset = BNCI2014009()
dataset.subject_list = dataset.subject_list[:]
datasets = [dataset]
paradigm = P300()



subject_list = dataset.subject_list[:]
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject_list)

with open('/data1/cxq/data/processed2014009/erp2014009x.pkl', 'wb') as file:
    pickle.dump(X, file)

with open('/data1/cxq/data/processed2014009/erp2014009y.pkl', 'wb') as file:
    pickle.dump(labels, file)

# np.savetxt('/data1/cxq/data/processed2014009/meta.csv', meta, delimiter=',')
meta.to_csv('/data1/cxq/data/processed2014009/meta.csv', index=False)

with open('/data1/cxq/data/processed2014009/erp2014009x.pkl', 'rb') as file:
    X1 = pickle.load(file)

with open('/data1/cxq/data/processed2014009/erp2014009y.pkl', 'rb') as file:
    labels1 = pickle.load(file)
# 加载CSV文件为NumPy数组
# meta1 = np.loadtxt('/data1/cxq/data/processed2014009/meta.csv', delimiter=',')
meta1 = pd.read_csv('/data1/cxq/data/processed2014009/meta.csv')
print(1)

save_file = "/data1/cxq/data/processed2014009/s{}_{}.mat"
if not os.path.exists("/data1/cxq/data/processed2014009/"):
    os.makedirs("/data1/cxq/data/processed2014009/")
for i in subject_list:
    x = X1[meta1['subject']==i]
    lab = labels1[meta1['subject']==i]
    m = meta1[meta1['subject']==i]

    lab[lab=='NonTarget'] = 0
    lab[lab=='Target'] = 1
    lab = lab.astype('int')

    sess  = m['session'].unique()
    sess_data = []
    sess_lab = []
    for j in range(len(sess)):
        sess_data.append(x[m['session']==sess[j],None,:,:])
        sess_lab.append(lab[m['session']==sess[j]])
        print(i,x[m['session']==sess[j],None,:,:].mean(),\
              x[m['session']==sess[j],None,:,:].std(),x[m['session']==sess[j],None,:,:].shape,\
                np.bincount(lab[m['session']==sess[j]]),\
                    len(lab[m['session']==sess[j]]))
        io.savemat(save_file.format(i-1,j), {'x': x[m['session']==sess[j],None,:,:], 'y': lab[m['session']==sess[j]]})



from utils.data_align import centroid_align
save_file = "/data1/cxq/data/processed2014009sea/s{}_{}.mat"
if not os.path.exists("/data1/cxq/data/processed2014009sea/"):
    os.makedirs("/data1/cxq/data/processed2014009sea/")
ca = centroid_align(center_type='euclid', cov_type='lwf')
for i in subject_list:
    x = X1[meta1['subject']==i]
    lab = labels1[meta1['subject']==i]
    m = meta1[meta1['subject']==i]

    lab[lab=='NonTarget'] = 0
    lab[lab=='Target'] = 1
    lab = lab.astype('int')

    sess  = m['session'].unique()
    sess_data = []
    sess_lab = []
    for j in range(len(sess)):
        d = x[m['session']==sess[j],None,:,:]
        l = lab[m['session']==sess[j]]

        cov_new, d = ca.fit_transform(np.squeeze(d))
        d = d[:,None,:,:]


        sess_data.append(d)
        sess_lab.append(l)

        print(i,d.mean(),\
              d.std(),d.shape,\
                np.bincount(l),\
                    len(l))
        io.savemat(save_file.format(i-1,j), {'x': d, 'y': l})


