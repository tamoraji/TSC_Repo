import numpy as np
import pyts
from pyts.metrics import dtw
import sklearn
import time

datasets = [
"Hydraulic_systems_10HZ_Multivar",
]

datasets_path = "../datasets"

for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy")
    Dataset = Dataset
    print(Dataset.shape)
    
    Labels_name = dataset + "_Labels"
    Labels = np.load(datasets_path + "/"  + Labels_name + ".npy")
    Labels = Labels.squeeze()
    print(Labels.shape)
    
Dataset = Dataset[:,:,0]

N = Dataset.shape[0]
T = Dataset.shape[1]


t_total = time.time() ##Start timing
#%%time
DTW_features = np.zeros((N,N))
print(DTW_features.shape)
for i in range(N):
    for j in range(i,N):
        dist = pyts.metrics.dtw(x=Dataset[i], y=Dataset[j],dist='square',
                        method='sakoechiba', options={'window_size': 0.05})
        DTW_features[i,j] = dist
        DTW_features[j,i] = dist
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

t_total = time.time() ##Start timing
DTW_fast_features = np.zeros((N,N))
print(DTW_fast_features.shape)
for i in range(N):
    for j in range(i,N):
        dist = pyts.metrics.dtw(x=Dataset[i], y=Dataset[j],dist='square',
                        method='fast', options={'radius': 0})
        DTW_fast_features[i,j] = dist
        DTW_fast_features[j,i] = dist
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))