from multiprocessing import Pool
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sklearn
import os
import sys

from classifiers import SVM_module

# define a list of datasets
datasets = ["PHM2022_Multivar", "PHM2022_Univar_PDIN"]
datasets_path = "../datasets"

# change this directory for your machine
root_dir = './'

#add the classifier path to the sys
#sys.path.append("./classifiers/")

algorirhms_path = "./classifiers"


# define the number of folds
n_folds = 5

# perform cross-validation for each dataset and algorithm combination
for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy")
    print(Dataset.shape)

    Labels_name = dataset + "_Labels"
    Labels = np.load(datasets_path + "/"  + Labels_name + ".npy")

    # Create a folder for results
    results_path = root_dir + "Results/" + Dataset_name
    if os.path.exists(results_path):
        pass
    else:
        try:
            os.makedirs(results_path)
        except:
            # in case another machine created the path meanwhile !:(
            pass

    pool = Pool(processes=20) # Create a pool of 4 worker processes
    results = pool.map(SVM_module.SVM(results_path, Dataset_name, Dataset, Labels, C=10), dataset) # Execute the function in parallel on the data
