from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
import sklearn
import numpy as np
import os
import sys



# define a list of datasets
datasets = ["Control_charts", "PHM2022_Multivar", "PHM2022_Univar_PDIN"]
datasets_path = "../datasets"

for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy")
    print(Dataset.shape)
    

    Labels_name = dataset + "_Labels"
    Labels = np.load(datasets_path + "/"  + Labels_name + ".npy")


# change this directory for your machine
root_dir = './'


# define a list of algorithms
algorirhms_path = "./classifiers"

from classifiers import xgboost_module


# define the number of folds
n_folds = 5

# perform cross-validation for each dataset and algorithm combination
for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy")
    

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



    #Run The XGboost Module
    xgboost_module.XGBOOST(results_path, Dataset_name, Dataset, Labels, nb_folds=n_folds,
                        learning_rate =0.3,
                        n_estimators=100,
                        max_depth=6,
                        min_child_weight=1,
                        gamma=0,
                        subsample=1,
                        colsample_bytree=1,
                        objective='multi:softmax',
                        scale_pos_weight=1,
                        n_jobs = 10)


    
