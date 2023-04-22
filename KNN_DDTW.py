from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
import sklearn
import numpy as np
import os
import sys
import time

# define a list of datasets
#datasets = ["BEARING_Univar"]


datasets = ["PHM2022_Multivar",
"PHM2022_Univar_PIN",
"PHM2022_Univar_PO",
"PHM2022_Univar_PDIN",
"ETCHING_Multivar",
"MFPT_48KHZ_Univar",
"MFPT_96KHZ_Univar",
"PADERBORN_64KHZ_Univar",
"PADERBORN_4KHZ_Univar",
"PADERBORN_64KHZ_Multivar",
"PADERBORN_4KHZ_Multivar",
"Hydraulic_systems_10HZ_Multivar",
"Hydraulic_systems_100HZ_Multivar",
"Gas_sensors_home_activity",
"Control_charts",
"CWRU_12k_DE_univar",
"CWRU_12k_DE_multivar",
"CWRU_12k_FE_univar",
"CWRU_12k_FE_multivar",
"CWRU_48k_DE_univar",
"CWRU_48k_DE_multivar"
]

#datasets = ["Control_charts","PHM2022_Multivar", "PHM2022_Univar_PDIN"]
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

from classifiers import TSKNN_module

# define the number of folds
n_folds = 5

# perform cross-validation for each dataset and algorithm combination
for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy")
    start = time.time() ##Start timing
    print(f"Starting to work on {Dataset_name} at {start}")
    

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



    #Run The DDTW Module
    TSKNN_module.KNN(results_path, Dataset_name, Dataset, Labels, dis= 'ddtw')



    

