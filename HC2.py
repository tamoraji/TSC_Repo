from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
import sklearn
import numpy as np
import os
import sys
import time
from datetime import datetime


# define a list of datasets
datasets = [
# "Control_charts",
# "ETCHING_Multivar",
# "Hydraulic_systems_10HZ_Multivar", #Run out of time
# "CWRU_12k_DE_univar", #Starting to work on CWRU_12k_DE_univar_Dataset at 2023-06-06 10:01:14 
# "CWRU_12k_DE_multivar", #Starting to work on CWRU_12k_DE_multivar_Dataset at 2023-06-08 10:42:43
# "CWRU_12k_FE_univar",#Starting to work on CWRU_12k_FE_univar_Dataset at 2023-06-10 11:37:03 no results after 48 hours
# "CWRU_12k_FE_multivar",#Starting to work on CWRU_12k_FE_multivar_Dataset at 2023-06-12 15:56:06
# "Hydraulic_systems_100HZ_Multivar",#Starting to work on Hydraulic_systems_100HZ_Multivar_Dataset at 2023-06-16 19:15:25
# "Gas_sensors_home_activity",#Starting to work on Gas_sensors_home_activity_Dataset at 2023-06-18 10:34:52
# "MFPT_48KHZ_Univar",#Starting to work on MFPT_48KHZ_Univar_Dataset at 2023-06-20 10:55:08
# "MFPT_96KHZ_Univar",
#"CWRU_48k_DE_univar",
#"CWRU_48k_DE_multivar",
# "PHM2022_Univar_PIN",#Starting to work on PHM2022_Univar_PIN_Dataset at 2023-06-22 14:45:23 -> Memory
# "PHM2022_Univar_PO", #Starting to work on PHM2022_Univar_PO_Dataset at 2023-06-24 11:40:09 -> memory
# "PHM2022_Univar_PDIN",
# "BEARING_Univar",#Starting to work on BEARING_Univar_Dataset at 2023-06-26 10:15:17 -> time
"PADERBORN_4KHZ_Univar",
# "PHM2022_Multivar",
#"PADERBORN_64KHZ_Univar",
# "PADERBORN_64KHZ_Multivar",
# "PADERBORN_4KHZ_Multivar",
]

datasets_path = "../datasets"
print(f"We are going to work on {len(datasets)} datasets!")


for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy", mmap_mode='r')
    print(Dataset.shape)
    

    Labels_name = dataset + "_Labels"
    Labels = np.load(datasets_path + "/"  + Labels_name + ".npy", mmap_mode='r')

    
# change this directory for your machine
root_dir = './'


# define a list of algorithms
algorirhms_path = "./classifiers"

from classifiers import HC2_module


# define the number of folds
n_folds = 5

# perform cross-validation for each dataset and algorithm combination
for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy", mmap_mode='r')
    start = time.time() ##Start timing
    start_formated = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting to work on {Dataset_name} at {start_formated}")
    print(f"The shape of the dataset is:{Dataset.shape}")
    

    Labels_name = dataset + "_Labels"
    Labels = np.load(datasets_path + "/"  + Labels_name + ".npy", mmap_mode='r')

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



    #Run The HC2 Module
    HC2_module.HC2(results_path, Dataset_name, Dataset, Labels, nb_folds=n_folds,
                n_jobs=20)
    
    print(f"Working on {Dataset_name} finished successfully!")


