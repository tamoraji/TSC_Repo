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
# "Hydraulic_systems_10HZ_Multivar",
# "Hydraulic_systems_100HZ_Multivar",
# "Gas_sensors_home_activity",
# "CWRU_12k_DE_univar",
# "CWRU_12k_DE_multivar",
# "CWRU_12k_FE_univar",
# "CWRU_12k_FE_multivar",
# "CWRU_48k_DE_univar",#Starting to work on CWRU_48k_DE_univar_Dataset at 2023-06-12 15:44:14 n_jobs=20 from here
# "CWRU_48k_DE_multivar", #Starting to work on CWRU_48k_DE_multivar_Dataset at 2023-06-21 00:05:39
# "MFPT_48KHZ_Univar",#Starting to work on MFPT_48KHZ_Univar_Dataset at 2023-06-27 10:43:10
"MFPT_96KHZ_Univar",#Start from here
"PHM2022_Multivar",
"PHM2022_Univar_PIN",
"PHM2022_Univar_PO",
"PHM2022_Univar_PDIN",
"BEARING_Univar",
#"PADERBORN_64KHZ_Univar",
# "PADERBORN_4KHZ_Univar",
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

from classifiers import STC_module


# define the number of folds
n_folds = 5

# perform cross-validation for each dataset and algorithm combination
for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy", mmap_mode='r')
    start = time.time() ##Start timing
    start_formated = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S') #change from epoch format to date format
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



    #Run The STC Module
    STC_module.STC(results_path, Dataset_name, Dataset, Labels, nb_folds=n_folds,
                n_shapelet_samples= 10000, max_shapelets=1000, batch_size=100,
                n_jobs=20)
    
    print(f"Working on {Dataset_name} finished successfully!")


print("STC algorithm finished!")

