# What version of Python do you have?
import sys
from datetime import datetime


import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import tensorflow.keras as keras


print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
import sklearn.preprocessing

import time
import sklearn
import numpy as np
import os
import sys
from joblib import parallel_backend

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
# "CWRU_48k_DE_univar",
# "CWRU_48k_DE_multivar",
# "PHM2022_Multivar", #Starting to work on PHM2022_Multivar_Dataset at 2023-06-14 09:18:21  
# "PHM2022_Univar_PIN",
# "PHM2022_Univar_PO",
# "PHM2022_Univar_PDIN",
"MFPT_48KHZ_Univar",
"MFPT_96KHZ_Univar",
"BEARING_Univar",
# "PADERBORN_4KHZ_Univar",
# "PADERBORN_4KHZ_Multivar",
# "PADERBORN_64KHZ_Multivar",
# "PADERBORN_64KHZ_Univar",
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

# define the number of folds
n_folds = 5

# perform cross-validation for each dataset and algorithm combination
for dataset in datasets:
    Dataset_name = dataset + "_Dataset"
    Dataset = np.load(datasets_path + "/" + Dataset_name + ".npy", mmap_mode='r')
    start = time.time() ##Start timing
    start_formated = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting to work on {Dataset_name} at {start_formated}")
    

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

        
    t_total = time.time() ##Start timing
    


    print(f"\n The dataset shape is:{Dataset.shape}")
    print(f"\n The number of data samples (N) is:{Dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{Dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{Dataset.shape[2]}")

    nb_classes = len(np.unique(Labels, axis=0))
    
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    Labels = enc.fit_transform(Labels.reshape(-1, 1)).toarray()


    kf = KFold(n_splits=n_folds, shuffle=True)
    accuracy_scores = []
    f1_scores = []
    confusion_matrices = []
    report_list = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(Dataset)):
        # split the data into training and testing sets
        X_train, X_test = Dataset[train_idx], Dataset[test_idx]
        y_train, y_test = Labels[train_idx], Labels[test_idx]
        
        t_fold = time.time() ##Start timing

        
        # save orignal y because later we will use binary
        y_true = np.argmax(y_test, axis=1)
        
        input_shape = X_train.shape[1:]
        
        ## Create Classification module
        from dl4tsc.classifiers import TSLSTM
        classifier = TSLSTM.Classifier_TSLSTM(output_directory=results_path, input_shape=input_shape,
                                        nb_classes=nb_classes, verbose=True,
                                        lr=0.001, batch_size=64, epoch = 500)
        
        # fit the algorithm on the training data
        accuracy, f1, confusion, report = classifier.fit(X_train, y_train, X_test, y_test, y_true)
            
        # calculate the evaluation metrics      
        accuracy_scores.append(accuracy)
        print(accuracy)

        f1_scores.append(f1)
        print(f1)

        confusion_matrices.append(confusion)
        print(confusion)

        report_list.append(report)
        print(report)
        
        print(f" fold {fold+1} of {Dataset_name} is Finished!")
        keras.backend.clear_session()
        
        import gc
        # Call the garbage collector
        gc.collect()
        
        # save the output to a text file
        with open(f'{results_path}/dataset_{Dataset_name}_TSLSTM_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
            f.write("Total time elapsed: {:.4f}s".format(time.time() - t_fold))

        
    with open(f'{results_path}/dataset_{Dataset_name}_TSLSTM_fold_.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.3f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.3f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    
