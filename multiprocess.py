from multiprocessing import Pool
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime
import sklearn
import os
import sys

from classifiers import PF_module

    t_total = time.time() ##Start timing


    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    if dataset.shape[2] > 1:
        print("PF is not capable of doing classification on MTS. so it will be done on only the first dimension")
    
    dataset_1 = dataset[:,:,0]
    labels = labels.reshape((-1,))

    ## Create Classification module
    from sktime.classification.distance_based import ProximityForest
    classifier = ProximityForest(n_estimators = n_estimators, n_stump_evaluations= n_stump_evaluations , n_jobs = 64)


    kf = KFold(n_splits=nb_folds, shuffle=True)
    accuracy_scores = []
    f1_scores = []
    confusion_matrices = []
    report_list = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_1)):
        # split the data into training and testing sets
        X_train, X_test = dataset_1[train_idx], dataset_1[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
            
        # fit the algorithm on the training data
            
        classifier.fit(X_train, y_train)
            
        # make predictions on the testing data
        y_pred = classifier.predict(X_test)
            
        # calculate the evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f1)

        confusion = confusion_matrix(y_test, y_pred)
        #print(confusion)

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        confusion_matrices.append(confusion)

        report = classification_report(y_test, y_pred, zero_division=1)
        report_list.append(report)
        print(report)
        
        print(f" fold {fold+1} is Finished!")
        
        # save the output to a text file
        with open(f'{results_path}/dataset_{dataset_name}_RF_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
        
    with open(f'{results_path}/dataset_{dataset_name}_RF.txt', 'w') as f:
        f.write("Mean accuracy: {:.3f} (std={:.3f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.3f} (std={:.3f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))

    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

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

    pool = Pool(processes=64) # Create a pool of 4 worker processes
    results = pool.map(PF_module.PF(results_path, Dataset_name, dataset, Labels, nb_folds=5, n_estimators= 100, n_stump_evaluations=5), dataset) # Execute the function in parallel on the data
