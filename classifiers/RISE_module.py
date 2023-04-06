from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime



<<<<<<< HEAD:classifiers/RISE_module.py
def RISE(results_path, dataset_name, dataset, labels, nb_folds=5,
        n_estimators = 200,
        n_jobs = 10):
=======
def CBOSS(results_path, dataset_name, dataset, labels, nb_folds=5, n_parameter_samples= 250, 
          max_ensemble_size=50, max_win_len_prop = 0.7,feature_selection="chi2", n_jobs = 10):
>>>>>>> ffded664129b3d440f7d3eae21b7403b1bc7b30c:classifiers/CBOSS_module.py

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    if dataset.shape[2] > 1:
        print("RISE is not capable of doing classification on MTS. so it will be done on only the first dimension")

    #input shape = [n_instances, series_length]
    ##Remove the last axis
    Dataset = dataset[:,:,0]

    #input shape = [n_instances, n_dimensions, series_length]
    ##Swzp axis
    #Dataset = np.swapaxes(dataset, 1,2)


    ## Input "n" series with "d" dimensions of length "m" . default config  based on [73] is : 
    """
    Parameters
    ----------
    n_estimators : int, default=200
        The number of trees in the forest.
    min_interval : int, default=16
        The minimum width of an interval.
    acf_lag : int, default=100
        The maximum number of autocorrelation terms to use.
    acf_min_values : int, default=4
        Never use fewer than this number of terms to find a correlation.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """


    ## Create Classification module
<<<<<<< HEAD:classifiers/RISE_module.py
    from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
    classifier = RandomIntervalSpectralEnsemble(n_estimators=n_estimators, n_jobs= n_jobs)
=======
    from sktime.classification.dictionary_based import ContractableBOSS
    classifier = ContractableBOSS(n_parameter_samples= n_parameter_samples , max_ensemble_size = max_ensemble_size , 
                                  max_win_len_prop = max_win_len_prop ,feature_selection=feature_selection, n_jobs= n_jobs)
>>>>>>> ffded664129b3d440f7d3eae21b7403b1bc7b30c:classifiers/CBOSS_module.py


    kf = KFold(n_splits=nb_folds, shuffle=True)
    accuracy_scores = []
    f1_scores = []
    confusion_matrices = []
    report_list = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(Dataset)):
        # split the data into training and testing sets
        X_train, X_test = Dataset[train_idx], Dataset[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
            
        # fit the algorithm on the training data
            
        classifier.fit(X_train, y_train)
        print("\n The classifier is fitted")
            
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
        with open(f'{results_path}/dataset_{dataset_name}_RISE_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
        
    with open(f'{results_path}/dataset_{dataset_name}_RISE.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.4f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.4f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

