from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime



def EE(results_path, dataset_name, dataset, labels, nb_folds=5,
        proportion_of_param_options=1,
        proportion_train_for_test=1,
        distance_measures = "all",
        majority_vote=True,
        n_jobs = 10):

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    if dataset.shape[2] > 1:
        print("EE is not capable of doing classification on MTS. so it will be done on only the first dimension")


    #input shape = [n_instances, series_length]
    ##Remove the last axis
    Dataset = dataset[:,:,0]
    labels = labels.squeeze()

    #input shape = [n_instances, n_dimensions, series_length]
    ##Swzp axis
    #Dataset = np.swapaxes(dataset, 1,2)


    ## Input "n" series with "d" dimensions of length "m" . default config  based on [73] is : 
    """
    Overview:
    - Input n series length m
    - EE is an ensemble of elastic nearest neighbor classifiers
    Parameters
    ----------
    distance_measures : list of strings, optional (default="all")
      A list of strings identifying which distance measures to include. Valid values
      are one or more of: euclidean, dtw, wdtw, ddtw, dwdtw, lcss, erp, msm
    proportion_of_param_options : float, optional (default=1)
      The proportion of the parameter grid space to search optional.
    proportion_train_in_param_finding : float, optional (default=1)
      The proportion of the train set to use in the parameter search optional.
    proportion_train_for_test : float, optional (default=1)
      The proportion of the train set to use in classifying new cases optional.
    n_jobs : int, optional (default=1)
      The number of jobs to run in parallel for both `fit` and `predict`.
      ``-1`` means using all processors.
    random_state : int, default=0
      The random seed.
    verbose : int, default=0
      If ``>0``, then prints out debug information.

    """


    ## Create Classification module
    from sktime.classification.distance_based import ElasticEnsemble
    classifier = ElasticEnsemble(proportion_of_param_options=proportion_of_param_options,
                                proportion_train_for_test=proportion_train_for_test,
                                distance_measures = distance_measures,
                                majority_vote=majority_vote,
                                n_jobs= n_jobs)


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
        with open(f'{results_path}/dataset_{dataset_name}_EE_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
        
    with open(f'{results_path}/dataset_{dataset_name}_EE.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.4f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.4f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

