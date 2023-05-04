from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime



def MRSQM(results_path, dataset_name, dataset, labels, nb_folds=5,
        strat='RS', features_per_rep=500, selection_per_rep=2000, nsax=1, nsfa=0):

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    if dataset.shape[2] > 1:
        print("MrSQM is not capable of doing classification on MTS. so it will be done on only the first dimension")

    #input shape = [n_instances, series_length]
    ##Remove the last axis
    Dataset = dataset[:,:,0]
    labels = labels.squeeze()

    #input shape = [n_instances, n_dimensions, series_length]
    ##Swzp axis
    #Dataset = np.swapaxes(dataset, 1,2)


    ## Input "n" series with "d" dimensions of length "m" . default config  based on [73] is : 
    """
    Parameters
    ----------
    strat               : str, one of 'R','S','SR', or 'RS', default="RS"
        feature selection strategy. By default set to 'RS'.
        R and S are single-stage filters while RS and SR are two-stage filters.
    features_per_rep    : int, default=500
        (maximum) number of features selected per representation.
    selection_per_rep   : int, default=2000
        (maximum) number of candidate features selected per representation.
        Only applied in two stages strategies (RS and SR), otherwise ignored.
    nsax                : int, default=1
        number of representations produced by sax transformation.
    nsfa                : int, default=0
        number of representations produced by sfa transformation.
        WARNING: setting this to 1 or larger will break estimator persistence (save),
        known bug, see https://github.com/mlgig/mrsqm/issues/7
    custom_config       : dict, default=None
        customized parameters for the symbolic transformation.
    random_state        : int, default=None.
        random seed for the classifier.
    sfa_norm            : bool, default=True.
        whether to apply time series normalisation (standardisation).

    """


    ## Create Classification module
    from sktime.classification.shapelet_based._mrsqm import MrSQM
    classifier = MrSQM(strat=strat, 
                        features_per_rep=features_per_rep,
                        selection_per_rep=selection_per_rep, 
                        nsax=nsax, 
                        nsfa=nsfa)


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
        
        print(f" fold {fold+1} for {dataset_name} is Finished!")
        
        # save the output to a text file
        with open(f'{results_path}/dataset_{dataset_name}_MrSQM_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
            f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        
    with open(f'{results_path}/dataset_{dataset_name}_MrSQM.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.4f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.4f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    #clear memory
    del classifier
    del y_pred
    del Dataset
    del labels
    del X_train
    del X_test
    del y_train
    del y_test
    
    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

