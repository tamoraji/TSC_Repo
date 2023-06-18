from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime
import pyts



def BOSSVS(results_path, dataset_name, dataset, labels, nb_folds=5, 
           word_size=4, 
           n_bins=3, 
           window_size=20):

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    if dataset.shape[2] > 1:
        print("BOSSVS is not capable of doing classification on MTS. so it will be done on only the first dimension")

    #input shape = [n_instances, series_length]
    ##Remove the last axis
    Dataset = dataset[:,:,0]
    labels = labels.squeeze()

    ## Input "n" series with "d" dimensions of length "m" . default config  based on [4] is : 
    """
    Parameters
    ----------
    word_size : int (default = 4)
        Size of each word.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and 26.

    window_size : int or float (default = 10)
        Size of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_size * n_timestamps)``.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_step * n_timestamps)``.

    anova : bool (default = False)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    drop_sum : bool (default = False)
        If True, the first Fourier coefficient (i.e. the sum of the subseries)
        is dropped. Otherwise, it is kept.

    norm_mean : bool (default = False)
        If True, center each subseries before scaling.

    norm_std : bool (default = False)
        If True, scale each subseries to unit variance.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    use_idf : bool (default = True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : bool (default = False)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool (default = True)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    """



    ## Create Classification module
    from pyts.classification import BOSSVS
    classifier = BOSSVS(word_size=word_size, n_bins=n_bins, window_size=window_size)


    kf = KFold(n_splits=nb_folds, shuffle=True)
    accuracy_scores = []
    f1_scores = []
    confusion_matrices = []
    report_list = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(Dataset)):
        # split the data into training and testing sets
        X_train, X_test = Dataset[train_idx], Dataset[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        t_fold = time.time() ##Start timing
            
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
        
        print(f" fold {fold+1} of {dataset_name} is Finished!")
        
        # save the output to a text file
        with open(f'{results_path}/dataset_{dataset_name}_BOSSVS_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
            f.write("Total time elapsed: {:.4f}s".format(time.time() - t_fold))

        
    with open(f'{results_path}/dataset_{dataset_name}_BOSSVS.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.3f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.3f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
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

