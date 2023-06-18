from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime
import pyts

# Suppress FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def LS(results_path, dataset_name, dataset, labels, nb_folds=5, 
       n_shapelets_per_size=0.1,
       min_shapelet_length=0.05,
       C=1000,
       verbose=1,
       n_jobs=10):
    
    t_total = time.time() ##Start timing


    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    if dataset.shape[2] > 1:
        print("LS is not capable of doing classification on MTS. so it will be done on only the first dimension")

    #input shape = [n_instances, series_length]
    ##Remove the last axis
    Dataset = dataset[:,:,0]
    labels = labels.squeeze()

    ## Input "n" series with "d" dimensions of length "m" . default config  based on [4] is : 
    """
    Parameters
    ----------
    n_shapelets_per_size : int or float (default = 0.2)
        Number of shapelets per size. If float, it represents
        a fraction of the number of timestamps and the number
        of shapelets per size is equal to
        ``ceil(n_shapelets_per_size * n_timestamps)``.

    min_shapelet_length : int or float (default = 0.1)
        Minimum length of the shapelets. If float, it represents
        a fraction of the number of timestamps and the minimum
        length of the shapelets per size is equal to
        ``ceil(min_shapelet_length * n_timestamps)``.

    shapelet_scale : int (default = 3)
        The different scales for the lengths of the shapelets.
        The lengths of the shapelets are equal to
        ``min_shapelet_length * np.arange(1, shapelet_scale + 1)``.
        The total number of shapelets (and features)
        is equal to ``n_shapelets_per_size * shapelet_scale``.

    penalty : 'l1' or 'l2' (default = 'l2')
        Used to specify the norm used in the penalization.

    tol : float (default = 1e-3)
        Tolerance for stopping criterion.

    C : float (default = 1000)
        Inverse of regularization strength. It must be a positive float.
        Smaller values specify stronger regularization.

    learning_rate : float (default = 1.)
        Learning rate for gradient descent optimization. It must be a positive
        float. Note that the learning rate will be automatically decreased
        if the loss function is not decreasing.

    max_iter : int (default = 1000)
        Maximum number of iterations for gradient descent algorithm.

    multi_class : {'multinomial', 'ovr', 'ovo'} (default = 'multinomial')
        Strategy for multiclass classification.
        'multinomial' stands for multinomial cross-entropy loss.
        'ovr' stands for one-vs-rest strategy.
        'ovo' stands for one-vs-one strategy.
        Ignored if the classification task is binary.

    alpha : float (default = -100)
        Scaling term in the softmin function. The lower, the more precised
        the soft minimum will be. Default value should be good for
        standardized time series.

    fit_intercept : bool (default = True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float (default = 1.)
        Scaling of the intercept. Only used if ``fit_intercept=True``.

    class_weight : dict, None or 'balanced' (default = None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have unit weight.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : int (default = 0)
        Controls the verbosity. It must be a non-negative integer.
        If positive, loss at each iteration is printed.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    n_jobs : None or int (default = None)
        The number of jobs to use for the computation. Only used if
        ``multi_class`` is 'ovr' or 'ovo'.
    """


    ## Create Classification module
    from pyts.classification import LearningShapelets
    classifier = LearningShapelets(n_shapelets_per_size=n_shapelets_per_size,
                                   min_shapelet_length=min_shapelet_length,
                                   C=C,
                                   verbose=verbose,
                                   n_jobs=n_jobs)

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
        with open(f'{results_path}/dataset_{dataset_name}_LS_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
            f.write("Total time elapsed: {:.4f}s".format(time.time() - t_fold))

        
    with open(f'{results_path}/dataset_{dataset_name}_LS.txt', 'w') as f:
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

