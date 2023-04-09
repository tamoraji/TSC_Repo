from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime



def STC(results_path, dataset_name, dataset, labels, nb_folds=5,
        n_shapelet_samples= 10000, max_shapelets=1000, batch_size=100,
        n_jobs = 10):

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    #if dataset.shape[2] > 1:
        #print("CBOSS is not capable of doing classification on MTS. so it will be done on only the first dimension")

    #input shape = [n_instances, series_length]
    ##Remove the last axis
    #Dataset = dataset[:,:,0]

    #input shape = [n_instances, n_dimensions, series_length]
    ##Swzp axis
    Dataset = np.swapaxes(dataset, 1,2)


    ## Input "n" series with "d" dimensions of length "m" . default config  based on [73] is : 
    """
    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to ``<= max_shapelets``, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to ``n_classes_ / max_shapelets``. If `None`, uses the
        minimum between ``10 * n_instances_`` and `1000`.
    max_shapelet_length : int or None, default=None
        Lower bound on candidate shapelet lengths for the transform. If ``None``, no
        max length is used
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn `BaseEstimator`. If
        `None` a default `RotationForest` classifier is used.
    transform_limit_in_minutes : int, default=0
        Time contract to limit transform time in minutes for the shapelet transform,
        overriding `n_shapelet_samples`. A value of `0` means ``n_shapelet_samples``
        is used.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding ``n_shapelet_samples``
        and ``transform_limit_in_minutes``. The ``estimator`` will only be contracted if
        a ``time_limit_in_minutes parameter`` is present. Default of `0` means
        ``n_shapelet_samples`` or ``transform_limit_in_minutes`` is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when contracting the transform with
        ``transform_limit_in_minutes`` or ``time_limit_in_minutes``.
    save_transformed_data : bool, default=False
        Save the data transformed in fit in ``transformed_data_`` for use in
        ``_get_train_probs``.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets in the transform.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    """


    ## Create Classification module
    from sktime.classification.shapelet_based import ShapeletTransformClassifier
    classifier = ShapeletTransformClassifier(n_shapelet_samples= n_shapelet_samples , 
                                            max_shapelets = max_shapelets , 
                                            batch_size = batch_size ,
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
        with open(f'{results_path}/dataset_{dataset_name}_STC_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
        
    with open(f'{results_path}/dataset_{dataset_name}_STC.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.4f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.4f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

