from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime
import pyts

# Suppress FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def ARSENAL(results_path, dataset_name, dataset, labels, nb_folds=5, 
       num_kernels= 2000,
       n_estimators= 25,
       rocket_transform= "rocket",
       n_jobs=20):
    
    t_total = time.time() ##Start timing


    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")


    #input shape = [n_instances, series_length]
    ##Remove the last axis
    Dataset = np.swapaxes(dataset, 1,2)
    labels = labels.squeeze()

    ## Input "n" series with "d" dimensions of length "m" . default config  based on [4] is : 
    """
    Arsenal ensemble.

    Overview: an ensemble of ROCKET transformers using RidgeClassifierCV base
    classifier. Weights each classifier using the accuracy from the ridge
    cross-validation. Allows for generation of probability estimates at the
    expense of scalability compared to RocketClassifier.

    Parameters
    ----------
    num_kernels : int, default=2,000
        Number of kernels for each ROCKET transform.
    n_estimators : int, default=25
        Number of estimators to build for the ensemble.
    rocket_transform : str, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket","minirocket","multirocket"]
    max_dilations_per_kernel : int, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=4
        MultiRocket only. The number of features per kernel.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    contract_max_n_estimators : int, default=100
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.
    """


    ## Create Classification module
    from sktime.classification.kernel_based import Arsenal
    classifier = Arsenal(num_kernels=num_kernels,
                         n_estimators=n_estimators,
                         rocket_transform=rocket_transform,
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
        with open(f'{results_path}/dataset_{dataset_name}_ARSENAL_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
            f.write("Total time elapsed: {:.4f}s".format(time.time() - t_fold))

        
    with open(f'{results_path}/dataset_{dataset_name}_ARSENAL.txt', 'w') as f:
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

