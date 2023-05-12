from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime



def DrCIF(results_path, dataset_name, dataset, labels, nb_folds=5, n_estimators=200, att_subsample_size= 10):

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")

    #input shape = [n_instances, n_dimensions, series_length]
    ##Swzp axis
    Dataset = np.swapaxes(dataset, 1,2)
    labels = labels.squeeze()

    ## Input "n" series with "d" dimensions of length "m" . default config  based on [4] is : 
    ## Default (trees: n_estimators = 500, intervals: n_intervals = sqrt(M) × sqrt(T), att_subsample_size = 8 attributes per tree)



    ## Create Classification module
    from sktime.classification.interval_based import DrCIF
    classifier = DrCIF(n_estimators= n_estimators,
                                     att_subsample_size= att_subsample_size, n_jobs=10)


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
        
        print(f" fold {fold+1} of {dataset_name} is Finished!")
        
        # save the output to a text file
        with open(f'{results_path}/dataset_{dataset_name}_DrCIF_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
            f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        
    with open(f'{results_path}/dataset_{dataset_name}_DrCIF.txt', 'w') as f:
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
