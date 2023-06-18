from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime
import pandas as pd

## Implement the Rocket classifier
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.linear_model import LogisticRegression

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


def FBL(results_path, dataset_name, dataset, labels, nb_folds=5):

    t_total = time.time() ##Start timing
    
    print(f"\n The dataset shape is:{dataset.shape}")
    N=dataset.shape[0]
    print(f"\n The number of data samples (N) is:{N}")

    T=dataset.shape[1]
    print(f"\n The number of TS length (T) is:{T}")

    M=dataset.shape[2]
    print(f"\n The number of TS dimention (M) is:{M}")

    # Create needed 2D DataFrame as the input of tsfresh feature extraction module
    df = pd.DataFrame(dataset.reshape(N*T, M), columns=[f'feature_{i}' for i in range(M)])
    df['id'] = np.repeat(np.arange(N), T)
    df['time'] = np.tile(np.arange(T), N)
    print(f'The new shape of the dataset is : {df.shape}')
    
    # Compute features using the tsfresh extract_features function
    X_features = extract_features(df, column_id='id', column_sort='time',
                     default_fc_parameters=EfficientFCParameters(),
                     # we impute = remove all NaN features automatically
                     impute_function=impute,
                     n_jobs = 10)
    del df
    print(f'The shape of the extracted features is : {X_features.shape}')


    Labels = labels.squeeze()


    ## Create Classification module
    classifier = LogisticRegression()


    kf = KFold(n_splits=nb_folds, shuffle=True)
    accuracy_scores = []
    f1_scores = []
    confusion_matrices = []
    report_list = []
    temp = X_features.to_numpy()
    for fold, (train_idx, test_idx) in enumerate(kf.split(temp)):
        # split the data into training and testing sets
        print(train_idx.shape)
        print(test_idx.shape)
        X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
        y_train, y_test = Labels[train_idx], Labels[test_idx]
            
        # Initialize an empty set of selected features and a list of candidate features
        selected_features = set()
        candidate_features = list(X_features.columns)
        # Perform greedy forward feature selection
        simplefilter("ignore", category=ConvergenceWarning)

        best_score = 0
        while candidate_features:

            # Evaluate each candidate feature by adding it to the set of selected features
            # and training a logistic regression model on the resulting feature set
            candidate_scores = []
            for feature in candidate_features:
                feature_set = selected_features.union({feature})
                X_train_selected = X_train[list(feature_set)]
                classifier.fit(X_train_selected, y_train)
                score = classifier.score(X_train_selected, y_train)
                candidate_scores.append((feature, score))

            # Select the candidate feature that results in the highest classification rate
            current_feature, current_score = max(candidate_scores, key=lambda x: x[1])

            # Check if the improvement in the training set classification rate upon adding
            # the current best feature is less than 3%, and terminate the loop if it is
            if best_score is not None and (current_score - best_score) < 0.0003:
                break

            # Set the best feature and score to the current feature and score if they are
            # None, or if the current score is better than the best score
            if best_score is None or current_score > best_score:
                best_feature, best_score = current_feature, current_score

            # Add the best feature to the set of selected features
            selected_features.add(best_feature)

            # Remove the best feature from the list of candidate features
            candidate_features.remove(best_feature)
            print(len(candidate_features))


            # Print the selected features and their classification rate on the training data
            print(f"Selected features: {selected_features}")
            print(f"Classification rate: {best_score}")
        

            
        # make predictions on the testing data
        X_test_selected = X_test[list(feature_set)]
        y_pred = classifier.predict(X_test_selected)
            
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
        with open(f'{results_path}/dataset_{dataset_name}_FBL_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
            f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        
    with open(f'{results_path}/dataset_{dataset_name}_FBL.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.3f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.3f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    #clear memory
    del classifier
    del y_pred
    del Labels
    del X_train
    del X_test
    del y_train
    del y_test
    del X_features
    
    
    
    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

