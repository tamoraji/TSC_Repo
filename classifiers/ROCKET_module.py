from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime

## Implement the Rocket classifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.utils.multiclass import class_distribution
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.rocket import Rocket

class ROCKETClassifier(BaseClassifier):
    """Classifier wrapped for the ROCKET transformer using RidgeClassifierCV.

    Parameters
    ----------
    num_kernels             : int, number of kernels for ROCKET transform
    (default=10,000)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    classifier              : ROCKET classifier
    n_classes               : extracted from the data

    Notes
    -----
    @article{dempster_etal_2019,
      author  = {Dempster, Angus and Petitjean, Francois and Webb,
      Geoffrey I},
      title   = {ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels},
      year    = {2019},
      journal = {arXiv:1910.13051}
    }

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/ROCKETClassifier.java
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    def __init__(
        self,
        num_kernels=10000,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classifier = None

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super(ROCKETClassifier, self).__init__()

    def _fit(self, X, y):
        """Build a pipeline containing the ROCKET transformer and RidgeClassifierCV.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self.classifier = rocket_pipeline = make_pipeline(
            Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)), verbose=True
        )
        rocket_pipeline.fit(X, y)

        return self

    def _predict(self, X):
        """Find predictions for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        return self.classifier.predict(X)

    def _predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        dists = np.zeros((X.shape[0], self.n_classes))
        preds = self.classifier.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists

def ROCKET(results_path, dataset_name, dataset, labels, nb_folds=5, n_kernel=1000):

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")

    #input shape = [n_instances, n_dimensions, series_length]
    ##Swzp axis
    Dataset = np.swapaxes(dataset, 1,2)
    labels = labels.squeeze()



    ## Create Classification module
    classifier = ROCKETClassifier(num_kernels= n_kernel ,n_jobs=10)


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
        with open(f'{results_path}/dataset_{dataset_name}_ROCKET_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
        
    with open(f'{results_path}/dataset_{dataset_name}_ROCKET.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.3f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.3f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

