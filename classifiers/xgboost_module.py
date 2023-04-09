from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import time
import numpy as np
import sktime
import xgboost as xgb



def XGBOOST(results_path, dataset_name, dataset, labels, nb_folds=5,
            learning_rate =0.3,
            n_estimators=100,
            max_depth=6,
            min_child_weight=1,
            gamma=0,
            subsample=1,
            colsample_bytree=1,
            objective='multi:softmax',
            n_jobs = 10):

    t_total = time.time() ##Start timing

    print(f"\n The dataset shape is:{dataset.shape}")
    print(f"\n The number of data samples (N) is:{dataset.shape[0]}")
    print(f"\n The number of TS length (T) is:{dataset.shape[1]}")
    print(f"\n The number of TS dimention (M) is:{dataset.shape[2]}")
    if dataset.shape[2] > 1:
        print("XGBoost is not capable of doing classification on MTS. so it will be done on only the first dimension")

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
    n_estimators : Optional[int][default=100]
        Number of gradient boosted trees.  Equivalent to number of boosting
        rounds.
    max_depth :  Optional[int] [default=6]
        Maximum tree depth for base learners.
    max_leaves :
        Maximum number of leaves; 0 indicates no limit.
    max_bin :
        If using histogram-based algorithm, maximum number of bins per feature
    grow_policy :
        Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
        depth-wise. 1: favor splitting at nodes with highest loss change.
    learning_rate : Optional[float][default=0.3]
        Boosting learning rate (xgb's "eta")
    verbosity : Optional[int]
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    objective : {SklObjective}
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
        https://xgboost.readthedocs.io/en/stable/parameter.html
    booster: Optional[str]
        Specify which booster to use: gbtree, gblinear or dart.
    tree_method: Optional[str]
        Specify which tree method to use.  Default to auto.  If this parameter is set to
        default, XGBoost will choose the most conservative option available.  It's
        recommended to study this option from the parameters document :doc:`tree method
        </treemethod>`
    n_jobs : Optional[int]
        Number of parallel threads used to run xgboost.  When used with other
        Scikit-Learn algorithms like grid search, you may choose which algorithm to
        parallelize and balance the threads.  Creating thread contention will
        significantly slow down both algorithms.
    gamma : Optional[float][default=0]
        (min_split_loss) Minimum loss reduction required to make a further partition on a
        leaf node of the tree.
    min_child_weight : Optional[float][default=1]
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : Optional[float]
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : Optional[float][default=1]
        Subsample ratio of the training instance.
    sampling_method :
        Sampling method. Used only by `gpu_hist` tree method.
          - `uniform`: select random training instances uniformly.
          - `gradient_based` select random training instances with higher probability when
            the gradient and hessian are larger. (cf. CatBoost)
    colsample_bytree : Optional[float][default=1]
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : Optional[float]
        Subsample ratio of columns for each level.
    colsample_bynode : Optional[float]
        Subsample ratio of columns for each split.
    reg_alpha : Optional[float]
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : Optional[float]
        L2 regularization term on weights (xgb's lambda).
    scale_pos_weight : Optional[float][default=1]
        Balancing of positive and negative weights.
    base_score : Optional[float]
        The initial prediction score of all instances, global bias.
    random_state : Optional[Union[numpy.random.RandomState, int]]
        Random number seed.
        .. note::
           Using gblinear booster with shotgun updater is nondeterministic as
           it uses Hogwild algorithm.
    missing : float, default np.nan
        Value in the data which needs to be present as a missing value.
    num_parallel_tree: Optional[int]
        Used for boosting random forest.
    monotone_constraints : Optional[Union[Dict[str, int], str]]
        Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
        for more information.
    interaction_constraints : Optional[Union[str, List[Tuple[str]]]]
        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
        3, 4]]``, where each inner list is a group of indices of features that are
        allowed to interact with each other.  See :doc:`tutorial
        </tutorials/feature_interaction_constraint>` for more information
    importance_type: Optional[str]
        The feature importance type for the feature_importances\\_ property:
        * For tree model, it's either "gain", "weight", "cover", "total_gain" or
          "total_cover".
        * For linear model, only "weight" is defined and it's the normalized coefficients
          without bias.
    gpu_id : Optional[int]
        Device ordinal.
    validate_parameters : Optional[bool]
        Give warnings for unknown parameter.
    predictor : Optional[str]
        Force XGBoost to use specific predictor, available choices are [cpu_predictor,
        gpu_predictor].
    enable_categorical : bool
        .. versionadded:: 1.5.0
        .. note:: This parameter is experimental
        Experimental support for categorical data.  When enabled, cudf/pandas.DataFrame
        should be used to specify categorical data type.  Also, JSON/UBJSON
        serialization format is required.
    feature_types : Optional[FeatureTypes]
        .. versionadded:: 1.7.0
        Used for specifying feature types without constructing a dataframe. See
        :py:class:`DMatrix` for details.
    max_cat_to_onehot : Optional[int]
        .. versionadded:: 1.6.0
        .. note:: This parameter is experimental
        A threshold for deciding whether XGBoost should use one-hot encoding based split
        for categorical data.  When number of categories is lesser than the threshold
        then one-hot encoding is chosen, otherwise the categories will be partitioned
        into children nodes. Also, `enable_categorical` needs to be set to have
        categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.
    max_cat_threshold : Optional[int]
        .. versionadded:: 1.7.0
        .. note:: This parameter is experimental
        Maximum number of categories considered for each split. Used only by
        partition-based splits for preventing over-fitting. Also, `enable_categorical`
        needs to be set to have categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.
    multi_strategy : Optional[str]
        .. versionadded:: 2.0.0
        .. note:: This parameter is working-in-progress.
        The strategy used for training multi-target models, including multi-target
        regression and multi-class classification. See :doc:`/tutorials/multioutput` for
        more information.
        - ``one_output_per_tree``: One model for each target.
        - ``multi_output_tree``:  Use multi-target trees.
    eval_metric : Optional[Union[str, List[str], Callable]]
        .. versionadded:: 1.6.0
        Metric used for monitoring the training result and early stopping.  It can be a
        string or list of strings as names of predefined metric in XGBoost (See
        doc/parameter.rst), one of the metrics in :py:mod:`sklearn.metrics`, or any other
        user defined metric that looks like `sklearn.metrics`.
        If custom objective is also provided, then custom metric should implement the
        corresponding reverse link function.
        Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
        object is provided, it's assumed to be a cost function and by default XGBoost will
        minimize the result during early stopping.
        For advanced usage on Early stopping like directly choosing to maximize instead of
        minimize, see :py:obj:`xgboost.callback.EarlyStopping`.
        See :doc:`Custom Objective and Evaluation Metric </tutorials/custom_metric_obj>`
        for more.
        .. note::
             This parameter replaces `eval_metric` in :py:meth:`fit` method.  The old
             one receives un-transformed prediction regardless of whether custom
             objective is being used.
        .. code-block:: python
            from sklearn.datasets import load_diabetes
            from sklearn.metrics import mean_absolute_error
            X, y = load_diabetes(return_X_y=True)
            reg = xgb.XGBRegressor(
                tree_method="hist",
                eval_metric=mean_absolute_error,
            )
            reg.fit(X, y, eval_set=[(X, y)])
    early_stopping_rounds : Optional[int]
        .. versionadded:: 1.6.0
        - Activates early stopping. Validation metric needs to improve at least once in
          every **early_stopping_rounds** round(s) to continue training.  Requires at
          least one item in **eval_set** in :py:meth:`fit`.
        - The method returns the model from the last iteration, not the best one, use a
          callback :py:class:`xgboost.callback.EarlyStopping` if returning the best
          model is preferred.
        - If there's more than one item in **eval_set**, the last entry will be used for
          early stopping.  If there's more than one metric in **eval_metric**, the last
          metric will be used for early stopping.
        - If early stopping occurs, the model will have three additional fields:
          :py:attr:`best_score`, :py:attr:`best_iteration`.
        .. note::
            This parameter replaces `early_stopping_rounds` in :py:meth:`fit` method.
    callbacks : Optional[List[TrainingCallback]]
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.
        .. note::
           States in callback are not preserved during training, which means callback
           objects can not be reused for multiple training sessions without
           reinitialization or deepcopy.
        .. code-block:: python
            for params in parameters_grid:
                # be sure to (re)initialize the callbacks before each run
                callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
                reg = xgboost.XGBRegressor(**params, callbacks=callbacks)
                reg.fit(X, y)
    kwargs : dict, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of parameters
        can be found :doc:`here </parameter>`.
        Attempting to set a parameter via the constructor args and \\*\\*kwargs
        dict simultaneously will result in a TypeError.
        .. note:: \\*\\*kwargs unsupported by scikit-learn
            \\*\\*kwargs is unsupported by scikit-learn.  We do not guarantee
            that parameters passed via this argument will interact properly
            with scikit-learn.
    """


    ## Create Classification module
    classifier = xgb.sklearn.XGBClassifier(learning_rate =learning_rate,
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_child_weight=min_child_weight,
                                        gamma=gamma,
                                        subsample=subsample,
                                        colsample_bytree=colsample_bytree,
                                        objective=objective,
                                        n_jobs = n_jobs)


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
        with open(f'{results_path}/dataset_{dataset_name}_XGBoost_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Confusion Matrix:\n{confusion}\n\n')
            f.write(f'Classification report:\n{report}\n\n')
        
    with open(f'{results_path}/dataset_{dataset_name}_XGBoost.txt', 'w') as f:
        f.write("Mean accuracy: {:.4f} (std={:.4f})\n".format(np.mean(accuracy_scores), np.std(accuracy_scores)))
        f.write("Mean F1 score: {:.4f} (std={:.4f})\n".format(np.mean(f1_scores), np.std(f1_scores)))
        f.write("Mean confusion matrix:\n{}\n".format(np.array2string(np.mean(confusion_matrices, axis=0))))
        f.write("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print(" Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

