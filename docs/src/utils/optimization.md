Module src.utils.optimization
=============================

Functions
---------

    
`att_model(estimator, space, values, neural=False)`
:   

Builds an estimator from a space and values.
    
    Args:
        estimator: The estimator to be built.
        space: The space of hyperparameters.
        values: The values of the hyperparameters.
        neural: Whether the estimator is a neural network.
    
    Returns:
        The built estimator.
    
    Note:
        In the case of neural networks, the names of the parameters associated with 
        hidden layers must be `n_neurons_per_layer` and `n_hidden_layer`.
    
    Example:
        att_model(estimator=LinearRegression(), space=space, values=values)

    
`change_nn_params(params)`
:   

Changes the parameters of a neural network regressor.
    
    Args:
        params: A dictionary of parameters for the regressor.
    
    Returns:
        The modified dictionary of parameters.
    
    Example:
        params = change_nn_params({"n_neurons_per_layer": 10, "n_hidden_layer": 2})

    
`convert_to_space(caller, parameter)`
:   

Converts a hyperparameter from a YAML file to a Space object.
    
    Args:
        caller: The dictionary containing the hyperparameters.
        parameter: The key of the hyperparameter to be converted.
    
    Returns:
        A Space object representing the hyperparameter.
    
    Example:
        convert_to_space(cfg.opt.estimator, "learning_rate")

    
`gp_optimize(estim, x, y, space, cv, n_calls=150, n_random_starts=100, neural=False, scoring='neg_mean_absolute_percentage_error', n_jobs_cv=4, n_jobs_opt=4, verbose=0)`
:   

Optimizes an estimator using Gaussian Process.
    
    Args:
        estim: The estimator to be optimized.
        x: The training data.
        y: The target values.
        space: The space of hyperparameters.
        cv: The cross-validation folds.
        n_calls: The number of function calls to the optimizer.
        n_random_starts: The number of random restarts.
        neural: Whether the estimator is a neural network.
        scoring: The scoring metric.
        n_jobs_cv: The number of jobs for the cross-validation.
        n_jobs_opt: The number of jobs for the optimizer.
        verbose: The verbosity level.
    
    Returns:
        The results of the optimization.
    
    Example:
        gp_optimize(estim=LinearRegression(), x=x, y=y, space=space, cv=cv)

    
`modify_scaling(list_of_models, list_of_models_names, name_of_features)`
:   

Modifies the scaling of a list of models.
    
    Args:
        list_of_models: A list of models to be modified.
        list_of_models_names: A list of the names of the models.
        name_of_features: The name of the features to be scaled.
    
    Returns:
        A list of the modified models.
    
    Example:
        modify_scaling([model1, model2], ["model1", "model2"], "all")

    
`nested_cv(estimator, space, x, y, out_cv, inner_cv, scoring='neg_mean_absolute_percentage_error', neural=False, n_calls=150, n_random_starts=100, verbose=0, shuffle=True, random_state=0, print_mode=False)`
:   

Perform nested cross-validation.
    
    The function will return the scores of each fold for the outer k-fold, the train scores for each fold of the inner k-fold, the test predictions for each element when it was in the test set, the test errors for each element when it was in the test set, and the cv score.
    
    It is important to remember that the mean of the test error will not be equivalent to the cv score. This is because the grouping of test data is different from the simple sum and division by N of the mean of the test error.
    
    Args:
        estimator: The estimator to be optimized.
        space: The space of hyperparameters.
        x: The training data.
        y: The target values.
        out_cv: The number of folds for the outer k-fold.
        inner_cv: The number of folds for the inner k-fold.
        scoring: The scoring metric.
        neural: Whether the estimator is a neural network.
        n_calls: The number of function calls to the optimizer.
        n_random_starts: The number of random restarts.
        verbose: The verbosity level.
        shuffle: Whether to shuffle the data.
        random_state: The random seed.
        print_mode: Whether to print the scores.
    
    Returns:
        out_test_scores: The score of the outer folds in the test set.
        out_train_scores: The score of the outer folds in the train set.
        best_models: The best estimators founded in each run of outer fold's to be utilized in stacked cv.

    
`opt_all(estimator_list, names_estimator_list, spaces_list, x, y, cv, path, verbose=0, print_models=False)`
:   

Optimizes all the estimators in the list and saves them.
    
    Args:
        estimator_list: The list of estimators to be optimized.
        names_estimator_list: The list of names of the estimators.
        spaces_list: The list of spaces of the estimators.
        x: The training data.
        y: The target values.
        cv: The cross-validation folds.
        path: The path to the directory where the models will be saved.
        verbose: The verbosity level.
        print_models: Whether to print the models.
    
    Returns:
        None.
    
    Example:
        opt_all(estimator_list=estimators, names_estimator_list=names, spaces_list=spaces, x=x, y=y, cv=cv, path="models")

    
`save_models(path: str, model)`
:   

Saves a model to a file.
    
    Args:
        path: The path to the file where the model will be saved.
        model: The model to be saved.
    
    Returns:
        None.
    
    Example:
        save_models(path="models/model.pkl", model=my_model)

    
`stacked_nested_cv(estimators, estimators_names, spaces, x, y, out_cv, inner_cv, scoring='neg_mean_absolute_percentage_error', neural=False, n_calls=150, n_random_starts=100, verbose=0, shuffle=True, random_state=0)`
:   

Perform stacked nested cross-validation.
    
    The function will return the scores of each fold for the outer k-fold, the train scores for each fold of the inner k-fold, the test predictions for each element when it was in the test set, the test errors for each element when it was in the test set, and the cv score for each model.
    
    It is important to remember that the mean of the test error will not be equivalent to the cv score. This is because the grouping of test data is different from the simple sum and division by N of the mean of the test error.
    
    Args:
        estimators: A list of estimators to be optimized.
        estimators_names: A list of names of the estimators.
        spaces: A list of spaces of the estimators.
        x: The training data.
        y: The target values.
        out_cv: The number of folds for the outer k-fold.
        inner_cv: The number of folds for the inner k-fold.
        scoring: The scoring metric.
        neural: Whether the estimators are neural networks.
        n_calls: The number of function calls to the optimizer.
        n_random_starts: The number of random restarts.
        verbose: The verbosity level.
        shuffle: Whether to shuffle the data.
        random_state: The random seed.
    
    Returns:
        dict_test: A dictionary with the test scores for each model.
        dict_train: A dictionary with the train scores for each model.