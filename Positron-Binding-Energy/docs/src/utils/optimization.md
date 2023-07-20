Module src.utils.optimization
=============================

Functions
---------

    
`att_model(estimator, space, values, neural=False)`
:   It is important to remember that in the case of neural networks,
    the name of the
    parameters associated with hidden_layers must be
    "n_neurons_per_layer"
    and "n_hidden_layer"

    
`change_nn_params(params)`
:   

    
`convert_to_space(caller, parameter)`
:   caller: Should have an form of cfg.opt.estimator
    parameter: Key of the hyperparameter in the way it's write
    in .yaml file.

    
`gp_optimize(estim, x, y, space, cv, n_calls=150, n_random_starts=100, neural=False, scoring='neg_mean_absolute_percentage_error', n_jobs_cv=4, n_jobs_opt=4, verbose=0)`
:   

    
`modify_scaling(list_of_models, list_of_models_names, name_of_features)`
:   

    
`nested_cv(estimator, space, x, y, out_cv, inner_cv, scoring='neg_mean_absolute_percentage_error', neural=False, n_calls=150, n_random_starts=100, verbose=0, shuffle=True, random_state=0, print_mode=False)`
:   A função retornará os scores de cada fold pra k_fold externo
    train_scores: Score de treino pra cada fold de treino.
    test_pred: Predição de teste pra cada elemento quando esteve no conjunto de teste.
    test_error: Erro de teste pra cada elemento quando esteve no conjunto de teste.
    cv_score: Erro de teste de cada k_fold externo.
    
    Importante lembrar que a média do test error não será equivalente ao cv score.
    Isto pois o agrupamento de dados de teste é diferente do que a simples soma e divisão por N
    da média do test error.
    # Return
    - out_test_scores: The score of the outer folds in the test set
    - out_train_scores: The score of the outer folds in the train set
    - best_models: The best estimators founded in each run of outer fold's to be utilized in stacked cv

    
`opt_all(estimator_list, names_estimator_list, spaces_list, x, y, cv, path, verbose=0, print_models=False)`
:   

    
`save_models(path: str, model)`
:   

    
`stacked_nested_cv(estimators, estimators_names, spaces, x, y, out_cv, inner_cv, scoring='neg_mean_absolute_percentage_error', neural=False, n_calls=150, n_random_starts=100, verbose=0, shuffle=True, random_state=0)`
: