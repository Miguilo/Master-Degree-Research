import copy
import pickle

import numpy as np
from IPython.display import clear_output
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skopt import gp_minimize
from skopt.space.space import Categorical, Integer, Real
from skopt.utils import use_named_args


def modify_scaling(list_of_models, list_of_models_names, name_of_features):
    """
    Modifies the scaling of a list of models.

    Args:
        list_of_models: A list of models to be modified.
        list_of_models_names: A list of the names of the models.
        name_of_features: The name of the features to be scaled.

    Returns:
        A list of the modified models.

    Example:
        modify_scaling([model1, model2], ["model1", "model2"], "all")
    """
    modified_models = []  # Lista para armazenar os modelos modificados

    for model, name in zip(list_of_models, list_of_models_names):
        if "xgb" in name:
            modified_models.append(model)
        else:
            ct = ColumnTransformer(
                [("scale", RobustScaler(), slice(0, None))],
                remainder="passthrough",
            )
            if (
                "pi" in name_of_features.lower()
                or "all" in name_of_features.lower()
            ):
                ct = ColumnTransformer(
                    [("scale", RobustScaler(), slice(0, -1))],
                    remainder="passthrough",
                )
            elif "poly" in name.lower() or "poli" in name.lower():
                ct = ColumnTransformer(
                    [("scale", RobustScaler(), slice(0, None))],
                    remainder="passthrough",
                )

            modified_model = clone(model)
            modified_model.set_params(regressor__scale=ct)
            modified_models.append(modified_model)

    return modified_models


def convert_to_space(caller, parameter):
    """
    Converts a hyperparameter from a YAML file to a Space object.

    Args:
        caller: The dictionary containing the hyperparameters.
        parameter: The key of the hyperparameter to be converted.

    Returns:
        A Space object representing the hyperparameter.

    Example:
        convert_to_space(cfg.opt.estimator, "learning_rate")
    """
    param_info = caller[parameter]
    name = param_info["name"]
    type = param_info["type"]

    if "low" in param_info:
        low = param_info["low"]
        high = param_info["high"]

        if type == "real":
            if "prior" in param_info:
                return Real(low, high, prior=param_info["prior"], name=name)
            else:
                return Real(low, high, name=name)

        else:
            return Integer(low, high, name=name)

    elif type == "scalers":
        return Categorical(
            [eval(i) for i in param_info["categories"]], name=name
        )

    else:
        return Categorical(param_info["categories"], name=name)


def change_nn_params(params):
    """
    Changes the parameters of a neural network regressor.

    Args:
        params: A dictionary of parameters for the regressor.

    Returns:
        The modified dictionary of parameters.

    Example:
        params = change_nn_params({"n_neurons_per_layer": 10, "n_hidden_layer": 2})
    """
    n_neurons = params["n_neurons_per_layer"]
    n_layers = params["n_hidden_layer"]

    params["regressor__reg__hidden_layer_sizes"] = (n_neurons,) * n_layers

    # the parameters are deleted to avoid an error from the MLPRegressor
    params.pop("n_neurons_per_layer")
    params.pop("n_hidden_layer")


def att_model(estimator, space, values, neural=False):
    """
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
    """
    params = {}

    for i in range(len(space)):
        params[f"{space[i].name}"] = values[i]
    if neural:
        change_nn_params(params)

    estimator.set_params(**params)

    return estimator


def gp_optimize(
    estim,
    x,
    y,
    space,
    cv,
    n_calls=150,
    n_random_starts=100,
    neural=False,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs_cv=4,
    n_jobs_opt=4,
    verbose=0,
):
    """
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
    """
    @use_named_args(space)
    def objective(**params):
        if neural:
            change_nn_params(params)

        estim.set_params(**params)

        mean_score = -np.mean(
            cross_val_score(
                estim,
                x,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs_cv,
                error_score=-100,
            )
        )

        if mean_score > 1000:
            mean_score = 100

        return mean_score

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=0,
        n_random_starts=n_random_starts,
        verbose=verbose,
        n_jobs=n_jobs_opt,
    )
    # print("Os hiper parâmetros que minimizam são: ", result.x)
    return result


def save_models(path: str, model):
    """
    Saves a model to a file.

    Args:
        path: The path to the file where the model will be saved.
        model: The model to be saved.

    Returns:
        None.

    Example:
        save_models(path="models/model.pkl", model=my_model)
    """
    pickle.dump(model, open(path, "wb"))


def opt_all(
    estimator_list,
    names_estimator_list,
    spaces_list,
    x,
    y,
    cv,
    path,
    verbose=0,
    print_models=False,
):
    """
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
    """
    list_of_models = []
    for j, k in enumerate(estimator_list):
        neural = False
        if verbose != 0:
            print(f"--- We're in the model {names_estimator_list[j]} ---")

        if "nn" in names_estimator_list[j]:
            neural = True

        opt = gp_optimize(
            k, x, y, spaces_list[j], cv=cv, neural=neural, verbose=verbose
        )
        best_model = att_model(k, spaces_list[j], opt.x, neural=neural)
        copy_of_best_model = copy.deepcopy(
            best_model
        )  # To ensure that we're not changing the original estimator
        trained_model = copy_of_best_model.fit(x, y)
        list_of_models.append(trained_model)

    stk_model = VotingRegressor(
        [(a, b) for a, b in zip(names_estimator_list, list_of_models)]
    )
    copy_of_stk = copy.deepcopy(stk_model)
    trained_stk = copy_of_stk.fit(x, y)  # The same as in line 152
    list_of_models.append(trained_stk)

    if print_models:
        for i in list_of_models:
            print(i)

    save_models(path, list_of_models)


def nested_cv(
    estimator,
    space,
    x,
    y,
    out_cv,
    inner_cv,
    scoring="neg_mean_absolute_percentage_error",
    neural=False,
    n_calls=150,
    n_random_starts=100,
    verbose=0,
    shuffle=True,
    random_state=0,
    print_mode=False,
):

    """
    A função retornará os scores de cada fold pra k_fold externo
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
    """

    kf_out = KFold(
        n_splits=out_cv, shuffle=shuffle, random_state=random_state
    )  # Folders externo.
    kf_in = KFold(
        n_splits=inner_cv, shuffle=shuffle, random_state=random_state
    )  # Folders internos.

    test_pred = np.zeros_like(y)
    test_error = np.zeros_like(y)

    best_models = []

    out_test_scores = []
    out_train_scores = []
    count = 1

    # Dividindo entre treino e teste na pasta de fora.
    for train_index, test_index in kf_out.split(x):
        print(f"Out folder {count} of {out_cv}.")
        count += 1
        x_train, x_test = x[list(train_index)], x[list(test_index)]
        y_train, y_test = y[train_index], y[test_index]

        # Otimizando o modelo pro x_train e y_train com validação cruzada interna.
        opt_model = gp_optimize(
            estimator,
            x_train,
            y_train,
            space=space,
            cv=kf_in,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            neural=neural,
            scoring=scoring,
            verbose=verbose,
        )

        # Pegando o melhor modelo com os valores de otimização
        att_model(estimator, space, opt_model.x, neural=neural)
        best_model = clone(estimator)
        best_models.append(best_model)
        # Fitando o melhor modelo
        fitted_model = best_model.fit(x_train, y_train)
        # Prevendo agora com o melhor modelo, a performance no dado de teste.
        y_train_pred = fitted_model.predict(x_train)
        y_pred = fitted_model.predict(x_test)

        # Mudando a métrica de acordo com scoring
        if scoring == "neg_mean_absolute_percentage_error":
            out_test_scores.append(
                mean_absolute_percentage_error(y_test, y_pred)
            )
            out_train_scores.append(
                mean_absolute_percentage_error(y_train, y_train_pred)
            )
        elif scoring == "neg_root_mean_squared_error":
            out_test_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            out_train_scores.append(
                np.sqrt(mean_squared_error(y_train, y_train_pred))
            )

        # Gerando o erro de teste índice a índice
        for i, z, w in zip(test_index, x_test, y_test):
            test_pred[i] = fitted_model.predict(z.reshape(1, -1))
            if scoring == "neg_mean_absolute_percentage_error":
                test_error[i] = mean_absolute_percentage_error(w, test_pred[i])
            elif scoring == "neg_root_mean_squared_error":
                test_error[i] = np.sqrt(mean_squared_error(w, test_pred[i]))

    if print_mode:
        clear_output()
        print("Cv Score:", np.mean(out_test_scores))
        print("Desvio Padrão:", round(np.std(out_test_scores), 2))
        print("Score de Treino:", np.mean(out_train_scores))

    return out_test_scores, out_train_scores, best_models


def stacked_nested_cv(
    estimators,
    estimators_names,
    spaces,
    x,
    y,
    out_cv,
    inner_cv,
    scoring="neg_mean_absolute_percentage_error",
    neural=False,
    n_calls=150,
    n_random_starts=100,
    verbose=0,
    shuffle=True,
    random_state=0,
):
    dict_train = {}
    dict_test = {}
    dict_best_models = {}

    for i, j in enumerate(estimators):
        actual_dict_key = estimators_names[i]
        print(f"--- We're in {actual_dict_key} model ---")
        dict_train[actual_dict_key] = []
        dict_test[actual_dict_key] = []
        dict_best_models[actual_dict_key] = []

        neural = False

        if "nn" in estimators_names[i].lower():
            neural = True

        out_test_scores, out_train_scores, best_models = nested_cv(
            j,
            spaces[i],
            x,
            y,
            out_cv,
            inner_cv,
            scoring=scoring,
            neural=neural,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            verbose=verbose,
            shuffle=shuffle,
            random_state=random_state,
        )

        dict_train[actual_dict_key].extend(out_train_scores)
        dict_test[actual_dict_key].extend(out_test_scores)
        dict_best_models[actual_dict_key].extend(best_models)

    # Stacking part
    print("--- We're in stacked model ---")
    counting = 0
    kf_out = KFold(out_cv, shuffle=shuffle, random_state=random_state)
    dict_train["stacked"] = []
    dict_test["stacked"] = []
    for train_index, test_index in kf_out.split(x):
        print(f"Out folder {counting + 1} of {out_cv}.")
        x_train, x_test = x[list(train_index)], x[list(test_index)]
        y_train, y_test = y[train_index], y[test_index]

        list_of_estimators = []

        for j in dict_best_models.keys():
            list_of_estimators.append((j, dict_best_models[j][counting]))

        stacked_estimator = VotingRegressor(list_of_estimators)

        stacked_estimator.fit(x_train, y_train.ravel())

        y_train_pred = stacked_estimator.predict(x_train)
        y_test_pred = stacked_estimator.predict(x_test)

        if scoring == "neg_mean_absolute_percentage_error":
            dict_test["stacked"].append(
                mean_absolute_percentage_error(y_test, y_test_pred)
            )
            dict_train["stacked"].append(
                mean_absolute_percentage_error(y_train, y_train_pred)
            )
        elif scoring == "neg_root_mean_squared_error":
            dict_test["stacked"].append(
                np.sqrt(mean_squared_error(y_test, y_test_pred))
            )
            dict_train["stacked"].append(
                np.sqrt(mean_squared_error(y_train, y_train_pred))
            )

        counting += 1

    return dict_test, dict_train
