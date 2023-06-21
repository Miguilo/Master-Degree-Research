import pickle
import warnings

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def convert_to_space(caller, parameter):
    """
    caller: Should have an form of cfg.opt.estimator
    parameter: Key of the hyperparameter in the way it's write
    in .yaml file.
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
    n_neurons = params["n_neurons_per_layer"]
    n_layers = params["n_hidden_layer"]

    params["regressor__reg__hidden_layer_sizes"] = (n_neurons,) * n_layers

    # the parameters are deleted to avoid an error from the MLPRegressor
    params.pop("n_neurons_per_layer")
    params.pop("n_hidden_layer")


def att_model(estimator, space, values, neural=False):
    """
    It is important to remember that in the case of neural networks,
    the name of the
    parameters associated with hidden_layers must be
    "n_neurons_per_layer"
    and "n_hidden_layer"
    """
    params = {}

    for i in range(len(space)):
        params[f"{space[i].name}"] = values[i]
    if neural:
        change_nn_params(params)

    estimator.set_params(**params)


def gp_optimize(
    estim,
    x,
    y,
    space,
    cv,
    n_calls=15,
    n_random_starts=10,
    neural=False,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs_cv=8,
    n_jobs_opt=8,
    verbose=0,
):
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

    gp_optimize.opt = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=0,
        n_random_starts=n_random_starts,
        verbose=verbose,
        n_jobs=n_jobs_opt,
    )
    # print("Os hiper parâmetros que minimizam são: ", gp_optimize.opt.x)
    return gp_optimize.opt


def save_models(path: str, model):
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
):
    list_of_models = []
    counting_j = 0
    for j, k in enumerate(estimator_list):
        neural = False
        if verbose != 0:
            print(f"--- We're in the model {names_estimator_list[j]} ---")
        if "nn" in names_estimator_list[j]:
            neural = True

        if counting_j == len(estimator_list) - 1:
            print("entered the loop you wished.")
            stk_model = VotingRegressor(
                [(a, b) for a, b in zip(names_estimator_list, list_of_models)]
            )
            list_of_models.append(stk_model)
            save_models(path, list_of_models)

        else:
            warnings.filterwarnings("ignore")
            opt = gp_optimize(k, x, y, spaces_list[j], cv=cv, neural=neural)
            best_model = att_model(k, spaces_list[j], opt.x, neural=neural)
            list_of_models.append(best_model)
            counting_j += 1
