"""
Script to get the final optimized models
from all apolar molecules and partial apolar molecules.
"""

import hydra
import numpy as np
import pandas as pd
from modifying_data import get_absolute_path
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skopt.space.space import Categorical, Integer, Real


def get_data(path: str):
    abs_path = get_absolute_path(path)
    df = pd.read_csv(abs_path)
    return df


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


@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    call_reg_scaler = cfg.opt
    call_transformer_scaler = cfg.opt

    call_svr = cfg.opt.svr
    call_poly = cfg.opt.poly
    call_nn = cfg.opt.nn
    call_xgb = cfg.opt.xgb

    space_svr = [
        convert_to_space(call_svr, "kernel"),
        convert_to_space(call_svr, "gamma"),
        convert_to_space(call_svr, "degree"),
        convert_to_space(call_svr, "coef0"),
        convert_to_space(call_svr, "C"),
        convert_to_space(call_svr, "epsilon"),
        convert_to_space(call_reg_scaler, "scaling_regressor"),
        convert_to_space(call_transformer_scaler, "scaling_transformer"),
    ]
    space_poly = [
        convert_to_space(call_poly, "alpha"),
        convert_to_space(call_poly, "fit_intercept"),
        convert_to_space(call_poly, "copy_X"),
        convert_to_space(call_poly, "degree"),
        convert_to_space(call_poly, "interaction_only"),
        convert_to_space(call_poly, "include_bias"),
        convert_to_space(call_reg_scaler, "scaling_regressor"),
        convert_to_space(call_transformer_scaler, "scaling_transformer"),
    ]

    space_nn = [
        convert_to_space(call_nn, "activation"),
        convert_to_space(call_nn, "learning_rate_init"),
        convert_to_space(call_nn, "n_hidden_layer"),
        convert_to_space(call_nn, "n_neurons_per_layer"),
        convert_to_space(call_nn, "beta_1"),
        convert_to_space(call_nn, "beta_2"),
        convert_to_space(call_nn, "epsilon"),
        convert_to_space(call_reg_scaler, "scaling_regressor"),
        convert_to_space(call_transformer_scaler, "scaling_transformer"),
    ]

    space_xgb = [
        convert_to_space(call_xgb, "learning_rate"),
        convert_to_space(call_xgb, "n_estimators"),
        convert_to_space(call_xgb, "max_depth"),
        convert_to_space(call_xgb, "min_child_weight"),
        convert_to_space(call_xgb, "gamma"),
        convert_to_space(call_xgb, "subsample"),
        convert_to_space(call_xgb, "colsample_bytree"),
        convert_to_space(call_xgb, "reg_alpha"),
        convert_to_space(call_xgb, "reg_lambda"),
    ]


if __name__ == "__main__":
    main()
