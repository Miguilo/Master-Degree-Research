import sys
from os import path

file_dir = path.dirname(__file__)

sys.path.insert(1, path.join(file_dir, "../src/"))

from copy import deepcopy
from datetime import datetime

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn import linear_model
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (PolynomialFeatures, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVR
from xgboost import XGBRegressor

from utils.data import create_df, get_absolute_path
from utils.evaluation import show_metrics
from utils.optimization import (convert_to_space, modify_scaling, opt_all,
                                stacked_nested_cv, test_pred_nested_cv)


@hydra.main(
    config_path=path.join(file_dir, "../config"), config_name="main.yaml"
)
def main(cfg: DictConfig):
    call_reg_scaler = cfg.opt
    call_transformer_scaler = cfg.opt

    call_svr = cfg.opt.svr
    call_poly = cfg.opt.poly

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

    pipe_elastic = Pipeline(
        [
            ("poli", PolynomialFeatures()),
            ("scale", RobustScaler()),
            ("reg", linear_model.Ridge(max_iter=30000, random_state=0)),
        ]
    )
    ridge = TransformedTargetRegressor(
        regressor=pipe_elastic, transformer=StandardScaler()
    )

    pipe_svr = Pipeline(
        [("scale", RobustScaler()), ("reg", SVR(max_iter=10000))]
    )
    svr = TransformedTargetRegressor(
        transformer=StandardScaler(), regressor=pipe_svr
    )

    # For all apolar molecules

    df_apolar_all = pd.read_csv(
        get_absolute_path(cfg.data.apolar.processed.path)
    )

    pi_alpha_x = df_apolar_all[cfg.opt.features.all.apolar.feat4].values

    y_all = df_apolar_all[["Expt"]].values

    list_of_x_all = [pi_alpha_x]
    list_of_models = [svr, ridge]
    list_of_spaces = [space_svr, space_poly]
    list_of_models_names = ["svr", "poly"]
    list_of_features = ["Pi + Alpha"]

    initial_t = datetime.now()

    pred_df = df_apolar_all[["Molecule", "Formula", "Expt"]].copy()

    pred_path = get_absolute_path(cfg.eval.all.apolar.dir)

    for i, j in enumerate(list_of_x_all):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(
            list_of_models, list_of_models_names, list_of_features[i]
        )

        predictions_svr = test_pred_nested_cv(
            new_list_of_models[0], list_of_spaces[0], j, y_all, 10, 5
        )

        predictions_poly = test_pred_nested_cv(
            new_list_of_models[1], list_of_spaces[1], j, y_all, 10, 5
        )

        predictions = stacked_nested_cv(
            new_list_of_models,
            list_of_models_names,
            list_of_spaces,
            j,
            y_all,
            10,
            5,
            show_individual_test_pred=True,
        )

        pred_df["Stacked Pred's"] = np.int32(np.round(predictions))
        pred_df["SVR Pred's"] = np.int32(np.round(predictions_svr))
        pred_df["Poly Pred's"] = np.int32(np.round(predictions_poly))

        pred_df.to_csv(f"{pred_path}/test_predictions.csv", index=False)

    final_t = datetime.now()
    execution_t = final_t - initial_t
    print("Date of execution: ", initial_t)
    print("Date of finalization: ", final_t)
    print("Time elapsed: ", execution_t)


if __name__ == "__main__":
    main()
