import sys
sys.path.append('../src/')

from copy import deepcopy
from datetime import datetime

import numpy as np
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import TransformedTargetRegressor
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (RobustScaler, PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVR
from utils.data import get_absolute_path, create_df
from utils.optimization import convert_to_space, opt_all, modify_scaling, stacked_nested_cv
from utils.evaluation import show_metrics
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer

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

    xgb = XGBRegressor(random_state=0)

    pipe_nn = Pipeline(
        [
            ("scale", RobustScaler()),
            (
                "reg",
                MLPRegressor(max_iter=5000, random_state=0, solver="adam"),
            ),
        ]
    )
    nn = TransformedTargetRegressor(
        regressor=pipe_nn, transformer=StandardScaler()
    )

    # For partial polar_apolar molecules

    df_polar_apolar_partial = pd.read_csv(
        get_absolute_path(cfg.data.polar_apolar.final.path)
    )
    print(df_polar_apolar_partial)
    x0_all = df_polar_apolar_partial[cfg.opt.features.partial.polar_apolar.feat1_aniso].values

    y_all = df_polar_apolar_partial[["Expt"]].values

    list_of_x_all = [x0_all]
    list_of_models = [ridge, svr, xgb, nn]
    list_of_spaces = [space_poly, space_svr, space_xgb, space_nn]
    list_of_models_names = ["poly", "svr", "xgb", "nn"]
    list_of_features = [
        "Ei + Alpha + Dipole + Pi"
    ]


    initial_t = datetime.now()
    
    pred_df = df_polar_apolar_partial[['Molecule', 'Formula', 'Expt']].copy()
    
    pred_path = get_absolute_path(cfg.eval.all.polar_apolar.dir)

    for i, j in enumerate(list_of_x_all):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(list_of_models, list_of_models_names, list_of_features[i])
        
        predictions = stacked_nested_cv(new_list_of_models, list_of_models_names,list_of_spaces,
                                                  j, y_all, 3, 2, show_individual_test_pred = True)

           
        pred_df["Stacked Pred's"] = np.int32(np.round(predictions))

        pred_df.to_csv(f"{pred_path}/test_predictions.csv", index=False)



    final_t = datetime.now()
    execution_t = final_t - initial_t
    print("Date of execution: ", initial_t)
    print("Date of finalization: ", final_t)
    print("Time elapsed: ", execution_t)

if __name__ == "__main__":
    main()