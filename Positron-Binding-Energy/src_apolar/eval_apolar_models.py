import sys
sys.path.append('../')

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
from utils.optimization import convert_to_space, opt_all, modify_scaling, stacked_nested_cv, nested_cv
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

    # For all apolar molecules

    df_apolar_all = pd.read_csv(
        get_absolute_path(cfg.data.apolar.processed.path)
    )

    x0_all = df_apolar_all[cfg.opt.features.all.apolar.feat1].values
    x1_all = df_apolar_all[cfg.opt.features.all.apolar.feat2].values
    x2_all = df_apolar_all[cfg.opt.features.all.apolar.feat3].values
    x3_all = df_apolar_all[cfg.opt.features.all.apolar.feat4].values

    y_all = df_apolar_all[["Expt"]].values

    list_of_x_all = [x0_all, x1_all, x2_all, x3_all]
    list_of_models = [svr, ridge, xgb, nn]
    list_of_spaces = [space_svr, space_poly, space_xgb,space_nn  ]
    list_of_models_names = ["svr", "poly", "xgb","nn" ]
    list_of_features = ["All", "Ei + Alpha", "Pi + Ei", "Pi + Alpha"]
    
    initial_t = datetime.now()
    rows = list_of_models_names.copy()
    rows.append('stacked')

    all_train_score_df = create_df(list_of_features, rows)
    all_test_score_df = create_df(list_of_features, rows)
    all_train_std_df = create_df(list_of_features, rows)
    all_test_std_df = create_df(list_of_features, rows)

    all_train_score_df_path = get_absolute_path(cfg.eval.all.apolar.train_score_path)
    all_test_score_df_path = get_absolute_path(cfg.eval.all.apolar.test_score_path)
    all_train_std_df_path = get_absolute_path(cfg.eval.all.apolar.train_std_path)
    all_test_std_df_path = get_absolute_path(cfg.eval.all.apolar.test_std_path)

    for i, j in enumerate(list_of_x_all):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(list_of_models, list_of_models_names, list_of_features[i])
        
        dict_test, dict_train = stacked_nested_cv(new_list_of_models, list_of_models_names, list_of_spaces,
                                                  j, y_all, 10, 5)

           
        for k in dict_test.keys():
            all_train_score_df[list_of_features[i]][k] = np.mean(dict_train[k])
            all_test_score_df[list_of_features[i]][k] = np.mean(dict_test[k])
            all_train_std_df[list_of_features[i]][k] = np.std(dict_train[k])
            all_test_std_df[list_of_features[i]][k] = np.std(dict_test[k])

        all_train_score_df.to_csv(all_train_score_df_path, index_label=False)
        all_test_score_df.to_csv(all_test_score_df_path, index_label=False)  
        all_train_std_df.to_csv(all_train_std_df_path, index_label=False)  
        all_train_std_df.to_csv(all_test_std_df_path, index_label=False)

    # For partial molecules
    df_apolar_partial = pd.read_csv(
        get_absolute_path(cfg.data.apolar.final.path)
    )

    x0_partial_iso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat1_iso
    ].values
    x1_partial_iso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat2_iso
    ].values
    x2_partial_iso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat3_iso
    ].values
    x3_partial_iso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat4_iso
    ].values

    x0_partial_aniso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat1_aniso
    ].values
    x1_partial_aniso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat2_aniso
    ].values
    x2_partial_aniso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat3_aniso
    ].values
    x3_partial_aniso = df_apolar_partial[
        cfg.opt.features.partial.apolar.feat4_aniso
    ].values

    y_partial = df_apolar_partial[["Expt"]].values

    list_of_x_partial_iso = [
        x0_partial_iso,
        x1_partial_iso,
        x2_partial_iso,
        x3_partial_iso,
    ]
    list_of_x_partial_aniso = [
        x0_partial_aniso,
        x1_partial_aniso,
        x2_partial_aniso,
        x3_partial_aniso,
    ]          

    partial_iso_train_score_df = create_df(list_of_features, rows)
    partial_iso_test_score_df = create_df(list_of_features, rows)
    partial_iso_train_std_df = create_df(list_of_features, rows)
    partial_iso_test_std_df = create_df(list_of_features, rows)

    partial_iso_train_score_df_path = get_absolute_path(cfg.eval.partial_iso.apolar.train_score_path)
    partial_iso_test_score_df_path = get_absolute_path(cfg.eval.partial_iso.apolar.test_score_path)
    partial_iso_train_std_df_path = get_absolute_path(cfg.eval.partial_iso.apolar.train_std_path)
    partial_iso_test_std_df_path = get_absolute_path(cfg.eval.partial_iso.apolar.test_std_path)

    for i, j in enumerate(list_of_x_partial_iso):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(list_of_models, list_of_models_names, list_of_features[i])
        
        dict_test, dict_train = stacked_nested_cv(new_list_of_models, list_of_models_names, list_of_spaces,
                                                  j, y_partial, 10, 5)
        
        for k in dict_test.keys():
            partial_iso_train_score_df[list_of_features[i]][k] = np.mean(dict_train[k])
            partial_iso_test_score_df[list_of_features[i]][k] = np.mean(dict_test[k])
            partial_iso_train_std_df[list_of_features[i]][k] = np.std(dict_train[k])
            partial_iso_test_std_df[list_of_features[i]][k] = np.std(dict_test[k])

        partial_iso_train_score_df.to_csv(partial_iso_train_score_df_path, index_label=False)
        partial_iso_test_score_df.to_csv(partial_iso_test_score_df_path, index_label=False)  
        partial_iso_train_std_df.to_csv(partial_iso_train_std_df_path, index_label=False)  
        partial_iso_train_std_df.to_csv(partial_iso_test_std_df_path, index_label=False)

    # For partial aniso

    partial_aniso_train_score_df = create_df(list_of_features, rows)
    partial_aniso_test_score_df = create_df(list_of_features, rows)
    partial_aniso_train_std_df = create_df(list_of_features, rows)
    partial_aniso_test_std_df = create_df(list_of_features, rows)

    partial_aniso_train_score_df_path = get_absolute_path(cfg.eval.partial_aniso.apolar.train_score_path)
    partial_aniso_test_score_df_path = get_absolute_path(cfg.eval.partial_aniso.apolar.test_score_path)
    partial_aniso_train_std_df_path = get_absolute_path(cfg.eval.partial_aniso.apolar.train_std_path)
    partial_aniso_test_std_df_path = get_absolute_path(cfg.eval.partial_aniso.apolar.test_std_path)

    for i, j in enumerate(list_of_x_partial_aniso):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(list_of_models, list_of_models_names, list_of_features[i])
        
        dict_test, dict_train = stacked_nested_cv(new_list_of_models, list_of_models_names, list_of_spaces,
                                                  j, y_partial, 10, 5)
        
        for k in dict_test.keys():
            partial_aniso_train_score_df[list_of_features[i]][k] = np.mean(dict_train[k])
            partial_aniso_test_score_df[list_of_features[i]][k] = np.mean(dict_test[k])
            partial_aniso_train_std_df[list_of_features[i]][k] = np.std(dict_train[k])
            partial_aniso_test_std_df[list_of_features[i]][k] = np.std(dict_test[k])

        partial_aniso_train_score_df.to_csv(partial_aniso_train_score_df_path, index_label=False)
        partial_aniso_test_score_df.to_csv(partial_aniso_test_score_df_path, index_label=False)  
        partial_aniso_train_std_df.to_csv(partial_aniso_train_std_df_path, index_label=False)  
        partial_aniso_train_std_df.to_csv(partial_aniso_test_std_df_path, index_label=False)

    final_t = datetime.now()
    execution_t = final_t - initial_t
    print("Date of execution: ", initial_t)
    print("Date of finalization: ", final_t)
    print("Time elapsed: ", execution_t)

        
if __name__ == "__main__":
    main()
