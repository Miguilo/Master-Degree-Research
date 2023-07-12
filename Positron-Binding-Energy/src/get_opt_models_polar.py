from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVR
from utils.data import get_absolute_path
from utils.evaluation import show_metrics
from utils.optimization import convert_to_space, opt_all, modify_scaling
from xgboost import XGBRegressor


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
            ("scale", MinMaxScaler()),
            ("reg", Ridge(max_iter=30000)),
        ]
    )
    ridge = TransformedTargetRegressor(
        regressor=pipe_elastic, transformer=StandardScaler()
    )

    pipe_svr = Pipeline(
        [("scale", StandardScaler()), ("reg", SVR(max_iter=10000))]
    )
    svr = TransformedTargetRegressor(
        transformer=StandardScaler(), regressor=pipe_svr
    )

    xgb = XGBRegressor(random_state=0)

    pipe_nn = Pipeline(
        [
            ("scale", MinMaxScaler()),
            (
                "reg",
                MLPRegressor(max_iter=5000, random_state=0, solver="adam"),
            ),
        ]
    )
    nn = TransformedTargetRegressor(
        regressor=pipe_nn, transformer=StandardScaler()
    )

    # For all polar molecules

    df_polar_all = pd.read_csv(
        get_absolute_path(cfg.data.polar.processed.path)
    )

    x0_all = df_polar_all[cfg.opt.features.all.polar.feat1].values
    x1_all = df_polar_all[cfg.opt.features.all.polar.feat2].values
    x2_all = df_polar_all[cfg.opt.features.all.polar.feat3].values
    x3_all = df_polar_all[cfg.opt.features.all.polar.feat4].values

    y_all = df_polar_all[["Expt"]].values

    list_of_x_all = [x0_all, x1_all, x2_all, x3_all]
    list_of_models = [ridge, svr, xgb, nn]
    list_of_spaces = [space_poly, space_svr, space_xgb, space_nn]
    list_of_models_names = ["poly", "svr", "xgb", "nn"]
    list_of_features = [
        "All",
        "Ei + Alpha + Dipole",
        "Alpha + Dipole",
        "Alpha + Dipole + Pi",
    ]
    list_of_paths = [
        get_absolute_path(cfg.models.polar["all"]),
        get_absolute_path(cfg.models.polar["ei_alpha_dipole"]),
        get_absolute_path(cfg.models.polar["alpha_dipole"]),
        get_absolute_path(cfg.models.polar["alpha_dipole_pi"]),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    initial_t = datetime.now()
    for i, j in enumerate(list_of_x_all):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(list_of_models, list_of_models_names, list_of_features[i])
        opt_all(
            new_list_of_models,
            list_of_models_names,
            list_of_spaces,
            j,
            y_all.ravel(),
            cv=kf,
            path=f"{list_of_paths[i]}/all_molecules_models.sav",
            verbose=1,
        )

    # For partial molecules
    df_polar_partial = pd.read_csv(
        get_absolute_path(cfg.data.polar.final.path)
    )

    x0_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat1_iso
    ].values
    x1_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat2_iso
    ].values
    x2_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat3_iso
    ].values
    x3_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat4_iso
    ].values

    x0_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat1_aniso
    ].values
    x1_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat2_aniso
    ].values
    x2_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat3_aniso
    ].values
    x3_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat4_aniso
    ].values

    y_partial = df_polar_partial[["Expt"]].values

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

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    # For Isotropic

    for i, j in enumerate(list_of_x_partial_iso):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(list_of_models, list_of_models_names, list_of_features[i])
        opt_all(
            new_list_of_models,
            list_of_models_names,
            list_of_spaces,
            j,
            y_partial.ravel(),
            cv=kf,
            path=f"{list_of_paths[i]}/partial_iso_molecules_models.sav",
            verbose=1,
        )

    # For Anisotropic Polarizability

    for i, j in enumerate(list_of_x_partial_aniso):
        print(f"=== {list_of_features[i]} Features ===")
        new_list_of_models = modify_scaling(list_of_models, list_of_models_names, list_of_features[i])
        opt_all(
            new_list_of_models,
            list_of_models_names,
            list_of_spaces,
            j,
            y_partial.ravel(),
            cv=kf,
            path=f"{list_of_paths[i]}/partial_aniso_molecules_models.sav",
            verbose=1,
        )
    final_t = datetime.now()
    execution_t = final_t - initial_t
    print("Date of execution: ", initial_t)
    print("Date of finalization: ", final_t)
    print("Time elapsed: ", execution_t)


if __name__ == "__main__":
    main()
