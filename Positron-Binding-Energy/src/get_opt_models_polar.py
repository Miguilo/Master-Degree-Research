import pickle

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
from utils.optimization import convert_to_space, opt_all
from xgboost import XGBRegressor


@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    estims = pickle.load(
        open(
            "/home/miguel/Documentos/Master-Degree-Research/Positron-Binding-Energy/models/apolar/all/all_molecules_models.sav",
            "rb",
        )
    )

    df_apol = pd.read_csv(get_absolute_path(cfg.data.apolar.processed.path))
    print(df_apol)

    x_all = df_apol[["Ei", "Alpha", "pi_bond"]].values
    y_all = df_apol[["Expt"]].values

    kf = KFold(10, shuffle=True, random_state=0)

    show_metrics(estims[2], x_all, y_all, cv=kf)


if __name__ == "__main__":
    main()
