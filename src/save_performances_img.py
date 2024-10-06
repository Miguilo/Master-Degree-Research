import sys

sys.path.append("../src/")

import pickle

import hydra
import pandas as pd
from omegaconf import DictConfig

from utils.data import get_absolute_path
from utils.evaluation import (create_graph_shap, plot_all_fast_graphs,
                              show_metrics)


@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    # Importing performance DF's

    list_of_img_names = [
        "all_test_score.png",
        "partial_iso_test_score.png",
        "partial_aniso_test_score.png",
    ]

    # Apolar
    df_test_scores_apolar_all = pd.read_csv(
        get_absolute_path(cfg.eval.all.apolar.test_score_path)
    )
    df_test_scores_apolar_partial_iso = pd.read_csv(
        get_absolute_path(cfg.eval.partial_iso.apolar.test_score_path)
    )
    df_test_scores_apolar_partial_aniso = pd.read_csv(
        get_absolute_path(cfg.eval.partial_aniso.apolar.test_score_path)
    )

    list_of_df_apolar = [
        df_test_scores_apolar_all,
        df_test_scores_apolar_partial_iso,
        df_test_scores_apolar_partial_aniso,
    ]

    # Polar
    df_test_scores_polar_all = pd.read_csv(
        get_absolute_path(cfg.eval.all.polar.test_score_path)
    )
    df_test_scores_polar_partial_iso = pd.read_csv(
        get_absolute_path(cfg.eval.partial_iso.polar.test_score_path)
    )
    df_test_scores_polar_partial_aniso = pd.read_csv(
        get_absolute_path(cfg.eval.partial_aniso.polar.test_score_path)
    )

    list_of_df_polar = [
        df_test_scores_polar_all,
        df_test_scores_polar_partial_iso,
        df_test_scores_polar_partial_aniso,
    ]

    # Polar + Apolar
    df_test_scores_polar_apolar_all = pd.read_csv(
        get_absolute_path(cfg.eval.all.polar_apolar.test_score_path)
    )
    df_test_scores_polar_apolar_partial_iso = pd.read_csv(
        get_absolute_path(cfg.eval.partial_iso.polar_apolar.test_score_path)
    )
    df_test_scores_polar_apolar_partial_aniso = pd.read_csv(
        get_absolute_path(cfg.eval.partial_aniso.polar_apolar.test_score_path)
    )

    list_of_df_polar_apolar = [
        df_test_scores_polar_apolar_all,
        df_test_scores_polar_apolar_partial_iso,
        df_test_scores_polar_apolar_partial_aniso,
    ]

    plot_all_fast_graphs(
        list_of_df_apolar,
        get_absolute_path(cfg.eval.imgs.apolar),
        list_of_img_names,
    )
    plot_all_fast_graphs(
        list_of_df_polar,
        get_absolute_path(cfg.eval.imgs.polar),
        list_of_img_names,
    )
    plot_all_fast_graphs(
        list_of_df_polar_apolar,
        get_absolute_path(cfg.eval.imgs.polar_apolar),
        list_of_img_names,
    )


if __name__ == "__main__":
    main()
