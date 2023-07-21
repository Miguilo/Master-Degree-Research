"""
Module to process raw data into data's that I want to utilize.
"""
import sys

sys.path.append("../src/")

import warnings

import hydra
import pandas as pd
from omegaconf import DictConfig
from utils.data import get_absolute_path, make_final_data, make_processed_data

warnings.filterwarnings("ignore")


@hydra.main(config_path="../config", config_name="main.yaml")
def make_apolar_data(config: DictConfig) -> pd.DataFrame:
    """"""
    make_processed_data(
        config.data.apolar.raw,
        config.data.apolar.processed.path,
        config.process.to_drop.apolar,
    )


@hydra.main(config_path="../config", config_name="main.yaml")
def make_aniso_apolar_data(config: DictConfig):
    """"""
    make_final_data(
        config.data.apolar.raw,
        config.data.apolar.final.path,
        config.process.to_drop.apolar,
    )


@hydra.main(config_path="../config", config_name="main.yaml")
def make_polar_data(config: DictConfig) -> pd.DataFrame:
    """"""
    make_processed_data(
        config.data.polar.raw,
        config.data.polar.processed.path,
        config.process.to_drop.polar,
    )


@hydra.main(config_path="../config", config_name="main.yaml")
def make_aniso_polar_data(config: DictConfig):
    """"""
    make_final_data(
        config.data.polar.raw,
        config.data.polar.final.path,
        config.process.to_drop.polar,
    )


@hydra.main(config_path="../config", config_name="main.yaml")
def make_polar_apolar_data(cfg: DictConfig) -> pd.DataFrame:
    """"""
    apol_path = get_absolute_path(cfg.data.apolar.processed.path)
    pol_path = get_absolute_path(cfg.data.polar.processed.path)
    pol_apol_path = get_absolute_path(cfg.data.polar_apolar.processed.path)

    df_apol = pd.read_csv(apol_path)
    df_pol = pd.read_csv(pol_path)

    polar_apolar_data = pd.concat([df_apol, df_pol], axis=0, ignore_index=True)
    polar_apolar_data["Dipole"] = polar_apolar_data["Dipole"].fillna(0)

    polar_apolar_data.to_csv(pol_apol_path, index=False)


@hydra.main(config_path="../config", config_name="main.yaml")
def make_aniso_polar_apolar_data(cfg: DictConfig):

    apol_path = get_absolute_path(cfg.data.apolar.final.path)
    pol_path = get_absolute_path(cfg.data.polar.final.path)
    pol_apol_path = get_absolute_path(cfg.data.polar_apolar.final.path)

    df_apol = pd.read_csv(apol_path)
    df_pol = pd.read_csv(pol_path)

    polar_apolar_data = pd.concat([df_apol, df_pol], axis=0, ignore_index=True)
    polar_apolar_data["Dipole"] = polar_apolar_data["Dipole"].fillna(0)

    polar_apolar_data.to_csv(pol_apol_path, index=False)


if __name__ == "__main__":
    make_apolar_data()
    make_aniso_apolar_data()
    make_polar_data()
    make_aniso_polar_data()
    make_aniso_polar_apolar_data()
