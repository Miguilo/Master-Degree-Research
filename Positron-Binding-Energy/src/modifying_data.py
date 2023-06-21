"""
Module to process raw data into data's that I want to utilize.
"""

import warnings

import hydra
import pandas as pd
from omegaconf import DictConfig
from utils.data import make_final_data, make_processed_data

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


if __name__ == "__main__":
    make_apolar_data()
    make_aniso_apolar_data()
    make_polar_data()
    make_aniso_polar_data()
