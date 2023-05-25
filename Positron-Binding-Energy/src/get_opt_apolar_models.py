"""
Script to get the final optimized models
from all apolar molecules and partial apolar molecules.
"""

import os

import hydra
import pandas as pd
from modifying_data import get_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main.yaml")
def get_full_data(cfg: DictConfig):
    """Function to get all the molecules
    for Apolar Molecules
    """
    cwd = os.getcwd()
    print(cwd)
    abs_path = get_absolute_path(cfg.apolar.processed.path)
    print(abs_path)
    df_full = pd.read_csv(abs_path)
    print(df_full.head())

    return df_full


df = get_full_data()
print(df)
