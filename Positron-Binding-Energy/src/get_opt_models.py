"""
Script to get the final optimized models
from all apolar molecules and partial apolar molecules.
"""

import hydra
import pandas as pd
from modifying_data import get_absolute_path
from omegaconf import DictConfig


def get_data(path: str):

    abs_path = get_absolute_path(path)
    df = pd.read_csv(abs_path)
    return df


@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    # df_all_apolar = get_data(cfg.data.apolar.processed.path)
    df_partial_apolar = get_data(cfg.data.apolar.final.path)

    print(df_partial_apolar.head())


if __name__ == "__main__":
    main()
