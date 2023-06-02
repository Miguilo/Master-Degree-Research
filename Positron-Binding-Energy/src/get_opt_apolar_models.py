"""
Script to get the final optimized models
from all apolar molecules and partial apolar molecules.
"""

import hydra
import pandas as pd
from modifying_data import get_absolute_path
from omegaconf import DictConfig


def get_full_data(path: str):
    """Function to get all the molecules
    for Apolar Molecules
    """
    abs_path = get_absolute_path(path)
    df = pd.read_csv(abs_path)
    return df


@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    """
    Main function to be initialized in this script
    """
    # df = get_full_data(cfg.apolar.processed.path)
    test_path = get_absolute_path("../models")
    print(test_path)


if __name__ == "__main__":
    main()
