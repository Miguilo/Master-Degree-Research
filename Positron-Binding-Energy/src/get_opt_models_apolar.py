"""
Script to get the final optimized models
from all apolar molecules and partial apolar molecules.
"""

import hydra
import pandas as pd
from modifying_data import get_absolute_path
from omegaconf import DictConfig
from skopt.space.space import Real


def get_data(path: str):
    abs_path = get_absolute_path(path)
    df = pd.read_csv(abs_path)
    return df


@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    # df_all_apolar = get_data(cfg.data.apolar.processed.path)
    space_poli = [Real(1e-5, 100, "log-uniform", name="regressor__reg__alpha")]

    space_poli_test = [
        Real(
            cfg.optimizing.poly_space.space.alpha.low,
            cfg.optimizing.poly_space.space.alpha.high,
            prior=cfg.optimizing.poly_space.space.alpha.prior,
            name=cfg.optimizing.poly_space.space.alpha.name,
        )
    ]

    print(space_poli == space_poli_test)


if __name__ == "__main__":
    main()
