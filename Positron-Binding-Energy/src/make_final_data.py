"""
Module to process raw data into data's that I want to utilize.
"""

import datasist as ds
import numpy as np
import pandas as pd
import warnings
import hydra
import os
from omegaconf import DictConfig

warnings.filterwarnings("ignore")

def get_absolute_path(file_path):
    """
    Takes the absolute path even with the change of
    directory due to hydra.
    """
    abs_path = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), 
                                            file_path))
    return abs_path

def make_processed_data(raw_path, processed_path, columns_to_drop):
    abs_file_path = get_absolute_path(raw_path)
    print(abs_file_path)

    df = pd.read_csv(abs_file_path)  # Importing Data

    # Dropping the columns with missing values
    # (It'll be the ones without Anisotropic Polarizability)
    if all:
        # Dropping the columns that we don't wanna to analyse here
        df.drop(
            columns_to_drop,
            axis=1,
            inplace=True,
        )
        df = ds.feature_engineering.drop_missing(df, percent=10)
    
    else:
        raise TypeError("Only boolean are allowed")
    
    new_path = get_absolute_path(processed_path)
    df.to_csv(new_path, index=False)

def make_final_data(raw_path, final_path, columns_to_drop):

    abs_file_path = get_absolute_path(raw_path)

    df = pd.read_csv(abs_file_path)
    # Dropping rows
    molecules_to_drop = df.loc[np.isnan(df["axx"])].index
    df.drop(molecules_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Dropping and Rearranging columns
    df["Alpha"] = np.round(
        df["AlphaB"], 2
    )  # Changing Alpha per AlphaB(Mean of Anisotropic Polarizability)
    df[["axx", "ayy", "azz"]] = np.round(
        df[["axx", "ayy", "azz"]], 2
    )  # Rounding to get the visualization better

    df.drop(
        columns_to_drop,
        axis=1,
        inplace=True,
    )
    new_path = get_absolute_path(final_path)
    df.to_csv(new_path, index=False)

@hydra.main(config_path="../config", config_name="main.yaml")
def make_apolar_data(config: DictConfig) -> pd.DataFrame:

    make_processed_data(config.apolar.raw, config.apolar.processed.path,
                        config.process.to_drop.apolar)
   

@hydra.main(config_path="../config", config_name="main.yaml")
def make_aniso_apolar_data(config: DictConfig):
    
    make_final_data(config.apolar.raw, config.apolar.final.path,
                        config.process.to_drop.apolar)
    
@hydra.main(config_path="../config", config_name="main.yaml")
def make_polar_data(config: DictConfig) -> pd.DataFrame:

    make_processed_data(config.polar.raw, config.polar.processed.path,
                        config.process.to_drop.polar)

@hydra.main(config_path="../config", config_name="main.yaml")
def make_aniso_polar_data(config: DictConfig):
    
    make_final_data(config.polar.raw, config.polar.final.path,
                        config.process.to_drop.polar)
if __name__ == "__main__":
    make_apolar_data()
    make_aniso_apolar_data()
    make_polar_data()
    make_aniso_polar_data()