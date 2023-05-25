"""
Module to process raw data into data's that I want to utilize.
"""

import os
import warnings

import datasist as ds
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")


def log_transform(x):
    """
    Function to log transform some np.array
    """
    return np.log(x)


def log_transform_dataframe(dataframe, column_name):
    """
    dataframe = DataFrame to have the columns transformed
    column_name = List containing the name of the columns
    """
    dataframe = dataframe.copy()

    transformer = FunctionTransformer(log_transform)
    dataframe_transformed = transformer.fit_transform(dataframe[column_name])

    dataframe[column_name] = dataframe_transformed

    return dataframe


def get_absolute_path(file_path):
    """
    Takes the absolute path even with the change of
    directory due to hydra.
    """
    abs_path = os.path.abspath(
        os.path.join(hydra.utils.get_original_cwd(), file_path)
    )
    return abs_path


def make_processed_data(raw_path, processed_path, columns_to_drop):
    "Function to process my apolar/polar data"
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
    """
    Function to make the final data to be used in apolar/polar
    molecules
    """
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
    """
    Making processed and final data for all apolar molecules
    """
    make_processed_data(
        config.apolar.raw,
        config.apolar.processed.path,
        config.process.to_drop.apolar,
    )


@hydra.main(config_path="../config", config_name="main.yaml")
def make_aniso_apolar_data(config: DictConfig):
    """Making Processed and final data
    for partial molecules"""
    make_final_data(
        config.apolar.raw,
        config.apolar.final.path,
        config.process.to_drop.apolar,
    )


@hydra.main(config_path="../config", config_name="main.yaml")
def make_polar_data(config: DictConfig) -> pd.DataFrame:
    """
    Making Processed and final data for all
    polar molecules
    """
    make_processed_data(
        config.polar.raw,
        config.polar.processed.path,
        config.process.to_drop.polar,
    )


@hydra.main(config_path="../config", config_name="main.yaml")
def make_aniso_polar_data(config: DictConfig):
    """
    Making Processed and final data
    for partial polar molecules
    """
    make_final_data(
        config.polar.raw, config.polar.final.path, config.process.to_drop.polar
    )


if __name__ == "__main__":
    make_apolar_data()
    make_aniso_apolar_data()
    make_polar_data()
    make_aniso_polar_data()
