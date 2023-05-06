"""
Module to process raw data into data's that I want to utilize.
"""

import datasist as ds
import numpy as np
import pandas as pd


def make_apolar_data(data_path: str, all: bool = True) -> pd.DataFrame:
    """
    Create Apolar Data with all molecules so we can use it.
    # Parameters
    - data_path: It can be relative or absolute path since it's
    read with pandas.read_csv
    - all: If you want to put all the molecules even the
    one's without anisotropy.

    # Returns
    Return an pd.DataFrame
    """
    df = pd.read_csv(data_path)  # Importing Data

    # Dropping the columns with missing values
    # (It'll be the ones without Anisotropic Polarizability)
    if all:
        # Dropping the columns that we don't wanna to analyse here
        df.drop(
            ["Molecule Type", "N of Carbon", "DYS", "SG", "PauloPred"],
            axis=1,
            inplace=True,
        )
        df = ds.feature_engineering.drop_missing(df, percent=10)
    elif not all:
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
            [
                "Molecule Type",
                "N of Carbon",
                "DYS",
                "SG",
                "PauloPred",
                "AlphaB",
            ],
            axis=1,
            inplace=True,
        )
    else:
        raise TypeError("Only boolean are allowed")

    return df
