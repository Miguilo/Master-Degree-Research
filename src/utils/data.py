import inspect
import os

import datasist as ds
import hydra
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def log_transform(x):
    return np.log(x)


def log_transform_dataframe(dataframe, column_name):
    dataframe = dataframe.copy()

    transformer = FunctionTransformer(log_transform)
    dataframe_transformed = transformer.fit_transform(dataframe[column_name])

    dataframe[column_name] = dataframe_transformed

    return dataframe


def get_absolute_path(file_path):
    """
    Get absolute path relative from CALLER
    """
    var = os.path.dirname(inspect.stack()[1].filename)
    abs_path = os.path.abspath(
        os.path.join(os.path.dirname(inspect.stack()[1].filename), file_path)
    )
    return abs_path


def make_processed_data(raw_path, processed_path, columns_to_drop):
    """
    Create a processed dataset by dropping missing values and columns.

    Args:
        raw_path: The path to the raw dataset.
        processed_path: The path where the processed dataset will be saved.
        columns_to_drop: The columns to be dropped.

    Returns:
        None.

    """
    abs_file_path = get_absolute_path(raw_path)
    print(abs_file_path)

    df = pd.read_csv(abs_file_path)  # Importing Data

    # Dropping the columns with missing values
    # (It'll be the ones without Anisotropic Polarizability)
    # Dropping the columns that we don't wanna to analyse here
    df.drop(
        columns_to_drop,
        axis=1,
        inplace=True,
    )
    df = ds.feature_engineering.drop_missing(df, percent=10)

    new_path = get_absolute_path(processed_path)
    df.to_csv(new_path, index=False)


def make_final_data(raw_path, final_path, columns_to_drop):
    """
    Create a final dataset by dropping rows, columns, and rounding the values.

    Args:
        raw_path: The path to the raw dataset.
        final_path: The path where the final dataset will be saved.
        columns_to_drop: The columns to be dropped.

    Returns:
        None.

    """
    abs_file_path = get_absolute_path(raw_path)

    df = pd.read_csv(abs_file_path)
    # Dropping rows
    molecules_to_drop = df.loc[np.isnan(df["axx"])].index
    df.drop(molecules_to_drop, inplace=True)
    try:
        indexes_to_drop = df.query(
            "Molecule == '2-2-Difluoropropane' or Molecule == '1-1-Dichloroethylene'"
        ).index
        df.drop(indexes_to_drop, inplace=True).reset_index(
            drop=True, inplace=True
        )
    except:
        pass
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


def get_data(path: str):
    abs_path = get_absolute_path(path)
    df = pd.read_csv(abs_path)
    return df


def create_df(column_names, row_names):
    dict = {}
    for i in column_names:
        dict[i] = np.full((len(row_names)), np.nan)
    df = pd.DataFrame(dict, index=row_names)
    return df
