import numpy as np
import pandas as pd

from . import utils

__all__ = ["backshift_idx_fields", "clean_0d_array_fields", "clean_integer_fields"]


@utils.copy_td
def backshift_idx_fields(trial_data: pd.DataFrame):
    """
    Adjust index fields from 1-based to 0-based indexing

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    Returns
    -------
    trial_data with the 'idx_' fields adjusted
    """
    idx_fields = [col for col in trial_data.columns.values if col.startswith("idx")]

    for col in idx_fields:
        # using a list comprehension to still work if the idx field itself is an array
        trial_data[col] = [idx - 1 for idx in trial_data[col]]

    return trial_data


@utils.copy_td
def clean_0d_array_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loading v7.3 MAT files, sometimes scalers are stored as 0-dimensional arrays for some reason.
    This converts those back to scalars.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    
    spike_fields = [name for name in df.columns.values if name.endswith("_spikes")]
    
    for c in spike_fields:
        df[c] = [arr 
                    if arr.ndim == 2
                    else arr.reshape(1,-1)
                    for arr in df[c]
                ]

    return df

@utils.copy_td
def clean_0d_spike_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    sometimes spike arrays are stored as 0-dimensional arrays when there is 1 time bin.
    This converts those back to 2D arrays.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    for c in df.columns:
        if all(isinstance(el, np.ndarray) for el in df[c].values):
            if all([arr.ndim == 0 for arr in df[c]]):
                df[c] = [arr.item() for arr in df[c]]

    return df


@utils.copy_td
def clean_integer_fields(df: pd.DataFrame):
    """
    Modify fields that store integers as floats to store them as integers instead.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    bad_fields = []
    for field in df.columns:
        try:
            if isinstance(df[field].values[0], np.ndarray):
                int_arrays = [np.int32(arr) for arr in df[field]]
                if all(
                    [
                        np.allclose(int_arr, arr)
                        for (int_arr, arr) in zip(int_arrays, df[field])
                    ]
                ):
                    df[field] = int_arrays
            else:
                if not isinstance(df[field].values[0], str):
                    int_version = np.int32(df[field])
                    if np.allclose(int_version, df[field]):
                        df[field] = int_version
        except:
            bad_fields.append(field)

    bad_fields = [field for field in bad_fields if 'label' not in field.lower()]
    if bad_fields:
        print(f"fields: {bad_fields} could not be converted to int.")

    return df
